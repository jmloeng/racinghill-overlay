"""Racinghill OVERLAY worker ("ghost run"): renders run B's skier over run A as one MP4. Google Cloud Run (EU).

POST /render  { jobId, supabaseUrl, anonKey, cdnHost, referer }      Authorization: Bearer <the user's Supabase token>
  The job row (overlay_jobs, made by api/overlay.js as the user) says which videos, the time offset, the start,
  (stats.max_seconds) how long the overlay may be — 20 or 40 s — and (stats.lines) whether to draw both race lines.
  kind "raceline" (no video_b): one video with the racer's race line (raceline.py), saved as RACELINE_<name>.mp4.
  kind "gates" (no video_b, beta v267): GATE TIMING — the coach's marked gates (stats.gates) → passing times + gate
  judge (gates.py), saved as GATES_<name>.mp4 with the numbers burned in; the result also goes on the marked video
  (videos.gates, as the user).
  kind "course" (no video_b, beta v272): SET COURSE — the session's course (public.courses: start / split / finish
  tapped once on the reference video = video_a) → pole feet on the reference video (courses.found), the same gates
  found in every video of stats.videos and the racer's times through them → videos.course on each (course.py). No
  video is made. Stops after COURSE_BUDGET_S (12 min: Cloud Run may stop an idle instance ~15 min after the last request); videos not reached stay "pending"
  with "more": true and the app asks again.
  Reading it with the user's token checks the login and that the job is theirs (RLS). Answers 202 at once, then works
  in the background, one render at a time per instance:
    download both Bunny MP4s (1080p, else 720p) → overlay.render() → upload the MP4 to Storage (bucket Videos,
    TUS, AS THE USER) → insert a videos row in A's session (as the user) → job done (result_video_id).
  Progress / errors are written to the job row as the user. RLS decides everything: a user who may not add videos
  to that session (locked team) gets a clear error. No secrets of its own; Supabase projects and Bunny CDN hosts
  must be on the allow lists (env ALLOWED_SUPABASE, ALLOWED_CDN), like the track worker (worker/server.js).
  Bunny ingest of the result is left to the app (the player asks /api/bunny "status" when the video is opened).
GET /health

Local test without login or upload:  python overlay.py … (see overlay.py), or
  python server.py --test <cdnHost> <referer> <guidA> <guidB> <offsetS> <startA> <out.mp4> [trackA.json trackB.json]
"""
import os
import re
import sys
import json
import time
import uuid
import base64
import shutil
import queue
import tempfile
import threading
import traceback
import unicodedata
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from overlay import render, OverlayError, MAX_SECONDS, HERE
from raceline import render_raceline
from gates import render_gates
import course as COURSE
from seg import Segmenter

PORT = int(os.environ.get("PORT", "8080"))
ALLOWED_SUPABASE = [s.strip() for s in os.environ.get("ALLOWED_SUPABASE", "").split(",") if s.strip()]
ALLOWED_CDN = [s.strip() for s in os.environ.get("ALLOWED_CDN", "").split(",") if s.strip()]
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
REFERER_RE = re.compile(r"^https://[a-z0-9.-]+/$", re.I)
CHUNK = 6 * 1024 * 1024
GENERIC_ERROR = "Something went wrong while making the video. Please try again."
COURSE_BUDGET_S = int(os.environ.get("COURSE_BUDGET_S") or 12 * 60)   # Cloud Run may stop an instance ~15 min after the last request


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), *a, flush=True)


# ───────────────────────── HTTP helpers ─────────────────────────

def http(method, url, headers=None, body=None, timeout=60):
    """(status, bytes) — never raises for HTTP errors"""
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    h = dict(headers or {})
    if isinstance(body, (dict, list)):
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers or {})


class Supa:
    """Supabase REST/Storage as the user (their token → RLS applies)."""

    def __init__(self, url, anon, token):
        self.url, self.anon, self.token = url, anon, token

    def h(self, extra=None):
        return {"apikey": self.anon, "Authorization": f"Bearer {self.token}", **(extra or {})}

    def get(self, path):
        s, b, _ = http("GET", f"{self.url}/rest/v1/{path}", self.h())
        return s, (json.loads(b) if s == 200 and b else None)

    def patch(self, path, body):
        s, b, _ = http("PATCH", f"{self.url}/rest/v1/{path}", self.h({"Prefer": "return=minimal"}), body)
        return s, b

    def post(self, path, body):
        s, b, _ = http("POST", f"{self.url}/rest/v1/{path}", self.h({"Prefer": "return=minimal"}), body)
        return s, b

    def user(self):
        s, b, _ = http("GET", f"{self.url}/auth/v1/user", self.h())
        return json.loads(b) if s == 200 else None

    def upload(self, path, file, on_progress=None):
        """TUS resumable upload to bucket Videos (like the app). Returns (ok, status)."""
        size = os.path.getsize(file)
        meta = ",".join(f"{k} {base64.b64encode(v.encode()).decode()}" for k, v in
                        (("bucketName", "Videos"), ("objectName", path), ("contentType", "video/mp4")))
        s, b, hd = http("POST", f"{self.url}/storage/v1/upload/resumable", self.h({
            "Tus-Resumable": "1.0.0", "Upload-Length": str(size), "Upload-Metadata": meta, "x-upsert": "false",
            "Content-Type": "application/octet-stream"}), b"")
        loc = next((v for k, v in hd.items() if k.lower() == "location"), None)
        if s not in (200, 201) or not loc:
            log("tus create", s, b[:200])
            return False, s
        off = 0
        with open(file, "rb") as f:
            while off < size:
                f.seek(off)
                chunk = f.read(CHUNK)
                for attempt in range(3):
                    s, b, hd = http("PATCH", loc, self.h({"Tus-Resumable": "1.0.0", "Upload-Offset": str(off),
                                                          "Content-Type": "application/offset+octet-stream"}), chunk, timeout=120)
                    if s in (200, 204):
                        break
                    log("tus patch", s, b[:200])
                    time.sleep(2 * (attempt + 1))
                else:
                    return False, s
                off += len(chunk)
                if on_progress:
                    on_progress(off / size)
        return True, 200

    def remove(self, path):
        s, _, _ = http("DELETE", f"{self.url}/storage/v1/object/Videos/{urllib.request.quote(path)}", self.h())
        return s


def download_bunny(cdn, guid, referer, dest):
    for q in ("1080p", "720p"):
        req = urllib.request.Request(f"https://{cdn}/{guid}/play_{q}.mp4", headers={"Referer": referer})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if os.path.getsize(dest) > 10000:
                return q
        except urllib.error.HTTPError as e:
            log("bunny", guid, q, e.code)
    return None


def safe_name(s, fallback):
    s = unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:40] or fallback


# ───────────────────────── jobs ─────────────────────────

jobs = queue.Queue()
state = {"running": 0}


def job_seconds(job):
    """How long the overlay may be: the user's choice (20 or 40 s, api/overlay.js puts it in stats.max_seconds),
    never more than overlay.MAX_SECONDS; anything odd → the full MAX_SECONDS."""
    st = job.get("stats") if isinstance(job.get("stats"), dict) else {}
    try:
        s = float(st.get("max_seconds") or MAX_SECONDS)
    except (TypeError, ValueError):
        s = MAX_SECONDS
    return max(5.0, min(float(MAX_SECONDS), s))


def job_lines(job):
    st = job.get("stats") if isinstance(job.get("stats"), dict) else {}
    return st.get("lines") is True


_seg = {}


def segmenter():
    """one ONNX session per worker process for race-line jobs (overlay.render makes its own)"""
    if "s" not in _seg:
        ncpu = int(os.environ.get("OVERLAY_THREADS") or 0) or os.cpu_count() or 4
        _seg["s"] = Segmenter(os.environ.get("OVERLAY_MODEL") or os.path.join(HERE, "model", "yolo11s-seg.onnx"),
                              threads=int(os.environ.get("OVERLAY_ORT_THREADS") or max(1, ncpu // 2)))
    return _seg["s"]


def token_left_s(token):
    """Seconds until the user's Supabase login (JWT) expires; a lot when it can't be read."""
    try:
        part = token.split(".")[1]
        exp = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))["exp"]
        return float(exp) - time.time()
    except Exception:
        return 24 * 3600.0


def run_course_job(j):
    """kind "course": one reference video + the videos to time (stats.videos)."""
    sb = Supa(j["supabaseUrl"], j["anonKey"], j["token"])
    jid, job = j["jobId"], j["job"]
    st = job.get("stats") if isinstance(job.get("stats"), dict) else {}
    cid = st.get("course_id")
    want = [v for v in (st.get("videos") or []) if isinstance(v, str) and UUID_RE.match(v)]
    t0 = time.time()
    # stop while the user's login still works: every result is saved with it (beta 07.11: it ran out 12 min in and
    # the last two videos' times were lost) — the rest are marked pending + more and the app asks again
    budget = min(COURSE_BUDGET_S, max(60, token_left_s(j["token"]) - 300))   # one video can take ~4 min
    tmp = tempfile.mkdtemp(prefix="co_")
    done_ids = set()

    def job_patch(body):
        sb.patch(f"overlay_jobs?id=eq.{jid}", {**body, "updated_at": now()})

    last = {"t": 0}

    def progress(p):
        if time.time() - last["t"] > 5:
            last["t"] = time.time()
            job_patch({"progress": int(max(1, min(99, p)))})

    def vid_patch(vid, body):
        s, b = sb.patch(f"videos?id=eq.{vid}", {"course": {"v": COURSE.VERSION, "course_id": cid, "job": jid, "at": now(), **body}})
        if s not in (200, 204):
            log("course result", vid, s, b[:300] if isinstance(b, bytes) else b)

    def fail(msg):
        log("course job failed", jid, msg)
        job_patch({"status": "failed", "error": msg[:500]})
        if cid:
            sb.patch(f"courses?id=eq.{cid}", {"status": "failed", "error": msg[:300], "updated_at": now()})
        for v in want:
            if v not in done_ids:
                vid_patch(v, {"status": "failed", "error": msg[:300]})

    try:
        if not cid or not UUID_RE.match(str(cid)):
            return fail(GENERIC_ERROR)
        job_patch({"status": "running", "progress": 1})
        s, rows = sb.get(f"courses?id=eq.{cid}&select=id,session_id,ref_video_id,gates,discipline")
        course = (rows or [None])[0]
        if not course or course["session_id"] != job["session_id"] or course["ref_video_id"] != job["video_a"]:
            return fail("The course is gone (deleted?).")
        if (course.get("discipline") or "").strip().upper() == "SL":
            return fail("Course timing works for GS, SG and DH for now - slalom comes later.")
        ids = ",".join(dict.fromkeys([job["video_a"]] + want))
        s, vrows = sb.get(f"videos?id=in.({ids})&select=id,session_id,athlete,bunny_id,bunny_status,track,file_name")
        vrows = {r["id"]: r for r in (vrows or []) if r.get("session_id") == job["session_id"]}
        ref = vrows.get(job["video_a"])
        if not ref or not ref.get("bunny_id") or ref.get("bunny_status") != "ready":
            return fail("The reference video is gone or still being processed.")
        pr = os.path.join(tmp, "ref.mp4")
        if not download_bunny(j["cdnHost"], ref["bunny_id"], j["referer"], pr):
            return fail("Couldn't fetch the reference video from the stream service. Please try again later.")
        refs = COURSE.ref_gates(pr, course.get("gates") or [], log=log)
        found = [{k: v for k, v in r.items() if not k.startswith("_")} for r in refs]
        sb.patch(f"courses?id=eq.{cid}", {"found": found, "status": "pending", "error": None, "updated_at": now()})
        seg = segmenter()
        track = lambda v: v.get("track") if isinstance(v.get("track"), dict) and v["track"].get("pts") else None  # noqa: E731
        order = [job["video_a"]] + [v for v in want if v != job["video_a"]]
        order = [v for v in order if v in want or v == job["video_a"]]
        results = {}
        ref_times = None          # the reference run's crossing times: a check for the other runs' times
        for n, vid in enumerate(order):
            if time.time() - t0 > budget:
                break
            v = vrows.get(vid)
            base = 5 + 90 * n / max(1, len(order))
            job_patch({"progress": int(base)})
            if not v:
                done_ids.add(vid)
                continue
            try:
                if vid == job["video_a"]:
                    path, gates = pr, [{k: x for k, x in r.items() if not k.startswith("_")} for r in refs]
                else:
                    path = os.path.join(tmp, "v.mp4")
                    if not v.get("bunny_id") or v.get("bunny_status") != "ready" or not download_bunny(j["cdnHost"], v["bunny_id"], j["referer"], path):
                        vid_patch(vid, {"status": "failed", "error": "Video not ready on the stream service."})
                        done_ids.add(vid)
                        continue
                    gates, _, _ = COURSE.locate(path, refs, log=log)
                if not any(gates):
                    res = {"status": "nomatch"}
                else:
                    res = COURSE.run_video(path, gates, seg, track=track(v), ref_times=None if vid == job["video_a"] else ref_times, is_ref=vid == job["video_a"],
                                           discipline=course.get("discipline") or "", log=log,
                                           progress=lambda p, b=base: progress(b + p * 90 / max(1, len(order))))
                if vid == job["video_a"]:
                    res["ref"] = True
                    ref_times = {g["label"]: g["t"] for g in res.get("gates") or [] if g.get("t") is not None and g.get("status") in ("ok", "unsure")} or None
                vid_patch(vid, res)
                results[vid] = res.get("status")
            except COURSE.OverlayError as e:
                vid_patch(vid, {"status": "failed", "error": str(e)[:300]})
                results[vid] = "failed"
            except Exception:  # noqa: BLE001
                log("course video error", vid, traceback.format_exc())
                vid_patch(vid, {"status": "failed", "error": GENERIC_ERROR})
                results[vid] = "failed"
            finally:
                done_ids.add(vid)
                if vid != job["video_a"]:
                    try:
                        os.remove(os.path.join(tmp, "v.mp4"))
                    except OSError:
                        pass
        left = [v for v in order if v not in done_ids]
        for v in left:          # out of time: the app asks again for these
            vid_patch(v, {"status": "pending", "more": True})
        stats = {"course_id": cid, "videos": len(order), "done": sum(1 for x in results.values() if x == "done"),
                 "nomatch": sum(1 for x in results.values() if x == "nomatch"), "failed": sum(1 for x in results.values() if x == "failed"),
                 "left": left, "total_s": round(time.time() - t0, 1)}
        sb.patch(f"courses?id=eq.{cid}", {"status": "done", "updated_at": now()})
        job_patch({"status": "done", "progress": 100, "stats": stats})
        log("course job done", jid, json.dumps(stats))
    except COURSE.OverlayError as e:
        fail(str(e))
    except Exception:  # noqa: BLE001
        log("course job error", jid, traceback.format_exc())
        fail(GENERIC_ERROR)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_job(j):
    if j["job"].get("kind") == "course":
        return run_course_job(j)
    sb = Supa(j["supabaseUrl"], j["anonKey"], j["token"])
    jid = j["jobId"]
    t0 = time.time()
    last = {"p": -1, "t": 0}

    def progress(p, force=False):
        p = int(max(0, min(99, p)))
        if force or (p != last["p"] and time.time() - last["t"] > 3):
            last.update(p=p, t=time.time())
            sb.patch(f"overlay_jobs?id=eq.{jid}", {"progress": p, "updated_at": now()})

    def fail(msg):
        log("job failed", jid, msg)
        sb.patch(f"overlay_jobs?id=eq.{jid}", {"status": "failed", "error": msg[:500], "updated_at": now()})
        if gate_job:
            sb.patch(f"videos?id=eq.{j['job']['video_a']}", {"gates": {"v": 1, "marks": gate_marks, "job": jid, "status": "failed",
                                                                         "error": msg[:300], "at": now()}})

    job = j["job"]
    gate_job = job.get("kind") == "gates"
    gate_marks = (job.get("stats") or {}).get("gates") if gate_job and isinstance(job.get("stats"), dict) else None
    tmp = tempfile.mkdtemp(prefix="ov_")
    try:
        single = job.get("kind") in ("raceline", "gates")
        what = "gate times" if gate_job else "race line" if single else "overlay"
        sb.patch(f"overlay_jobs?id=eq.{jid}", {"status": "running", "progress": 1, "updated_at": now()})
        ids = job["video_a"] if single else f"{job['video_a']},{job['video_b']}"
        s, rows = sb.get(f"videos?id=in.({ids})"
                         "&select=id,session_id,athlete,run_tag,position,bunny_id,bunny_status,track,file_name,recorded_time,recorded_date,tagged_by")
        rows = {r["id"]: r for r in (rows or [])}
        va, vb = rows.get(job["video_a"]), (None if single else rows.get(job["video_b"]))
        if not va or (not single and not vb):
            return fail("The video is gone or you no longer have access to it." if single else
                        "One of the videos is gone or you no longer have access to it.")
        if va["session_id"] != job["session_id"]:
            return fail(GENERIC_ERROR)
        for v, n in ((va, "1"), (vb, "2")):
            if v is not None and (not v.get("bunny_id") or v.get("bunny_status") != "ready"):
                return fail(("The video" if single else f"Video {n}") + " is still being processed. Open it once, wait a few minutes and try again.")
        pa, pb = os.path.join(tmp, "a.mp4"), os.path.join(tmp, "b.mp4")
        qa = download_bunny(j["cdnHost"], va["bunny_id"], j["referer"], pa)
        progress(5, True)
        qb = "-" if single else download_bunny(j["cdnHost"], vb["bunny_id"], j["referer"], pb)
        if single and not qa:
            return fail("Couldn't fetch the video from the stream service. Please try again later.")
        if not qa or not qb:
            return fail(f"Couldn't fetch video {'1' if not qa else '2'} from the stream service. Please try again later.")
        progress(8, True)
        t_dl = time.time() - t0

        user = sb.user() or {}
        md = user.get("user_metadata") or {}
        uname = (md.get("full_name") or md.get("name") or (user.get("email") or "").split("@")[0] or "User")[:80]
        out = os.path.join(tmp, "overlay.mp4")
        track = lambda v: v.get("track") if isinstance(v.get("track"), dict) and v["track"].get("pts") else None  # noqa: E731
        gate_result = None
        if gate_job:
            stats, gate_result = render_gates(pa, out, marks=gate_marks, seg=segmenter(), track=track(va),
                                              name=va.get("athlete") or "", progress=lambda p: progress(8 + p * 82), log=log)
        elif single:
            stats = render_raceline(pa, out, start=float(job["start_a"] or 0), seg=segmenter(), track=track(va),
                                    name=va.get("athlete") or "", max_seconds=job_seconds(job),
                                    progress=lambda p: progress(8 + p * 82), log=log)
        else:
            stats = render(pa, pb, out, offset_s=float(job["offset_s"]), start_a=float(job["start_a"] or 0),
                           track_a=track(va), track_b=track(vb), name_a=va.get("athlete") or "Video 1",
                           name_b=vb.get("athlete") or "Video 2", max_seconds=job_seconds(job),
                           progress=lambda p: progress(8 + p * 82), log=log, lines=job_lines(job))
        if not gate_job:
            stats["max_seconds"] = job_seconds(job)

        # upload as the user, then the videos row in A's session
        vid = str(uuid.uuid4())
        fname = (f"GATES_{safe_name(va.get('athlete'), 'Video')}.mp4" if gate_job else
                 f"RACELINE_{safe_name(va.get('athlete'), 'Video')}.mp4" if single else
                 f"OVERLAY_{safe_name(va.get('athlete'), 'Video1')}_vs_{safe_name(vb.get('athlete'), 'Video2')}.mp4")
        path = f"{va['session_id']}/{vid}_{fname}"
        progress(90, True)
        ok, us = sb.upload(path, out, lambda f: progress(90 + f * 8))
        if not ok:
            if us in (400, 401, 403):
                return fail("You can't add videos to this session (in this team only the owner and admins can add videos).")
            if us == 413:
                return fail(f"The {what} video is too large to save.")
            return fail(f"Saving the {what} video failed. Please try again.")
        s, b = sb.post("videos", {
            "id": vid, "session_id": va["session_id"], "athlete": va.get("athlete"), "file_name": fname,
            "storage_path": path, "recorded_time": va.get("recorded_time"), "recorded_date": va.get("recorded_date"),
            "uploaded_by": uname, "run_tag": va.get("run_tag"), "position": va.get("position"),
            "tagged_by": va.get("tagged_by") if va.get("athlete") else None,
        })
        if s not in (200, 201, 204):
            log("insert video", s, b[:300])
            sb.remove(path)
            if s in (401, 403) or b"row-level security" in b:
                return fail("You can't add videos to this session (in this team only the owner and admins can add videos).")
            return fail(f"Saving the {what} video failed. Please try again.")
        stats.update(download_s=round(t_dl, 1), total_s=round(time.time() - t0, 1), quality=[qa] if single else [qa, qb],
                     mb=round(os.path.getsize(out) / 1e6, 1))
        if gate_job:
            # the result on the marked video (session overview, card line); marks kept for "mark again" / course map
            gs, gb = sb.patch(f"videos?id=eq.{va['id']}", {"gates": {**gate_result, "marks": gate_marks, "job": jid, "status": "done",
                                                                     "result_video_id": vid, "at": now()}})
            if gs not in (200, 204):
                log("gates result", gs, gb[:300] if isinstance(gb, bytes) else gb)
            stats["result"] = {k: gate_result[k] for k in ("missed", "through", "unsure", "splits", "total", "ok")}
        sb.patch(f"overlay_jobs?id=eq.{jid}", {"status": "done", "progress": 100, "result_video_id": vid,
                                                "stats": stats, "updated_at": now()})
        log("job done", jid, json.dumps(stats))
    except OverlayError as e:
        fail(str(e))
    except Exception:  # noqa: BLE001
        log("job error", jid, traceback.format_exc())
        fail(GENERIC_ERROR)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def worker_loop():
    while True:
        j = jobs.get()
        state["running"] += 1
        try:
            run_job(j)
        finally:
            state["running"] -= 1


# ───────────────────────── HTTP server ─────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body):
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, {"ok": True, "queue": jobs.qsize(), "running": state["running"]})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/render":
            return self.send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 100000:
                return self.send(413, {"error": "too big"})
            body = json.loads(self.rfile.read(n) or b"{}")
            token = re.sub(r"^Bearer\s+", "", self.headers.get("Authorization") or "", flags=re.I)
            jid, sbu, anon, cdn = body.get("jobId"), body.get("supabaseUrl"), body.get("anonKey"), body.get("cdnHost")
            referer = body.get("referer") if REFERER_RE.match(body.get("referer") or "") else "https://racinghill.com/"
            if not token or not isinstance(jid, str) or not UUID_RE.match(jid) or not isinstance(anon, str):
                return self.send(400, {"error": "bad request"})
            if sbu not in ALLOWED_SUPABASE or cdn not in ALLOWED_CDN:
                return self.send(403, {"error": "not allowed"})
            # The job must exist, be this user's (RLS) and not be started yet
            s, rows = Supa(sbu, anon, token).get(f"overlay_jobs?id=eq.{jid}&select=id,status,kind,session_id,video_a,video_b,offset_s,start_a,stats")
            if s == 401:
                return self.send(401, {"error": "not logged in"})
            if not rows:
                return self.send(404, {"error": "job not found"})
            if rows[0]["status"] != "queued":
                return self.send(409, {"error": "job already started"})
            jobs.put({"jobId": jid, "supabaseUrl": sbu, "anonKey": anon, "cdnHost": cdn, "referer": referer,
                      "token": token, "job": rows[0]})
            self.send(202, {"queued": True, "position": jobs.qsize() + state["running"]})
        except Exception:  # noqa: BLE001
            log("bad request", traceback.format_exc())
            self.send(400, {"error": "bad request"})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # python server.py --test <cdnHost> <referer> <guidA> <guidB> <offsetS> <startA> <out.mp4> [trackA trackB]
        cdn, ref, ga, gb, off, start, out = sys.argv[2:9]
        # tracks: file paths or inline JSON (for a Cloud Run job test), "-" = none
        rd = lambda p: None if p == "-" else json.loads(p) if p.lstrip().startswith("{") else json.load(open(p, encoding="utf-8"))  # noqa: E731
        tr = [rd(p) for p in sys.argv[9:11]] if len(sys.argv) > 10 else [None, None]
        tmp = tempfile.mkdtemp(prefix="ovt_")
        t0 = time.time()
        qa = download_bunny(cdn, ga, ref, os.path.join(tmp, "a.mp4"))
        qb = download_bunny(cdn, gb, ref, os.path.join(tmp, "b.mp4"))
        log("downloaded", qa, qb, f"{time.time() - t0:.1f}s")
        stats = render(os.path.join(tmp, "a.mp4"), os.path.join(tmp, "b.mp4"), out, offset_s=float(off),
                       start_a=float(start), track_a=tr[0], track_b=tr[1], name_a="Video 1", name_b="Video 2", log=log)
        stats["download_s"] = round(time.time() - t0 - stats["render_s"], 1)
        print("RESULT", json.dumps(stats), flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return
    for _ in range(int(os.environ.get("PARALLEL", "1"))):
        threading.Thread(target=worker_loop, daemon=True).start()
    log(f"overlay worker on :{PORT}, supabase allow-list {len(ALLOWED_SUPABASE)}, cdn allow-list {len(ALLOWED_CDN)}")
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
