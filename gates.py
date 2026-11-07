"""GATE TIMING + GATE JUDGE (beta, job kind "gates"): passing time at every gate the coach marked, and whether the
racer passed it, from one ordinary training video.

The owner's rule (2026-11-06): a gate = a panel held by two poles. The racer passes a gate when the feet cross the
extended line through the two pole feet, close to the gate. Valid passage (owner 2026-11-07, GS / SG / DH): the feet
cross the segment from between the two poles out to 5 m beyond them along that line (scale from the pole spacing,
GS panel ≈ 75 cm). SL (single poles) is not handled yet. The coach marks each gate once on the clip — paused in
the player, taps the two pole feet (marks = [{t, a: [x, y], b: [x, y]}], x/y as 0..1 of the picture at time t).

  analyse()   (raceline.py) racer's foot point + camera motion every frame, at the video's full frame rate
  passes()    each mark is carried with the camera to the frames around it; the crossing is the frame pair where the
              foot point changes side of the line, interpolated between the two frames (50 fps → ~±10 ms of frame
              timing; foot-point noise and the pole-to-snow geometry add to that — see docs/ROADMAP.md for the check)
              status per gate:
                ok        crossed the line between the poles or up to 5 m beyond them (between: true = between the poles)
                missed    no crossing near the gate (skied out / missed it, or the racer wasn't found)
                unsure    a crossing was found but the camera tracking was lost on the way → check by eye
  render_gates()  the clip with every gate line, its number and passing time burned in, so it can be checked by eye
                  (saved as GATES_<name>.mp4), plus the result for videos.gates.

Designed for the course map later (docs/ROADMAP.md "Course map"): every result keeps the pole feet in the frame
where the coach marked them, so another clip filmed from the same spot can predict and verify them.
"""
import subprocess
import time

import cv2
import numpy as np

from overlay import OverlayError, Reader, scale_mat, label_image, paste, OUT_MAX_W
from raceline import analyse, chain, apply

VERSION = 1
MAX_GATES = 40
WINDOW_S = 1.3        # look for the crossing this long before / after the moment the coach marked the gate
PAD_S = 1.5           # analysed / shown before the first and after the last mark
MAX_SPAN_S = 60       # longest stretch of video one job looks at
PANEL_M = 0.75        # pole spacing of a GS panel (m) — the scale in the picture
SEGMENT_M = 5.0       # a passage counts up to this far beyond the nearer pole (owner 2026-11-07)
NEAR_POLES = SEGMENT_M / PANEL_M   # … in pole spacings
PARTIAL_S = 0.15      # extra ± when the racer is partly hidden at the crossing (seen on Hintertux 0220 gate 1)
MAX_GAP_S = 0.4      # longest stretch without a trusted foot point that a crossing may be interpolated across
TAP_PX = 3.0         # how far off a coach's tap on a pole foot typically is (px at 1080p, zoomed in)
FOOT_PX = 3.0         # foot point noise (px at 1080p)
JUMP = 0.45         # foot point moving more than this many body heights over the snow in one frame (at 50 fps) = a jump
GREEN = (80, 220, 60)
RED = (29, 0, 232)
WHITE = (255, 255, 255)
GREY = (190, 190, 190)


def clean_marks(raw):
    """[{t, a, b}] from the job → sorted list, junk dropped. Raises OverlayError when nothing usable is left."""
    out = []
    for m in raw if isinstance(raw, list) else []:
        try:
            t = float(m["t"])
            a = [float(m["a"][0]), float(m["a"][1])]
            b = [float(m["b"][0]), float(m["b"][1])]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if not (0 <= t <= 7200) or not all(-0.05 <= v <= 1.05 for v in a + b):
            continue
        if np.hypot(a[0] - b[0], a[1] - b[1]) < 0.002:
            continue
        out.append({"t": t, "a": a, "b": b})
    out.sort(key=lambda m: m["t"])
    if not out:
        raise OverlayError("No gates were marked. Pause where the racer passes a gate and tap the two pole feet.")
    return out[:MAX_GATES]


def marks_track(marks, track=None):
    """Where the racer surely is: next to each marked gate at its moment (the coach paused there). Used as the
    analysis' track with pull=True. The AI ZOOM track (if any) fills in before the first and after the last mark."""
    pts = [[m["t"], (m["a"][0] + m["b"][0]) / 2, (m["a"][1] + m["b"][1]) / 2] for m in marks]
    extra = (track or {}).get("pts") if isinstance(track, dict) else None
    if extra:
        lo, hi = marks[0]["t"] - 0.6, marks[-1]["t"] + 0.6
        pts += [list(p[:3]) for p in extra if p[0] < lo or p[0] > hi]
    pts.sort(key=lambda p: p[0])
    return {"pts": pts}


def side(p, A, B):
    """signed distance (px) of p from the line A–B"""
    d = B - A
    n = np.array([-d[1], d[0]]) / max(1e-6, float(np.linalg.norm(d)))
    return float((np.asarray(p) - A) @ n)


def _sure_between(sure, i, k):
    lo, hi = (i, k) if i <= k else (k, i)
    return all(sure[lo + 1:hi + 1])


def passes(a, marks, start):
    """One result per mark, in passing order: {t, status, dist (in panel widths), at (crossing point in that frame,
    px), k (frame), mark}. t = video time of the crossing (s)."""
    fps, feet, Hs, boxes, sure = a["fps"], a["feet"], a["Hs"], a["boxes"], a["sure"]
    W, H = a["size"]
    n = len(feet)
    win = int(round(WINDOW_S * fps))
    # frames where the foot point jumps (the detection switched to a bystander, a pole or the panel for a moment):
    # a "crossing" there is not the racer. Racer ~15 m/s = ~0.2 body heights per frame at 50 fps.
    jump = [False] * n
    for k in range(1, n):
        if feet[k] is None or feet[k - 1] is None or boxes[k] is None:
            continue
        p = apply(Hs[k], feet[k - 1])
        jump[k] = np.hypot(feet[k][0] - p[0], feet[k][1] - p[1]) > JUMP * max(10, boxes[k][3] - boxes[k][1]) * 50 / fps
    res = []
    for mk in marks:
        i = int(round((mk["t"] - start) * fps))
        i = max(0, min(n - 1, i))
        A0 = np.array([mk["a"][0] * W, mk["a"][1] * H])
        B0 = np.array([mk["b"][0] * W, mk["b"][1] * H])
        s = {}
        if mk.get("win"):         # SET COURSE: the stretch where this gate is in the picture (course.py)
            klo = max(0, int(np.floor((mk["win"][0] - start) * fps)))
            khi = min(n, int(np.ceil((mk["win"][1] - start) * fps)) + 1)
        else:
            klo, khi = max(0, i - win), min(n, i + win + 1)
        for k in range(klo, khi):
            if feet[k] is None:
                continue
            M = chain(Hs, i, k)
            A, B = np.array(apply(M, A0)), np.array(apply(M, B0))
            s[k] = (side(feet[k], A, B), A, B)
        best = None
        # frames we trust: racer found, no jump nearby. Across a short stretch without them (racer hidden behind the
        # panel, seen from the camera, or the detection jumping) the crossing is interpolated, with a larger ±.
        ks = [k for k in sorted(s) if not any(jump[max(0, k - 2):min(n, k + 3)])]
        for u, v in zip(ks, ks[1:]):
            if v - u > MAX_GAP_S * fps:
                continue
            su, sv = s[u][0], s[v][0]
            if su == 0 or np.sign(su) == np.sign(sv):
                continue
            f = su / (su - sv)
            fu, fv = np.array(feet[u]), np.array(feet[v])
            # the line, carried to the crossing moment (between frame u and v)
            A = s[u][1] + f * (s[v][1] - s[u][1])
            B = s[u][2] + f * (s[v][2] - s[u][2])
            cross = fu + f * (fv - fu)
            pw = max(1.0, float(np.linalg.norm(B - A)))
            along = float((cross - A) @ (B - A)) / pw ** 2
            if 0 <= along <= 1:
                dist = 0.0
            else:
                dist = float(min(np.linalg.norm(cross - A), np.linalg.norm(cross - B)))
            near = NEAR_POLES * pw
            if dist > near:
                continue
            t = start + (u + f * (v - u)) / fps
            between = 0 <= along <= 1
            status = "ok" if _sure_between(sure, i, u) else "unsure"
            # How sure the time is: a tap a few px off moves the extended line most far from the poles; the racer
            # crossing the line steeply (fast across it) makes that matter little, running along it makes it matter a
            # lot (filmed from below, the racer often moves along the gate line in the picture).
            # px per frame across the line, over ±3 frames (one frame pair is too noisy)
            lo = min((q for q in s if u - 3 <= q <= u), default=u)
            hi = max((q for q in s if v <= q <= v + 3), default=v)
            perp = min(abs(su - sv) / (v - u), abs(s[lo][0] - s[hi][0]) / max(1, hi - lo))
            e_line = TAP_PX * (H / 1080) * np.hypot(1 + dist / pw, dist / pw)
            e_foot = FOOT_PX * (H / 1080)
            err = max(0.5 / fps, float(np.hypot(e_line, e_foot)) / max(0.3, perp) / fps)
            if v - u > 1:
                err = float(np.hypot(err, (v - u) / fps / 3))      # hidden stretch: the crossing is somewhere in it
            # the racer's outline changing size a lot around the crossing (legs hidden behind a pole, panel or another
            # person for a moment) makes the foot point jump: the crossing may be a few frames off
            hs = [boxes[q][3] - boxes[q][1] for q in range(max(0, u - 4), min(n, v + 5)) if boxes[q] is not None]
            partial = len(hs) > 1 and max(hs) > 1.3 * min(hs)
            if partial:
                err = float(np.hypot(err, PARTIAL_S))
            cand = {"t": round(t, 3), "status": status, "dist": round(dist / pw, 2), "at": [round(float(cross[0]), 1), round(float(cross[1]), 1)],
                    "k": u, "frames": [u, v], "mark": mk, "err": round(min(err, 0.5), 3), "perp": round(perp, 2),
                    **({"hidden": round((v - u - 1) / fps, 2)} if v - u > 1 else {}), **({"between": True} if between else {}),
                    **({"partial": True} if partial else {})}
            # several crossings near one gate (foot point wobbling on the line): the one closest to the poles,
            # then the one closest to the marked moment; a seen crossing beats one across a hidden stretch
            # (SET COURSE: closest to the expected moment, near_t, from the reference run's times)
            key = (v - u > 1, round(dist / pw, 1), abs(t - mk.get("near_t", mk["t"])))
            if best is None or key < best[0]:
                best = (key, cand)
        if best:
            res.append(best[1])
        else:
            res.append({"t": None, "status": "missed", "dist": None, "at": None, "k": i, "frames": None, "mark": mk, "err": None, "perp": None})
    # passing order: crossing time, missed gates at the moment they were marked
    res.sort(key=lambda r: r["t"] if r["t"] is not None else r["mark"]["t"])
    for gi, r in enumerate(res):
        r["n"] = gi + 1
    return res


def summary(res):
    """{passes, judge, splits, total} for videos.gates. Splits only between gates that were passed (ok/unsure)."""
    timed = [r for r in res if r["t"] is not None and r["status"] in ("ok", "unsure")]
    t0 = timed[0]["t"] if timed else None
    out = []
    for r in res:
        out.append({"n": r["n"], "status": r["status"], "t": r["t"], "from_first": round(r["t"] - t0, 3) if (r["t"] is not None and t0 is not None) else None,
                    "dist": r["dist"], "err": r["err"], "mark": r["mark"], **({"between": True} if r.get("between") else {})})
    missed = [r["n"] for r in res if r["status"] == "missed"]
    through = [r["n"] for r in res if r["status"] == "through"]
    unsure = [r["n"] for r in res if r["status"] == "unsure"]
    def pm(a, b):
        return round(float(np.hypot(a["err"], b["err"])), 3)
    splits = [{"from": a["n"], "to": b["n"], "s": round(b["t"] - a["t"], 3), "err": pm(a, b)} for a, b in zip(timed, timed[1:])]
    return {"passes": out, "missed": missed, "through": through, "unsure": unsure, "splits": splits,
            "total": {"from": timed[0]["n"], "to": timed[-1]["n"], "s": round(timed[-1]["t"] - timed[0]["t"], 3), "err": pm(timed[0], timed[-1])} if len(timed) > 1 else None,
            "ok": not missed and not through}


def _ext(A, B, f=2.2):
    d = B - A
    return A - f * d, B + f * d


def render_gates(path, out_path, *, marks, seg, track=None, name="", progress=None, log=print, analysis=None):
    """Job kind "gates". Analysis 0–70 %, drawing 70–100 %. Returns (stats, result for videos.gates)."""
    T0 = time.time()
    marks = clean_marks(marks)
    R = Reader(path)
    fps = R.fps
    dur = R.duration or (marks[-1]["t"] + PAD_S)
    if marks[0]["t"] > dur + 0.5:
        raise OverlayError("The marked gates are past the end of the video.")
    start = max(0.0, marks[0]["t"] - PAD_S - WINDOW_S)
    end = min(dur, marks[-1]["t"] + PAD_S + WINDOW_S)
    if end - start > MAX_SPAN_S:
        raise OverlayError(f"The marked gates span more than {MAX_SPAN_S} seconds of video. Mark gates in one run only.")
    # full frame rate: every frame counts for timing (50 fps → 20 ms between frames)
    start = round(start * fps) / fps
    total = max(1, int((end - start) * fps))
    a = analysis or analyse(path, start, 1.0 / fps, total, seg, marks_track(marks, track), pull=True,
                            progress=(lambda p: progress(0.7 * p)) if progress else None, log=log)
    if a["found"] < 0.15 * total:
        raise OverlayError("Couldn't find the racer in this video, so there are no gate times.")
    res = passes(a, marks, start)
    result = summary(res)
    result.update(v=VERSION, fps=round(fps, 3), start=round(start, 3), found_pct=round(100 * a["found"] / max(1, total), 1))

    W, Hh = a["size"]
    so = min(1.0, OUT_MAX_W / max(W, Hh) if W >= Hh else 1080 / W)
    oW, oH = int(round(W * so)) // 2 * 2, int(round(Hh * so)) // 2 * 2
    S = scale_mat(so)
    ui = oH / 1080.0 if oW >= oH else oW / 1080.0
    th = max(2, round(3 * ui))
    title = label_image(((name or "Gates") + " · GATES").upper(), int(40 * ui), (232, 0, 29))
    labels = {}

    def lab(text, color, px=64):
        key = (text, color, px)
        if key not in labels:
            labels[key] = label_image(text, int(px * ui), color)
        return labels[key]

    t_first = next((r["t"] for r in res if r["t"] is not None and r["status"] in ("ok", "unsure")), None)
    by_frame = {r["n"]: r for r in res}
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{oW}x{oH}", "-r", f"{fps:.6f}",
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)
    R = Reader(path)
    frames = 0
    show = int(round(1.4 * fps))
    try:
        for k in range(len(a["Hs"])):
            t = start + k / fps
            frame, _, _ = R.at(t)
            if frame is None:
                break
            if (oW, oH) != (W, Hh):
                frame = cv2.resize(frame, (oW, oH), interpolation=cv2.INTER_AREA)
            for r in res:
                mk = r["mark"]
                i = int(round((mk["t"] - start) * fps))
                ref = r["k"] if r["t"] is not None else i
                if abs(k - ref) > show:
                    continue
                M = S @ chain(a["Hs"], i, k)
                A = np.array(apply(M, (mk["a"][0] * W, mk["a"][1] * Hh)))
                B = np.array(apply(M, (mk["b"][0] * W, mk["b"][1] * Hh)))
                passed = r["t"] is not None and t >= r["t"]
                gone = r["t"] is None and k > i + int(WINDOW_S * fps)
                col = GREEN if passed and r["status"] == "ok" else RED if (gone or (passed and r["status"] == "through")) else (0, 200, 255) if passed else WHITE
                E0, E1 = _ext(A, B)
                cv2.line(frame, tuple(int(v) for v in E0), tuple(int(v) for v in E1), (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.line(frame, tuple(int(v) for v in E0), tuple(int(v) for v in E1), col, th, cv2.LINE_AA)
                for P in (A, B):
                    cv2.circle(frame, (int(P[0]), int(P[1])), th * 3, col, -1, cv2.LINE_AA)
                if passed:
                    txt = f"{r['n']}  {r['t'] - t_first:.2f}" if t_first is not None and r["status"] != "through" else f"{r['n']}  ?"
                elif gone:
                    txt = f"{r['n']}  MISSED"
                else:
                    txt = f"{r['n']}"
                L = lab(txt, tuple(int(c) for c in col[::-1]))
                x = int(max(A[0], B[0]) + 14 * ui)
                y = int(min(A[1], B[1]) - L[0].shape[0] - 40 * ui)
                if 0 <= x < oW and 0 <= y < oH:
                    paste(frame, L, x, max(0, y))
            # running clock from gate 1 and the judge strip
            paste(frame, title, int(30 * ui), int(30 * ui))
            if t_first is not None and t >= t_first:
                last = max((r for r in res if r["t"] is not None and r["t"] <= t and r["status"] in ("ok", "unsure")), key=lambda r: r["t"], default=None)
                clock = lab(f"{t - t_first:5.2f} s", WHITE[::-1], 56)
                paste(frame, clock, oW - clock[0].shape[1] - int(30 * ui), int(30 * ui))
            strip = " ".join(
                f"{r['n']}" + ("✓" if r["status"] == "ok" and r["t"] is not None and t >= r["t"] else
                               "✗" if (r["status"] == "missed" and k > int(round((r["mark"]["t"] - start) * fps)) + int(WINDOW_S * fps)) or (r["status"] == "through" and r["t"] is not None and t >= r["t"]) else
                               "?" if r["status"] == "unsure" and r["t"] is not None and t >= r["t"] else "·")
                for r in res)
            sl = lab(strip, (255, 255, 255), 40)
            paste(frame, sl, int(30 * ui), oH - sl[0].shape[0] - int(30 * ui))
            try:
                ff.stdin.write(frame.tobytes())
            except BrokenPipeError:
                raise OverlayError("The video encoder stopped.")
            frames += 1
            if progress and k % 10 == 0:
                progress(0.7 + 0.3 * k / len(a["Hs"]))
    finally:
        try:
            ff.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        ff.wait()
    if ff.returncode != 0:
        raise OverlayError("The video encoder failed.")
    stats = {"kind": "gates", "frames": frames, "fps": round(fps, 2), "seconds": round(frames / fps, 1), "gates": len(res),
             "ok": sum(r["status"] == "ok" for r in res), "missed": result["missed"], "through": result["through"],
             "found_pct": result["found_pct"], "render_s": round(time.time() - T0, 1), "size": [oW, oH]}
    del by_frame
    return stats, result
