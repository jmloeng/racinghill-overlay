"""Ghost-run OVERLAY renderer (Racinghill). Production version of scripts/overlay-test (feasibility test 2026-10-25).

Run A is the main video; run B's skier is cut out and drawn over A as a see-through "ghost" with a thin cyan outline.
One pass over both videos, constant memory:
  1. Align every frame: SIFT on the background (skier masked out with the AI ZOOM track, videos.track), RANSAC
     homography B -> A. Frame-to-frame optical flow inside each video predicts the next alignment; the direct match pulls
     it back 35 % per frame (no jitter). A failed/odd direct match is bridged by the prediction for up to 2 s.
  2. Cut out B's racer: YOLO11-seg (ONNX) in a 640x640 crop around the track position / the racer's last position.
  3. Compose: warped racer at 70 % + cyan outline + labels + timer; an arrow at the edge when the ghost is off-screen.
  4. ffmpeg: H.264, yuv420p, faststart.

Timing: frame i of the output = A at startA + i/fps, B at startA + offsetS + i/fps (one fixed offset, like GhostLine).

Local test without login (writes the MP4 locally):
  python overlay.py a.mp4 b.mp4 out.mp4 --offset 0.68 --start 5 --track-a ta.json --track-b tb.json --name-a "William" --name-b "Maxi"
"""
import os
import sys
import time
import json
import queue
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from seg import Segmenter, crop_at, letterbox, SIZE

WORK = 960              # feature work resolution (longest side)
MAX_SECONDS = 40        # longest overlay
OUT_MAX_W = 1920        # output never larger than 1080p
GHOST_ALPHA = 0.7
CYAN = (255, 230, 0)    # BGR
BLUE_LINE = (255, 130, 0)  # BGR, the ghost (outline, label, race line) when race lines are on
RED_LINE = (29, 0, 232)    # BGR, video 1's race line
ALPHA_FUSE = 0.35       # how hard a direct match pulls the predicted alignment
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",            # container (fonts-dejavu-core)
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",               # macOS (local test)
    "C:/Windows/Fonts/arialbd.ttf",
]


class OverlayError(Exception):
    """Error with a message that can be shown to the user as is."""


# ───────────────────────── helpers (from scripts/overlay-test/common.py) ─────────────────────────

def track_pos(track, t):
    """skier centre (x, y as 0..1) at time t from an AI ZOOM track, or None"""
    pts = (track or {}).get("pts") if isinstance(track, dict) else None
    if not pts:
        return None
    if t < pts[0][0] - 0.5 or t > pts[-1][0] + 0.5:
        return None
    ts = [p[0] for p in pts]
    return float(np.interp(t, ts, [p[1] for p in pts])), float(np.interp(t, ts, [p[2] for p in pts]))


_clahe_local = threading.local()


def prep_gray(frame, scale):
    small = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    c = getattr(_clahe_local, "c", None)
    if c is None:
        c = _clahe_local.c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return c.apply(g)


def skier_mask(shape, pos=None, box=None, bw=0.13, bh=0.26):
    """255 = background usable for matching. Hides the skier (track position or a segmentation box) and the borders."""
    h, w = shape[:2]
    m = np.full((h, w), 255, np.uint8)
    if pos is not None:
        x, y = pos
        x0, x1 = int((x - bw / 2) * w), int((x + bw / 2) * w)
        y0, y1 = int((y - bh / 2) * h), int((y + bh / 2) * h)
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    m[:6, :] = 0
    m[-6:, :] = 0
    m[:, :6] = 0
    m[:, -6:] = 0
    return m


_sift_local = threading.local()


def sift_feats(gray, mask):
    s = getattr(_sift_local, "s", None)
    if s is None:
        s = _sift_local.s = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.015)
    return s.detectAndCompute(gray, mask)


def match_h(fb, fa, ratio=0.78, thr=3.0):
    """homography mapping B points -> A points (work res). returns H, inliers"""
    kb, db = fb
    ka, da = fa
    if db is None or da is None or len(kb) < 10 or len(ka) < 10:
        return None, 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, 0
    pb = np.float32([kb[m.queryIdx].pt for m in good])
    pa = np.float32([ka[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(pb, pa, cv2.USAC_MAGSAC, thr, maxIters=5000, confidence=0.999)
    if H is None:
        return None, 0
    return H, int(inl.sum())


def corners(w, h):
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def h_sane(H, wb, hb, wa, ha):
    """reject degenerate homographies: sensible scale, mild perspective, convex corner quad"""
    if H is None or not np.isfinite(H).all():
        return False
    d = cv2.perspectiveTransform(corners(wb, hb), H).reshape(-1, 2).astype(np.float32)
    s = cv2.contourArea(d) / (wa * ha)
    if not (0.1 < s < 9):
        return False
    if not cv2.isContourConvex(d):
        return False
    if abs(H[2, 0]) * wb > 0.6 or abs(H[2, 1]) * hb > 0.6:
        return False
    return True


def corners_dev(H1, H2, w, h):
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(c, H1).reshape(-1, 2)
    b = cv2.perspectiveTransform(c, H2).reshape(-1, 2)
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def blend_h(Hp, Hd, alpha, w, h):
    """move predicted homography towards the direct one (corner interpolation)"""
    c = corners(w, h)
    a = cv2.perspectiveTransform(c, Hp).reshape(-1, 2)
    b = cv2.perspectiveTransform(c, Hd).reshape(-1, 2)
    m = (1 - alpha) * a + alpha * b
    return cv2.getPerspectiveTransform(c.reshape(-1, 2), m.astype(np.float32))


def flow_h(g0, g1, mask0):
    """frame-to-frame homography (g0 -> g1) with sparse LK optical flow"""
    p0 = cv2.goodFeaturesToTrack(g0, 800, 0.005, 8, mask=mask0)
    if p0 is None or len(p0) < 15:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=4)
    ok = st.reshape(-1) == 1
    if ok.sum() < 15:
        return None
    H, _ = cv2.findHomography(p0[ok], p1[ok], cv2.USAC_MAGSAC, 2.0)
    return H


def scale_mat(s):
    return np.diag([s, s, 1.0])


def translate(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)


# ───────────────────────── video input ─────────────────────────

class Reader:
    """Sequential frame reader that hands out the frame nearest to a requested time (forward only)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise OverlayError("Could not open the video file.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not (1 <= self.fps <= 240):
            self.fps = 25.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.idx = -1
        self.frame = None
        self.ended = False

    @property
    def duration(self):
        return self.n / self.fps if self.n else 0

    def at(self, t):
        """(frame, index, is_new) nearest to time t, or (None, idx, False) after the end / before 0"""
        if t < -0.5 / self.fps:
            return None, -1, False
        k = max(0, int(round(t * self.fps)))
        new = False
        while self.idx < k:
            if self.ended:
                return None, self.idx, False
            if self.idx < k - 1:
                ok, f = self.cap.grab(), None      # skipped frame: no need to decode into an image
            else:
                ok, f = self.cap.read()
            if not ok:
                self.ended = True
                return None, self.idx, False
            self.idx += 1
            if self.idx == k:
                self.frame, new = f, True
        return self.frame, self.idx, new


# ───────────────────────── labels ─────────────────────────

def load_font(px):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


def label_image(text, px, color_rgb):
    """RGBA label (text on a dark translucent box), rendered once with Pillow (handles æøå, ä …)."""
    font = load_font(px)
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font)
    pad = int(px * 0.45)
    im = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 140))
    ImageDraw.Draw(im).text((pad - x0, pad - y0), text, font=font, fill=color_rgb + (255,))
    a = np.array(im)
    bgr = a[..., [2, 1, 0]].astype(np.float32)
    alpha = a[..., 3:4].astype(np.float32) / 255.0
    return bgr, alpha


def paste(frame, lab, x, y):
    bgr, alpha = lab
    h, w = bgr.shape[:2]
    H, W = frame.shape[:2]
    if x >= W or y >= H:
        return
    w2, h2 = min(w, W - x), min(h, H - y)
    roi = frame[y:y + h2, x:x + w2].astype(np.float32)
    a = alpha[:h2, :w2]
    frame[y:y + h2, x:x + w2] = (roi * (1 - a) + bgr[:h2, :w2] * a).astype(np.uint8)


def edge_arrow(frame, cx, cy, scale):
    """cyan arrow at the frame edge pointing to an off-screen ghost"""
    H, W = frame.shape[:2]
    m = int(40 * scale)
    px, py = float(np.clip(cx, m, W - m)), float(np.clip(cy, m, H - m))
    d = np.array([cx - px, cy - py])
    n = np.linalg.norm(d)
    if n < 1:
        return
    d /= n
    p = np.array([px, py])
    s = 26 * scale
    tip = p + d * s
    left = p - d * s * 0.6 + np.array([-d[1], d[0]]) * s * 0.8
    right = p - d * s * 0.6 - np.array([-d[1], d[0]]) * s * 0.8
    pts = np.int32([tip, left, right])
    ov = frame.copy()
    cv2.fillConvexPoly(ov, pts, CYAN, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)


# ───────────────────────── same spot? ─────────────────────────

SAME_SPOT_ERROR = ("These two runs don't look like they were filmed from the same spot, so they can't be lined up. "
                   "Pick a run filmed from the same camera position.")


def check_same_spot(a_path, b_path, start_a, start_b, seconds, log=print):
    """Quick check before the real work: do 6 frame pairs spread over the overlay match (SIFT, same as the render)?
    Runs filmed from different spots (e.g. Cam Pos 1 vs Cam Pos 2) match almost nowhere. Raises OverlayError."""
    RA, RB = Reader(a_path), Reader(b_path)
    span = seconds
    if RA.duration:
        span = min(span, RA.duration - start_a)
    if RB.duration:
        span = min(span, RB.duration - start_b)
    good = n = 0
    for x in np.linspace(0, max(0.0, span - 0.5), 6):
        fa, _, _ = RA.at(start_a + x)
        fb, _, _ = RB.at(start_b + x)
        if fa is None or fb is None:
            continue
        n += 1
        ga, gb = prep_gray(fa, WORK / max(fa.shape[:2])), prep_gray(fb, WORK / max(fb.shape[:2]))
        H, inl = match_h(sift_feats(gb, skier_mask(gb.shape)), sift_feats(ga, skier_mask(ga.shape)))
        if H is not None and inl >= 25 and h_sane(H, gb.shape[1], gb.shape[0], ga.shape[1], ga.shape[0]):
            good += 1
    log(f"same spot check: {good}/{n} frame pairs match")
    if n >= 3 and good < 2:
        raise OverlayError(SAME_SPOT_ERROR)
    return good, n


# ───────────────────────── the renderer ─────────────────────────

def render(a_path, b_path, out_path, *, offset_s, start_a, track_a=None, track_b=None, name_a="", name_b="",
           model_path=None, max_seconds=MAX_SECONDS, progress=None, threads=None, log=print, lines=False):
    """lines: also draw both racers' race lines (raceline.py) — video 1 red, the ghost blue. Analysis first (both videos
    at once), then the usual render with the lines drawn under the ghost."""
    T0 = time.time()
    model_path = model_path or os.environ.get("OVERLAY_MODEL") or os.path.join(HERE, "model", "yolo11s-seg.onnx")
    ncpu = threads or int(os.environ.get("OVERLAY_THREADS") or 0) or os.cpu_count() or 4  # Cloud Run: set to the vCPUs
    if os.environ.get("OVERLAY_CV_THREADS"):
        cv2.setNumThreads(int(os.environ["OVERLAY_CV_THREADS"]))
    seg = Segmenter(model_path, threads=int(os.environ.get("OVERLAY_ORT_THREADS") or max(1, ncpu // 2)))
    RA, RB = Reader(a_path), Reader(b_path)
    fps_in = RA.fps
    step = 2 if fps_in > 32 else 1                  # 50/60 fps sources → 25/30 fps output (half the work)
    fps = fps_in / step
    start_a = max(0.0, float(start_a))
    start_b = start_a + float(offset_s)
    if RA.duration and start_a >= RA.duration - 0.5:
        raise OverlayError("The start point is at the very end of video 1.")
    if RB.duration and (start_b < -0.2 or start_b >= RB.duration - 0.5):
        raise OverlayError("The moment you picked in video 2 is outside that video.")
    total = int(min(max_seconds, (RA.duration - start_a) if RA.duration else max_seconds) * fps)
    total = max(1, total)
    check_same_spot(a_path, b_path, start_a, start_b, total / fps, log=log)
    p_render = (lambda p: progress(p)) if progress else None
    lineA = lineB = an_a = an_b = None
    if lines:
        from raceline import analyse, line_for
        parts = [0.0, 0.0]

        def prog(n):
            def f(p):
                parts[n] = p
                if progress:
                    progress(0.5 * (parts[0] + parts[1]) / 2)
            return f
        with ThreadPoolExecutor(max_workers=2) as ex:
            fa_ = ex.submit(analyse, a_path, start_a, step / fps_in, total, seg, track_a, prog(0), log)
            fb_ = ex.submit(analyse, b_path, start_b, step / fps_in, total, seg, track_b, prog(1), log)
            an_a, an_b = fa_.result(), fb_.result()
        lineA, lineB = line_for(an_a), line_for(an_b)
        log(f"race lines: racer found A {an_a['found']}/{total}, B {an_b['found']}/{total}, tail A {lineA.tail} B {lineB.tail}, analysis {time.time() - T0:.0f}s")
        if progress:
            p_render = lambda p: progress(0.5 + 0.5 * p)  # noqa: E731

    # frame sizes / scales
    fa0, _, _ = RA.at(start_a)
    fb0, _, _ = RB.at(max(0.0, start_b))
    if fa0 is None or fb0 is None:
        raise OverlayError("Could not read the videos at the chosen moment.")
    hA, wA = fa0.shape[:2]
    hB, wB = fb0.shape[:2]
    sA, sB = WORK / max(wA, hA), WORK / max(wB, hB)          # full → work
    waW, haW, wbW, hbW = round(wA * sA), round(hA * sA), round(wB * sB), round(hB * sB)
    so = min(1.0, OUT_MAX_W / max(wA, hA) if wA >= hA else 1080 / wA)  # A full → output (max 1080p)
    oW, oH = int(round(wA * so)) // 2 * 2, int(round(hA * so)) // 2 * 2
    to_out_from_Awork = scale_mat(so / sA)
    from_Bfull_to_Bwork = scale_mat(sB)
    ui = oH / 1080.0 if oW >= oH else oW / 1080.0

    ghost_col = BLUE_LINE if lines else CYAN       # with race lines the ghost and its line are blue, video 1 red
    lab_a = label_image((name_a or "Video 1").upper(), int(44 * ui), (232, 0, 29) if lines else (255, 255, 255))
    lab_b = label_image(f"{(name_b or 'Video 2').upper()} (ghost)", int(44 * ui), tuple(int(c) for c in ghost_col[::-1]))
    th_line = max(3, round(oW / 380))
    lab_y2 = int(40 * ui) + lab_a[0].shape[0] + int(12 * ui)

    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{oW}x{oH}", "-r", f"{fps:.6f}",
         "-i", "-", "-c:v", "libx264", "-preset", os.environ.get("OVERLAY_X264_PRESET", "veryfast"), "-crf", "22",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)

    # Decoding in its own thread (prefetch) so it overlaps with the analysis
    q = queue.Queue(maxsize=6)

    def reader():
        try:
            for k in range(total):
                ta = start_a + k * step / fps_in
                fa, ia, _ = RA.at(ta)
                if fa is None:
                    break
                fb, ib, newb = RB.at(start_b + k * step / fps_in)
                q.put((k, ta, fa, fb, ib, newb or k == 0))  # k 0: frame already read for the sizes
        except Exception as e:  # noqa: BLE001
            q.put(e)
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    pool = ThreadPoolExecutor(max_workers=5)
    st = {"direct": 0, "predicted": 0, "lost": 0, "no_b": 0, "ghost": 0, "frames": 0}
    prevA = prevB = None          # (gray, mask)
    H = None                      # B work → A work
    since_direct = 999
    seg_prev = None               # last racer centre in B (full px)
    seg_age = 99
    b_end_frames = 0
    last_progress = 0
    MB_I = np.eye(3)
    seg_res_prev = None
    last_box_B = None             # racer box in B (full px), hides the racer from matching when there is no track
    featsB = None
    status = ""

    def segment(fb, tb):
        """racer mask in B: (mask crop, x0, y0, box_full) or None"""
        tp = track_pos(track_b, tb)
        exp = None
        if seg_prev is not None and seg_age <= 3:
            exp, lim = seg_prev, 120
        elif tp is not None:
            exp, lim = (tp[0] * wB, tp[1] * hB), 260
        if exp is None:
            # no track and no recent racer: look at the whole frame, take the most confident person
            img, s = letterbox(fb)
            res = seg.persons(img, conf=0.3)
            if not res:
                return None
            bx = max(res, key=lambda r: r[1])[0] / s
            exp, lim = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), 400
        crop, x0, y0 = crop_at(fb, exp[0], exp[1])
        best = None
        for box, conf, m in seg.persons(crop):
            bx = box + np.array([x0, y0, x0, y0], np.float32)
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            dist = float(np.hypot(cx - exp[0], cy - exp[1]))
            score = dist - 40 * conf
            if best is None or score < best[0]:
                best = (score, m, bx, dist)
        if best is None or best[3] >= lim:
            return None
        _, m, bx, _ = best
        hgt = bx[3] - bx[1]
        r_ = max(3, int(hgt * 0.06))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1)))
        m = cv2.GaussianBlur(m, (0, 0), max(1.5, r_ / 2))
        return m, x0, y0, bx

    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            k, ta, fa, fb, ib, newb = item
            tb = ib / RB.fps if fb is not None else None
            if fb is None:
                b_end_frames += 1
                if b_end_frames > 2 * fps:      # ghost's video ended 2 s ago: stop
                    break

            # ── parallel per-frame work ──
            ga = pool.submit(prep_gray, fa, sA)
            gb = pool.submit(prep_gray, fb, sB) if (fb is not None and newb) else None
            sg = pool.submit(segment, fb, tb) if (fb is not None and newb) else None
            ga = ga.result()
            posA = track_pos(track_a, ta)
            maskA = skier_mask(ga.shape, posA)
            fA = pool.submit(sift_feats, ga, maskA)
            flA = pool.submit(flow_h, prevA[0], ga, prevA[1]) if (prevA is not None and an_a is None) else None
            if gb is not None:
                gbv = gb.result()
                posB = track_pos(track_b, tb)
                # no track: hide the racer's last box instead
                boxB = last_box_B * sB if (posB is None and last_box_B is not None) else None
                maskB = skier_mask(gbv.shape, posB, boxB)
                fB = pool.submit(sift_feats, gbv, maskB)
                flB = pool.submit(flow_h, prevB[0], gbv, prevB[1]) if (prevB is not None and an_a is None) else None
            seg_res = sg.result() if sg is not None else (seg_res_prev if fb is not None else None)

            # ── alignment (sequential fusion, as in align.py) ──
            MA = flA.result() if flA is not None else None
            if MA is not None and not h_sane(MA, waW, haW, waW, haW):
                MA = None
            if fb is None:
                H = None
                st["no_b"] += 1
                status = "no_b"
            else:
                if gb is not None:
                    MB = flB.result() if flB is not None else None
                    if MB is not None and not h_sane(MB, wbW, hbW, wbW, hbW):
                        MB = None
                    featsB = fB.result()
                else:
                    MB = MB_I                    # same B frame as last time (B has a lower frame rate)
                if an_a is not None and k > 0:
                    # race lines on: the race-line analysis has steadier camera motion (keyframe SIFT, no drift)
                    MA = scale_mat(sA) @ an_a["Hs"][k] @ scale_mat(1 / sA)
                    MB = scale_mat(sB) @ an_b["Hs"][k] @ scale_mat(1 / sB)
                pred = None
                if H is not None and MA is not None and MB is not None and prevA is not None:
                    pred = MA @ H @ np.linalg.inv(MB)
                    pred /= pred[2, 2]
                Hd, n = match_h(featsB, fA.result()) if featsB is not None else (None, 0)
                ok = Hd is not None and n >= 30 and h_sane(Hd, wbW, hbW, waW, haW)
                if ok and pred is not None:
                    if corners_dev(pred, Hd, wbW, hbW) > 60 and since_direct < 25:
                        ok = False
                if ok:
                    H = Hd if pred is None else blend_h(pred, Hd, ALPHA_FUSE, wbW, hbW)
                    since_direct = 0
                    status = "direct"
                elif pred is not None and since_direct < (4 if an_a is not None else 2) * fps:
                    H = pred
                    since_direct += 1
                    status = "predicted"
                else:
                    H = None
                    since_direct += 1
                    status = "lost"
                st[status] += 1
            prevA = (ga, maskA)
            if gb is not None:
                prevB = (gbv, maskB)

            # racer tracking state
            if sg is not None:
                if seg_res is not None:
                    bx = seg_res[3]
                    seg_prev = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
                    seg_age = 0
                    last_box_B = bx
                else:
                    seg_age += 1
            seg_res_prev = seg_res

            # ── compose ──
            frame = cv2.resize(fa, (oW, oH), interpolation=cv2.INTER_AREA) if (oW, oH) != (wA, hA) else fa.copy()
            if lineA is not None:          # race lines under the ghost: the ghost's (mapped into A) first, A's on top
                lineA.step(k)
                lineB.step(k)
                if H is not None:
                    lineB.draw(frame, BLUE_LINE, th_line, to_out_from_Awork @ H @ from_Bfull_to_Bwork)
                lineA.draw(frame, RED_LINE, th_line, scale_mat(so))
            if H is not None and seg_res is not None:
                m, x0, y0, bx = seg_res
                Hf = to_out_from_Awork @ H @ from_Bfull_to_Bwork            # B full px → output px
                ys, xs = np.nonzero(m > 8)
                if len(xs):
                    bx0, by0 = int(xs.min()) + x0, int(ys.min()) + y0
                    bx1, by1 = min(wB, int(xs.max()) + x0 + 1), min(hB, int(ys.max()) + y0 + 1)
                    c = np.float32([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]).reshape(-1, 1, 2)
                    d = cv2.perspectiveTransform(c, Hf).reshape(-1, 2)
                    dx0, dy0 = np.floor(d.min(0)).astype(int)
                    dx1, dy1 = np.ceil(d.max(0)).astype(int)
                    cx, cy = d.mean(0)
                    ex0, ey0, ex1, ey1 = max(0, dx0 - 2), max(0, dy0 - 2), min(oW, dx1 + 2), min(oH, dy1 + 2)
                    if ex1 - ex0 > 2 and ey1 - ey0 > 2 and (dx1 - dx0) < 3 * oW:
                        # warp only the racer's region: B crop → output ROI
                        Hroi = translate(-ex0, -ey0) @ Hf @ translate(bx0, by0)
                        srcB = fb[by0:by1, bx0:bx1]
                        srcM = np.ascontiguousarray(m[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0])
                        size = (ex1 - ex0, ey1 - ey0)
                        wB_ = cv2.warpPerspective(srcB, Hroi, size, flags=cv2.INTER_LINEAR)
                        wM = cv2.warpPerspective(srcM, Hroi, size, flags=cv2.INTER_LINEAR)
                        roi = frame[ey0:ey1, ex0:ex1]
                        a = wM.astype(np.float32)[..., None] / 255.0 * GHOST_ALPHA
                        roi[:] = (roi * (1 - a) + wB_ * a).astype(np.uint8)
                        cnts, _ = cv2.findContours((wM > 100).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            ov = roi.copy()
                            cv2.drawContours(ov, cnts, -1, ghost_col, max(2, int(2 * ui)), cv2.LINE_AA)
                            cv2.addWeighted(ov, 0.6, roi, 0.4, 0, roi)
                            st["ghost"] += 1
                    elif not (0 <= cx < oW and 0 <= cy < oH) and abs(cx) < 4 * oW and abs(cy) < 4 * oH:
                        edge_arrow(frame, cx, cy, ui)
            paste(frame, lab_a, int(40 * ui), int(40 * ui))
            paste(frame, lab_b, int(40 * ui), lab_y2)
            t_rel = k / fps
            cv2.putText(frame, f"{t_rel:5.2f}s", (oW - int(200 * ui), int(80 * ui)), cv2.FONT_HERSHEY_DUPLEX, 1.1 * ui,
                        (0, 0, 0), max(3, int(5 * ui)), cv2.LINE_AA)
            cv2.putText(frame, f"{t_rel:5.2f}s", (oW - int(200 * ui), int(80 * ui)), cv2.FONT_HERSHEY_DUPLEX, 1.1 * ui,
                        (255, 255, 255), max(1, int(2 * ui)), cv2.LINE_AA)
            try:
                ff.stdin.write(frame.tobytes())
            except BrokenPipeError:
                raise OverlayError("The video encoder stopped.")
            st["frames"] += 1
            if p_render and time.time() - last_progress > 2:
                last_progress = time.time()
                p_render(k / total)
            # early stop: 3 s in and the two videos hardly ever line up (backup for check_same_spot)
            with_b_now = st["frames"] - st["no_b"]
            if with_b_now == int(3 * fps) and st["direct"] < 0.1 * with_b_now:
                raise OverlayError(SAME_SPOT_ERROR)
            if k % 100 == 0:
                log(f"frame {k}/{total} {status} {time.time() - T0:.1f}s")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            ff.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        ff.wait()
    if ff.returncode != 0:
        raise OverlayError("The video encoder failed.")
    n = max(1, st["frames"])
    with_b = max(1, n - st["no_b"])
    stats = {
        "frames": st["frames"], "fps": round(fps, 2), "seconds": round(st["frames"] / fps, 1),
        "aligned_pct": round(100 * st["direct"] / with_b, 1), "bridged": st["predicted"], "lost": st["lost"],
        "ghost_pct": round(100 * st["ghost"] / with_b, 1), "render_s": round(time.time() - T0, 1),
        "size": [oW, oH], "cpus": ncpu,
    }
    if lineA is not None:
        stats.update(lines=True, line_a_pct=round(100 * lineA.drawn / n, 1), line_b_pct=round(100 * lineB.drawn / n, 1),
                     tail=[lineA.tail, lineB.tail])
    if st["frames"] < fps:
        raise OverlayError("The overlay would be shorter than one second.")
    if st["direct"] < 0.2 * with_b:
        raise OverlayError(SAME_SPOT_ERROR)
    return stats


# ───────────────────────── local test ─────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--track-a")
    ap.add_argument("--track-b")
    ap.add_argument("--name-a", default="Video 1")
    ap.add_argument("--name-b", default="Video 2")
    ap.add_argument("--model")
    ap.add_argument("--seconds", type=float, default=MAX_SECONDS)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--lines", action="store_true", help="also draw both race lines")
    a = ap.parse_args()
    load = lambda p: json.load(open(p, encoding="utf-8")) if p else None  # noqa: E731
    stats = render(a.a, a.b, a.out, offset_s=a.offset, start_a=a.start, track_a=load(a.track_a), track_b=load(a.track_b),
                   name_a=a.name_a, name_b=a.name_b, model_path=a.model, max_seconds=a.seconds, threads=a.threads,
                   progress=lambda p: print(f"progress {p * 100:.0f}%", flush=True), lines=a.lines)
    print("RESULT", json.dumps(stats))
