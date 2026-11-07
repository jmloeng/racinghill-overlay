"""Race line (beta): the path the racer's feet took on the snow, drawn glued to the slope while the camera pans and zooms.
Production version of scripts/raceline-test/raceline.py (feasibility test 2026-10-27), with the ONNX segmenter.

  analyse()  one pass over a stretch of video: racer's foot point + camera motion per output frame
             (YOLO11-seg person masks in a crop around the expected position; camera motion = SIFT to a keyframe,
             predicted by optical flow — bounded drift, so fog/flat light works when filmed from the side).
  Line       the line of one video, kept in that video's current-frame coordinates; draw() can map it into another
             video (the ghost's line in OVERLAY).
  render_raceline()  a single video with its line (job kind "raceline").

The line is only drawn while the racer moves steadily over the snow (no lines among people waiting), is cleared when
the camera tracking is lost or the racer is gone for 1.5 s, and becomes a 3 s fading tail when the camera films
straight up the slope (turns look flat from there).
"""
import os
import subprocess
import time

import cv2
import numpy as np

from overlay import (OverlayError, Reader, prep_gray, flow_h, h_sane, track_pos, scale_mat, sift_feats, match_h, corners,
                     corners_dev, blend_h, skier_mask, label_image, paste, WORK, MAX_SECONDS, OUT_MAX_W)
from seg import SIZE, letterbox

SMOOTH = 3            # frames each side for foot smoothing
LINE_SMOOTH_S = float(os.environ.get("RACELINE_SMOOTH_S") or 0.4)  # drawn line averaged over this much time (0 = off); ends stay put
GLOW = float(os.environ.get("RACELINE_GLOW") or 0.45)              # soft coloured glow round the line (0 = off)
GAP_S = 0.35          # missing racer longer than this = break in the line
SIFT_EVERY = int(os.environ.get("RACELINE_SIFT_EVERY") or 2)  # keyframe match every n-th frame (optical flow in between)
RED = (29, 0, 232)    # BGR, Racinghill red (video 1 / the single video)
BLUE = (255, 130, 0)  # BGR, the ghost's line in OVERLAY
WHITE = (255, 255, 255)
PULL = 12.0           # analyse(pull=True): score per frame width away from the marked track
PULL_FAR = 0.2        # … and further than this from it, the whole picture is searched again (not just a crop)
PULL_NEAR = 0.08     # … and a person this close to it may be picked even if far from the one followed so far


def apply(H, p):
    v = H @ np.array([p[0], p[1], 1.0])
    return v[0] / v[2], v[1] / v[2]


def detect(seg, frame, prior, size):
    """people near prior (x, y px) -> [(box_full, mask bool (crop coords, full px scale), (x0, y0), conf)];
    whole frame (letterboxed) when there is no prior"""
    H, W = frame.shape[:2]
    out = []
    if prior is None:
        img, s = letterbox(frame)
        for box, cf, m in seg.persons(img, conf=0.2):
            if cf < 0.2:
                continue
            b = box / s
            x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
            x1, y1 = min(W, int(np.ceil(b[2]))), min(H, int(np.ceil(b[3])))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            mm = cv2.resize(m[int(y0 * s):int(np.ceil(y1 * s)) + 1, int(x0 * s):int(np.ceil(x1 * s)) + 1], (x1 - x0, y1 - y0),
                            interpolation=cv2.INTER_LINEAR) > 127
            out.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), mm, (x0, y0), float(cf)))
        return out
    n = seg.small                 # the fast 320 model when there is one; a small racer is cropped at full resolution
    c = int(min(min(H, W), max(n, (size or 0) * 3.2)))
    x0 = int(np.clip(prior[0] - c / 2, 0, W - c))
    y0 = int(np.clip(prior[1] - c / 2, 0, H - c))
    crop = frame[y0:y0 + c, x0:x0 + c]
    k = c / n
    if c != n:
        crop = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA)
    for box, cf, m in seg.persons(crop, conf=0.2):
        b = box * k
        if c != n:
            m = cv2.resize(m, (c, c), interpolation=cv2.INTER_LINEAR)
        out.append(((float(b[0] + x0), float(b[1] + y0), float(b[2] + x0), float(b[3] + y0)), m > 127, (x0, y0), float(cf)))
    return out


def foot_of(mask, off):
    ys, xs = np.nonzero(mask)
    if len(ys) < 30:
        return None
    y1, y0 = ys.max(), ys.min()
    sel = ys >= y1 - 0.12 * (y1 - y0 + 1)
    return float(xs[sel].mean() + off[0]), float(y1 + off[1])


class CamTrack:
    """Camera motion with bounded drift: every frame is matched (SIFT) to a keyframe, predicted by frame-to-frame optical
    flow. Chaining only frame to frame let small zoom errors pile up over hundreds of frames (stacked lines in fog)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.kf = None
        self.A = np.eye(3)            # keyframe -> previous frame (work coords)
        self.prev_g = self.prev_m = None

    def overlap(self, A):
        d = cv2.perspectiveTransform(corners(self.w, self.h), A).reshape(-1, 2)
        x0, y0 = np.clip(d.min(0), 0, [self.w, self.h])
        x1, y1 = np.clip(d.max(0), 0, [self.w, self.h])
        return max(0, x1 - x0) * max(0, y1 - y0) / (self.w * self.h)

    def step(self, g, m, direct=True):
        """-> (H prev->cur in work coords, sure). direct=False: optical flow only, no keyframe match (SIFT is the slow
        part; matching every 2nd frame still keeps the drift bounded) unless the flow fails or a new keyframe is due."""
        if self.kf is None:
            self.kf, self.A, self.prev_g, self.prev_m = sift_feats(g, m), np.eye(3), g, m
            return np.eye(3), True
        w, h = self.w, self.h
        Hf = flow_h(self.prev_g, g, self.prev_m)
        flow_ok = Hf is not None and h_sane(Hf, w, h, w, h) and corners_dev(Hf, np.eye(3), w, h) < 0.35 * w
        pred = (Hf if flow_ok else np.eye(3)) @ self.A
        if not direct and flow_ok and self.overlap(pred) >= 0.55:
            H = pred @ np.linalg.inv(self.A)
            z = np.sqrt(abs(np.linalg.det(H[:2, :2])))
            self.A, self.prev_g, self.prev_m = pred, g, m
            return H, bool(0.9 < z < 1.11)
        feats = sift_feats(g, m)
        Hd, inl = match_h(self.kf, feats)
        direct_ok = Hd is not None and inl >= 25 and h_sane(Hd, w, h, w, h) and (not flow_ok or corners_dev(Hd, pred, w, h) < 0.05 * w)
        if direct_ok:
            A = blend_h(pred, Hd, 0.5, w, h) if flow_ok else Hd
        else:
            A = pred
        H = A @ np.linalg.inv(self.A)
        sure = flow_ok or direct_ok
        z = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        if not (0.9 < z < 1.11):
            sure = False
        self.A = A
        if not direct_ok or inl < 70 or self.overlap(A) < 0.55:     # new keyframe
            self.kf, self.A = feats, np.eye(3)
        self.prev_g, self.prev_m = g, m
        return H, sure


def analyse(path, t0, step_s, total, seg, track=None, progress=None, log=print, pull=False):
    """Frames at t0 + k*step_s (k < total). Returns dict fps, size, Hs (prev->cur, full px), feet, boxes, sure.
    Frames before 0 / after the end: no racer, no camera motion (sure False).
    pull=True (gate timing): the track is where the racer surely is (the coach's marked gates) — it also steers the
    choice while the racer is being followed, so a bystander picked by mistake is dropped again."""
    R = Reader(path)
    fps = 1.0 / step_s
    W = Hh = None
    s = S = Si = None
    cam = None
    Hs, feet, boxes, sure_l = [], [], [], []
    hist = []                     # [(k, [(foot, h), ...])] every person seen, moved with the camera
    back = max(1, int(0.4 * fps))
    last, last_i, last_h, vel = None, -999, None, (0.0, 0.0)
    T0 = time.time()
    for i in range(total):
        t = t0 + i * step_s
        frame, _, new = R.at(t)
        if frame is None or (not new and i > 0 and W is not None):
            # missing (before 0 / after the end) or the same frame again (lower frame rate): repeat
            Hs.append(np.eye(3))
            sure_l.append(frame is not None)
            feets = feet[-1] if (frame is not None and feet) else None
            feet.append(feets)
            boxes.append(boxes[-1] if (frame is not None and boxes) else None)
            hist.append((i, []))
            continue
        if W is None:
            Hh, W = frame.shape[:2]
            s = WORK / max(W, Hh)
            S, Si = scale_mat(s), scale_mat(1 / s)
        g = prep_gray(frame, s)
        if cam is None:
            cam = CamTrack(g.shape[1], g.shape[0])
        recent = i - last_i <= int(fps * 0.6)
        tp = track_pos(track, t)
        prior = (last[0] + vel[0], last[1] - (last_h or 100) * 0.5 + vel[1]) if (last is not None and recent) else None
        dets = detect(seg, frame, prior, last_h if recent else None)
        drift = pull and prior is not None and tp is not None and np.hypot(prior[0] / W - tp[0], prior[1] / Hh - tp[1]) > PULL_FAR
        if drift or (pull and prior is not None and not dets):
            # far from where the racer surely is, or nobody where we looked (the guess of where the racer went was
            # off — Hintertux 0225 lost the racer for 0.6 s before the finish): also look at the whole picture
            # (the crop's people stay candidates — the whole-picture pass can miss a small, blurred racer)
            dets = dets + detect(seg, frame, None, None)
        bg = skier_mask(g.shape)
        for (bx0, by0, bx1, by1), _, _, _ in dets:   # hide every person from the camera-motion estimate
            pad = 0.25 * (by1 - by0)
            cv2.rectangle(bg, (int((bx0 - pad) * s), int((by0 - pad) * s)), (int((bx1 + pad) * s), int((by1 + pad) * s)), 0, -1)
        Hw, sure = cam.step(g, bg, direct=SIFT_EVERY < 2 or i % SIFT_EVERY == 0)
        Hc = Si @ Hw @ S
        Hs.append(Hc)
        sure_l.append(sure)
        for k, (fi, lst) in enumerate(hist):
            hist[k] = (fi, [(apply(Hc, f), hh) for f, hh in lst])
        hist = [x for x in hist if x[0] >= i - back]
        if last is not None:
            last = apply(Hc, last)
        cands = []
        old = [x for x in hist if x[0] == i - back]
        for d in dets:
            (bx0, by0, bx1, by1), m, off, cf = d
            ft = foot_of(m, off)
            if ft is None:
                continue
            h = by1 - by0
            mv = None             # motion relative to the snow over ~0.4 s, in body heights (standing people ~0)
            if old and old[0][1]:
                mv = min(np.hypot(ft[0] - f[0], ft[1] - f[1]) for f, hh in old[0][1]) / max(h, 10)
            cands.append((d, ft, h, mv))
        hist.append((i, [(ft, h) for _, ft, h, _ in cands]))
        best, bscore = None, 1e9
        lost = []                 # pull: candidates refused only for being far from the one followed
        last_tp = np.hypot(last[0] / W - tp[0], last[1] / Hh - tp[1]) if (pull and tp is not None and last is not None) else 9.0
        for d, ft, h, mv in cands:
            score = -d[3]
            if mv is not None and mv < 0.15:
                score += 4                                 # standing still: almost never the racer
            near_tp = None
            if pull and tp is not None:
                near_tp = np.hypot(ft[0] / W - tp[0], ft[1] / Hh - tp[1])
                score += PULL * near_tp
            if last is not None and recent:
                exp = (last[0] + vel[0], last[1] + vel[1])
                dist = np.hypot(ft[0] - exp[0], ft[1] - exp[1])
                # far from the one followed: only taken when it is where the racer surely is — or, while the one
                # followed has drifted away from there (a bystander, seen on Hintertux 0225), anyone closer to it
                if dist > max(120, 2.5 * (last_h or h)) and not (near_tp is not None and (near_tp < PULL_NEAR or (drift and near_tp < PULL_FAR))):
                    if pull and not (mv is not None and mv < 0.15) and h >= 0.6 * (last_h or h) and                             (near_tp is None or near_tp < last_tp + 0.05):   # not someone smaller, nor further from the gate
                        lost.append((score, (d, ft, h)))     # (beta IMG_5815: a skier far below while the racer hid)
                    continue
                if pull and last_h and h < 0.5 * last_h and dist > last_h:
                    continue          # someone much smaller, well away: not the racer we follow (0225 finish)
                score += min(dist / max(h, 20), 6)
                if pull and last_h:
                    score += 3 * abs(np.log(max(h, 5) / max(last_h, 5)))   # the racer doesn't halve in size in a moment
            else:
                if tp is not None:
                    score += 3 * np.hypot(ft[0] / W - tp[0], ft[1] / Hh - tp[1])
                if mv is None:
                    score += 1
            if score < bscore:
                best, bscore = (d, ft, h), score
        if best is None and lost:
            # nobody where the one followed should be (it was a bystander who left, or the guess was off): take the
            # best moving person now instead of waiting 0.6 s (Hintertux 0225 lost the racer at the finish gate)
            best = min(lost, key=lambda x: x[0])[1]
        foot = None
        if best is not None:
            d, foot, h = best
            if last is not None and recent:
                k = i - last_i
                vel = (0.6 * vel[0] + 0.4 * (foot[0] - last[0]) / k, 0.6 * vel[1] + 0.4 * (foot[1] - last[1]) / k)
            else:
                vel = (0.0, 0.0)
            last, last_i, last_h = foot, i, h
        feet.append(foot)
        boxes.append(best[0][0] if best is not None else None)
        if progress and i % 10 == 0:
            progress(i / total)
        if i % 100 == 0:
            log(f"raceline analyse {i}/{total} {i / max(0.001, time.time() - T0):.1f} fps")
    feet = smooth_feet(Hs, feet)
    return {"fps": fps, "size": (W or 0, Hh or 0), "Hs": Hs, "feet": feet, "boxes": boxes, "sure": sure_l,
            "found": sum(f is not None for f in feet)}


def chain(Hs, k, j):
    """homography mapping frame k coords -> frame j coords"""
    M = np.eye(3)
    if k <= j:
        for q in range(k + 1, j + 1):
            M = Hs[q] @ M
        return M
    return np.linalg.inv(chain(Hs, j, k))


def smooth_feet(Hs, feet):
    out = []
    for j, f in enumerate(feet):
        if f is None:
            out.append(None)
            continue
        pts = [apply(chain(Hs, k, j), feet[k]) for k in range(max(0, j - SMOOTH), min(len(feet), j + SMOOTH + 1)) if feet[k] is not None]
        out.append((float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts]))))
    return out


def moving(Hs, feet, boxes, fps):
    """racer speed over the snow in body heights per second (None when unknown). Standing people ~0."""
    k = max(1, int(0.5 * fps))
    out = []
    for i, f in enumerate(feet):
        j = i - k
        if f is None or boxes[i] is None or j < 0 or feet[j] is None:
            out.append(None)
            continue
        p = apply(chain(Hs, j, i), feet[j])
        out.append(np.hypot(f[0] - p[0], f[1] - p[1]) / max(boxes[i][3] - boxes[i][1], 10) * fps / k)
    return out


def flat_view(a):
    """True when the camera looks straight up the slope: the line over 3 s is wide but hardly goes down the picture"""
    Hs, feet, boxes, fps = a["Hs"], a["feet"], a["boxes"], a["fps"]
    k = int(3 * fps)
    flat = n = 0
    for i in range(k, len(feet), max(1, int(fps / 2))):
        pts, M = [], np.eye(3)
        for j in range(i, i - k - 1, -1):      # walk back, building frame j -> frame i
            if feet[j] is not None:
                pts.append(apply(M, feet[j]))
            M = M @ Hs[j]
        if len(pts) < k / 2 or boxes[i] is None:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        bh = boxes[i][3] - boxes[i][1]
        if max(xs) - min(xs) < 3 * bh:
            continue
        n += 1
        flat += (max(ys) - min(ys)) < 0.18 * (max(xs) - min(xs))
    return bool(n > 0 and flat / n > 0.5)


class Line:
    """The race line of one video, kept in that video's current-frame coordinates. step(i) once per frame, then draw()."""

    def __init__(self, a, tail=None):
        self.fps, self.W, self.Hs, self.feet, self.sure = a["fps"], a["size"][0], a["Hs"], a["feet"], a["sure"]
        self.tail = tail
        self.speed = moving(a["Hs"], a["feet"], a["boxes"], a["fps"])
        self.segs = []            # list of lists of [x, y, frame]
        self.last_i = -999
        self.i, self.foot = -1, None
        self.drawn = 0

    def step(self, i):
        if i >= len(self.Hs):
            self.foot = None
            return None
        fps = self.fps
        if not self.sure[i] or i - self.last_i > 1.5 * fps:
            self.segs = []     # camera tracking lost (fast pan, blur) or racer gone: drop the old line rather than draw it in the wrong place
        for seg in self.segs:
            if seg:
                a = cv2.perspectiveTransform(np.float32([q[:2] for q in seg]).reshape(-1, 1, 2), self.Hs[i]).reshape(-1, 2)
                for q, (x, y) in zip(seg, a):
                    q[0], q[1] = float(x), float(y)
        f = self.feet[i]
        # only draw while the "racer" moves steadily over the snow, >= 1.5 body heights/s for 0.5 s
        win = [v for v in self.speed[max(0, i - int(0.5 * fps)):i + 1] if v is not None]
        steady = len(win) >= 0.6 * 0.5 * fps and float(np.median(win)) >= 1.5 and min(win) >= 0.5
        if f is not None and steady:
            segs = self.segs
            jump = segs and segs[-1] and np.hypot(f[0] - segs[-1][-1][0], f[1] - segs[-1][-1][1]) > 0.06 * self.W * (i - self.last_i)
            if i - self.last_i > int(GAP_S * fps) or not segs or jump:
                segs.append([])
            segs[-1].append([f[0], f[1], i])
            self.last_i = i
        else:
            f = None
        if self.tail:
            self.segs = [s for s in ([q for q in seg if i - q[2] <= self.tail * fps] for seg in self.segs) if s]
        self.i, self.foot = i, f
        return f

    @staticmethod
    def project(part, M, shape):
        """points -> frame coords (polyline pieces, shift 2). Points that land behind the camera or far outside the
        picture (a homography blows up near its horizon) split the line instead of drawing a long straight stroke."""
        h, w = shape[:2]
        P = np.hstack([np.float64([q[:2] for q in part]), np.ones((len(part), 1))]) @ M.T
        ok = P[:, 2] > 1e-6
        xy = np.zeros((len(part), 2))
        xy[ok] = P[ok, :2] / P[ok, 2:3]
        ok &= np.isfinite(xy).all(1) & (np.abs(xy[:, 0] - w / 2) < 1.5 * w) & (np.abs(xy[:, 1] - h / 2) < 1.5 * h)
        pieces, cur = [], []
        for good, p in zip(ok, xy):
            if good:
                cur.append(p)
            elif cur:
                pieces.append(cur)
                cur = []
        if cur:
            pieces.append(cur)
        return [np.round(np.float64(c) * 4).astype(np.int32).reshape(-1, 1, 2) for c in pieces if len(c) >= 2]

    @staticmethod
    def smoothed(seg, h):
        """seg's points averaged over h points each side (Gaussian), the window shrinking towards both ends so the line
        still starts and ends exactly where the racer was (no lag behind the feet, no corners cut at the ends)"""
        P = np.float64([q[:2] for q in seg])
        n = len(P)
        if h < 1 or n < 3:
            return [[x, y, q[2]] for (x, y), q in zip(P, seg)]
        out = np.empty_like(P)
        for k in range(n):
            r = min(h, k, n - 1 - k)
            w = np.exp(-0.5 * (np.arange(-r, r + 1) / max(1.0, h / 2)) ** 2)
            out[k] = (P[k - r:k + r + 1] * w[:, None]).sum(0) / w.sum()
        return [[x, y, q[2]] for (x, y), q in zip(out, seg)]

    def draw(self, frame, color, th, M):
        """M: this video's frame coords -> frame's coords"""
        fps, i, tail = self.fps, self.i, self.tail
        lim = 1.5 * max(frame.shape[:2])
        buckets = [(0, 1e9, 1.0)] if not tail else [(0, 0.5 * tail, 1.0), (0.5 * tail, 0.75 * tail, 0.6), (0.75 * tail, tail + 1, 0.3)]
        h = int(round(LINE_SMOOTH_S * fps / 2))
        segs = [self.smoothed(seg, h) for seg in self.segs if len(seg) >= 0.5 * fps]  # pieces shorter than 0.5 s: usually someone who isn't the racer
        pieces = []
        for lo, hi, alpha in buckets:
            ps = []
            for seg in segs:
                part = [q for q in seg if lo * fps <= i - q[2] <= hi * fps + 1]
                if len(part) >= 2:
                    ps += self.project(part, M, frame.shape)
            if ps:
                pieces.append((ps, alpha))
        edge = 4
        if pieces and GLOW > 0:
            th, edge = max(2, round(0.7 * th)), 2     # with the glow a thinner line with a thin white edge looks cleaner
            # glow: the line's colour blended into the picture round it (additive light would vanish on white snow)
            s = 4
            small = np.zeros((frame.shape[0] // s, frame.shape[1] // s), np.float32)
            for ps, alpha in pieces:
                cv2.polylines(small, [p // s for p in ps], False, alpha, max(1, (2 * th) // s), cv2.LINE_AA, shift=2)
            k = max(3, (3 * th) // s) | 1
            small = cv2.GaussianBlur(small, (k, k), 0)
            a = cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)[..., None] * GLOW
            frame[:] = (frame * (1 - a) + np.float32(color) * a).astype(np.uint8)
        for ps, alpha in pieces:
            layer = frame.copy() if alpha < 1 else frame
            for p in ps:
                cv2.polylines(layer, [p], False, WHITE, th + edge, cv2.LINE_AA, shift=2)
                cv2.polylines(layer, [p], False, color, th, cv2.LINE_AA, shift=2)
            if alpha < 1:
                cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, dst=frame)
        any_drawn = bool(pieces)
        if self.foot is not None:
            x, y = apply(M, self.foot)
            if np.isfinite(x) and np.isfinite(y) and abs(x) < lim and abs(y) < lim:
                cv2.circle(frame, (int(x), int(y)), th + 3, WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), th + 1, color, -1, cv2.LINE_AA)
        self.drawn += any_drawn
        return any_drawn


class Delta:
    """Time gap between video 1's racer and the ghost at the same spot on the course, counted from the locked moment
    (the fixed point where both were lined up, gap 0). + = video 1 is behind.
    Followed step by step: j = the ghost's frame that matches video 1's racer now, i.e. the ghost's foot point closest
    to video 1's racer. j never goes back, may stay at most 2 frames and goes forward at most 0.5 s per frame, with
    a small pull towards an unchanged gap, so the zig-zag of the line crossing itself in the picture can't make it jump;
    after 0.3 s without a match it looks within 2 s. Only points where the ghost moves count (not people standing).
    The ghost's whole analysed path is known in advance, so it also works while video 1 is ahead. No match: j moves on
    with the clock (gap unchanged)."""

    def __init__(self, an_a, an_b, fps):
        self.fps, self.feetA, self.feetB = fps, an_a["feet"], an_b["feet"]
        speed = moving(an_b["Hs"], an_b["feet"], an_b["boxes"], fps)
        self.okB = [f is not None and v is not None and v >= 0.5 for f, v in zip(an_b["feet"], speed)]
        G = [np.eye(3)]                         # ghost frame 0 -> frame j
        for Hk in an_b["Hs"][1:]:
            G.append(Hk @ G[-1])
        self.G = G
        self.j = 0.0
        self.miss = self.same = 0
        self.vals = []                          # (frame, gap s)
        self.shown = None

    def step(self, k, MA, MB, h):
        """MA / MB: video 1 / ghost frame coords -> output coords (MB None while the two aren't lined up)"""
        fa = self.feetA[k] if k < len(self.feetA) else None
        jp = int(round(self.j))
        best = None
        if MB is not None and fa is not None and k < len(self.G):
            pa = np.float64(apply(MA, fa))
            Gk = self.G[k]
            wide = self.miss > 0.3 * self.fps
            lo, hi = (jp - int(2 * self.fps), jp + int(2 * self.fps)) if wide else (jp, jp + 1 + int(0.5 * self.fps))
            for j in range(max(0, lo), min(len(self.feetB), hi + 1)):
                if not self.okB[j]:
                    continue
                pb = apply(MB @ Gk @ np.linalg.inv(self.G[j]), self.feetB[j])   # ghost's foot at frame j, as seen now
                d = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
                c = d + 0.004 * h * abs(j - jp - 1)       # small pull towards "gap unchanged"
                if best is None or c < best[2]:
                    best = (d, j, c)
            if best is not None and best[0] >= (0.06 if wide else 0.12) * h:
                best = None
        if best is not None and best[1] == jp:
            self.same += 1
            if self.same > 2:                   # stuck on one ghost frame while video 1 moves on: not a real match
                best = None
        else:
            self.same = 0
        if best is not None:
            self.j, self.miss = float(best[1]), 0
        else:
            self.j += 1.0
            self.miss += 1
        if k >= 0.5 * self.fps:                 # a moment to settle after the lock
            self.vals.append((k, (k - self.j) / self.fps))
        self.vals = [v for v in self.vals if k - v[0] <= 0.4 * self.fps]
        if len(self.vals) >= 3:
            self.shown = float(np.median([v[1] for v in self.vals]))
        return self.shown


def line_for(a):
    return Line(a, tail=3.0 if flat_view(a) else None)


def render_raceline(path, out_path, *, start, seg, track=None, name="", max_seconds=MAX_SECONDS, progress=None, log=print):
    """One video with the racer's line (job kind "raceline"). Analysis 0–70 %, drawing 70–100 %."""
    T0 = time.time()
    R = Reader(path)
    fps_in = R.fps
    step = 2 if fps_in > 32 else 1
    fps = fps_in / step
    start = max(0.0, float(start))
    if R.duration and start >= R.duration - 1:
        raise OverlayError("The start point is at the very end of the video.")
    total = max(1, int(min(max_seconds, (R.duration - start) if R.duration else max_seconds) * fps))
    a = analyse(path, start, step / fps_in, total, seg, track, progress=(lambda p: progress(0.7 * p)) if progress else None, log=log)
    if a["found"] < 0.2 * total:
        raise OverlayError("Couldn't find the racer in this video, so there is no line to draw.")
    flat = flat_view(a)
    line = Line(a, tail=3.0 if flat else None)
    W, Hh = a["size"]
    so = min(1.0, OUT_MAX_W / max(W, Hh) if W >= Hh else 1080 / W)
    oW, oH = int(round(W * so)) // 2 * 2, int(round(Hh * so)) // 2 * 2
    M = scale_mat(so)
    th = max(3, round(oW / 380))
    ui = oH / 1080.0 if oW >= oH else oW / 1080.0
    lab = label_image((name or "Race line").upper(), int(44 * ui), (232, 0, 29))
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{oW}x{oH}", "-r", f"{fps:.6f}",
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)
    R = Reader(path)
    frames = 0
    try:
        for k in range(len(a["Hs"])):
            frame, _, _ = R.at(start + k * step / fps_in)
            if frame is None:
                break
            if (oW, oH) != (W, Hh):
                frame = cv2.resize(frame, (oW, oH), interpolation=cv2.INTER_AREA)
            line.step(k)
            line.draw(frame, RED, th, M)
            paste(frame, lab, int(40 * ui), int(40 * ui))
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
    return {"kind": "raceline", "frames": frames, "fps": round(fps, 2), "seconds": round(frames / fps, 1),
            "found_pct": round(100 * a["found"] / max(1, total), 1), "line_pct": round(100 * line.drawn / max(1, frames), 1),
            "flat": bool(flat), "render_s": round(time.time() - T0, 1), "size": [oW, oH]}
