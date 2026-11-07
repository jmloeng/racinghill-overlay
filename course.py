"""SET COURSE (beta, job kind "course"): start / split / finish gate marked once on a reference video, found again in
every other video of the session filmed from the same spot, and the racer's times through them.

Owner's flow (2026-11-07): the coach taps once on each gate (start, split, finish) in the reference video; the image
analysis finds the two pole feet, so the direction of the gate line comes from the poles, not from the coach. Then:

  ref_gates()   reference video: a clean background around each tap (frames ±1.2 s lined up on the tapped frame,
                per-pixel median → the racer is gone), the panel next to the tap, the pole on each side edge of the
                panel followed down to the snow → the two pole feet. sure = False when the two feet look wrong
                (very different heights) → the app asks the coach to drag them.
  locate()      other video: every 0.1 s the picture is matched to the reference frame of each gate (SIFT, like the
                overlay's same-spot check). Best match → the reference feet are carried into that frame (homography),
                then refined on the poles found there (a camera that moved a little sees the near gate shifted
                against the far background). No gate matches → filmed from another spot → no times.
  windows       the crossing is looked for while the gate is in the picture (gate carried with the video's own camera
                motion from the matched frame), between the neighbouring gates.
  run_video()   racer analysis at full frame rate (raceline.analyse, pull towards the matched split / finish gates)
                + gates.passes() (crossing of the feet with the gate line, valid = between the poles or ≤ 5 m beyond)
                → videos.course: per gate status / time / ± and the gate's position over time (pts) so the player can
                draw the markers following the camera; start→split, split→finish, total.

SL (single poles, owner 2026-11-07: segment from the pole to the turn apex of the racers' average line) is not
handled yet: GS / SG / DH only.
"""
import time

import cv2
import numpy as np

from overlay import OverlayError, Reader, prep_gray, sift_feats, match_h, skier_mask, h_sane, scale_mat
from raceline import analyse, chain, apply
import gates as G

VERSION = 1
LABELS = ("start", "split", "finish")
PLATE_S = 1.2          # frames this far either side of the tap are lined up for the clean background
PLATE_N = 21
MATCH_STEP_S = 0.1     # how often another video's picture is compared with the reference gates
MATCH_WORK = 960       # px (long side) for that comparison
MIN_INLIERS = 30       # fewer matching points than this = gate not seen in that video
MAX_WIN_S = 4.0        # crossing search at most this far from the matched moment
PAD_S = 1.0
REF_WIN_S = 2.0      # reference video: the crossing within this of the coach's pause at the gate


# ───────────────────────── pole feet from one tap ─────────────────────────

def read_frames(path, times):
    """{t: frame} for the given times (one forward pass; frames before 0 / after the end are left out)."""
    R = Reader(path)
    out = {}
    for t in sorted(set(round(float(t), 4) for t in times)):
        f, _, _ = R.at(t)
        if f is not None:
            out[t] = f.copy()
    return out, R.fps


def plate(frames, t, tap, R=300):
    """Clean background around tap (px) in the frame at t: the other frames lined up on it with SIFT, per-pixel
    median. Returns (image the size of the frame — only the area around the tap is cleaned, used frame count)."""
    ref = frames[t]
    H, W = ref.shape[:2]
    s = MATCH_WORK / max(H, W)
    gr = prep_gray(ref, s)
    fr = sift_feats(gr, skier_mask(gr.shape))
    S = scale_mat(s)
    Si = np.linalg.inv(S)
    x0, y0 = max(int(tap[0]) - R, 0), max(int(tap[1]) - int(1.3 * R), 0)
    x1, y1 = min(int(tap[0]) + R, W), min(int(tap[1]) + R, H)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    stack = [ref[y0:y1, x0:x1].astype(np.float32)]
    for tt, f in frames.items():
        if tt == t:
            continue
        g = prep_gray(f, s)
        Hm, inl = match_h(sift_feats(g, skier_mask(g.shape)), fr)
        if Hm is None or inl < 40 or not h_sane(Hm, g.shape[1], g.shape[0], gr.shape[1], gr.shape[0]):
            continue
        Hf = T @ Si @ Hm @ S
        w = cv2.warpPerspective(f, Hf, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR).astype(np.float32)
        valid = cv2.warpPerspective(np.full(f.shape[:2], 255, np.uint8), Hf, (x1 - x0, y1 - y0)) > 0
        w[~valid] = np.nan
        stack.append(w)
    st = np.stack(stack)
    with np.errstate(all="ignore"):
        med = np.nanmedian(st, axis=0)
    med = np.where(np.isnan(med), st[0], med).astype(np.uint8)
    out = ref.copy()
    out[y0:y1, x0:x1] = med
    return out, len(stack) - 1


def _x_at(fit, y):
    return float(np.polyval(fit, y))


def _features(roi, sc):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S = cv2.GaussianBlur(hsv[..., 1], (3, 3), 0)
    V = cv2.GaussianBlur(hsv[..., 2], (3, 3), 0)
    k = max(5, int(11 * sc)) | 1
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    ts = cv2.morphologyEx(S, cv2.MORPH_TOPHAT, hk).astype(np.int16)     # thin and more coloured than beside it
    bv = cv2.morphologyEx(V, cv2.MORPH_BLACKHAT, hk).astype(np.int16)   # thin and darker than beside it
    return np.clip(np.maximum(ts, bv), 0, 255).astype(np.uint8), S, bv


def _sky(roi, S, sc):
    """Blue sky is as coloured as a panel and swallows panels at the horizon (upright phone videos of a far-away
    course): the big coloured area touching the top of the search box, and every pixel of the same colour, is sky."""
    m = (S > 70).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    H, W = S.shape
    big = [i for i in range(1, n) if st[i][1] == 0 and st[i][4] > 0.08 * H * W]
    if not big:
        return None
    lab_img = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    sky = np.zeros((H, W), bool)
    for i in big:
        sel = lab_img[lab == i]
        c = np.median(sel, axis=0)
        spread = np.percentile(np.linalg.norm(sel - c, axis=1), 90)
        sky |= np.linalg.norm(lab_img - c, axis=2) < max(14.0, 1.3 * spread)
    return sky


def _panel(S, tx, ty, sc, sky=None):
    m = (S > 70).astype(np.uint8)
    if sky is not None:
        m[sky] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((max(3, int(5 * sc)),) * 2, np.uint8))   # thin poles gone
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 160 * sc * sc or w < 12 * sc or h < 8 * sc or area / (w * h) < 0.35 or w > 220 * sc or h > 220 * sc:
            continue
        dx = max(x - tx, 0, tx - (x + w))
        dy = max(y - ty, 0, ty - (y + h))
        d = np.hypot(dx, 0.6 * dy)
        if best is None or d < best[0]:
            best = (d, i, (x, y, w, h))
    if best is None or best[0] > 160 * sc:
        return None
    return lab == best[1], best[2]


def _feet_panel(frame, tap, _again=True):
    """frame (a clean plate is best), tap px anywhere on the gate → ([(xa, ya), (xb, yb)] left/right pole foot px,
    sure) or (None, False). GS-style gates: a panel between two poles."""
    H, W = frame.shape[:2]
    sc = min(H, W) / 1080          # short side: an upright 1080x1920 phone video is "1080p" too
    R = int(230 * sc)
    x0, y0 = max(int(tap[0]) - R, 0), max(int(tap[1]) - int(1.3 * R), 0)
    x1, y1 = min(int(tap[0]) + R, W), min(int(tap[1]) + R, H)
    roi = frame[y0:y1, x0:x1]
    F, S, BV = _features(roi, sc)
    pan = _panel(S, tap[0] - x0, tap[1] - y0, sc, _sky(roi, S, sc))
    if pan is None:
        return None, False
    pm, (px, py, pw, ph) = pan
    if _again:        # the same answer wherever on the gate the coach tapped: look again centred on the panel
        return _feet_panel(frame, (px + pw / 2 + x0, py + ph / 2 + y0), False)
    rows = np.flatnonzero(pm.any(1))
    r0, r1 = rows.min(), rows.max()
    edges = []
    for yy in range(r0 + (r1 - r0) // 5, r1 - (r1 - r0) // 5 + 1):
        xs = np.flatnonzero(pm[yy])
        if len(xs):
            edges.append((yy, xs.min(), xs.max()))
    if not edges:
        return None, False
    edges = np.array(edges, float)
    B = (F > 16).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(B, 1, np.pi / 360, threshold=int(12 * sc) + 4, minLineLength=int(18 * sc) + 4,
                           maxLineGap=int(5 * sc) + 2)
    segs = [] if segs is None else segs.reshape(-1, 4).astype(float)
    feet = []
    for side in (0, 1):
        ecx = float(np.median(edges[:, 1] if side == 0 else edges[:, 2]))
        ecy = float(np.median(edges[:, 0]))
        cands = []
        for x_1, y_1, x_2, y_2 in segs:
            if y_2 < y_1:
                x_1, y_1, x_2, y_2 = x_2, y_2, x_1, y_1
            if y_2 - y_1 <= 0 or abs(x_2 - x_1) > (y_2 - y_1):         # within 45° of upright
                continue
            fit = np.polyfit([y_1, y_2], [x_1, x_2], 1)
            if abs(_x_at(fit, ecy) - ecx) > 6 * sc + 3:                 # along the panel's side edge
                continue
            if y_2 < r1 - 2:                                             # reaching below the panel
                continue
            cands.append((y_2 - max(y_1, r1), (x_1, y_1, x_2, y_2), fit))
        if not cands:
            return None, False
        cands.sort(key=lambda c: -c[0])
        bfit = cands[0][2]
        pts = [(ecx, ecy)]
        for _, l, _ in cands:
            if abs(_x_at(bfit, l[1]) - l[0]) < 4 * sc + 2 and abs(_x_at(bfit, l[3]) - l[2]) < 4 * sc + 2:
                pts += [(l[0], l[1]), (l[2], l[3])]
        pts = np.array(pts)
        fit = np.polyfit(pts[:, 1], pts[:, 0], 1)
        # follow the pole down to the snow: coloured or darker than the snow
        y = pts[:, 1].max()
        last, miss = y, 0
        while y + 1 < F.shape[0] and miss <= 3 * sc + 2:
            y += 1
            x = int(round(_x_at(fit, y)))
            if not (1 <= x < F.shape[1] - 1):
                break
            yi = int(y)
            if S[yi, x - 1:x + 2].max() > 55 or BV[yi, x - 1:x + 2].max() > 28:
                last, miss = y, 0
            else:
                miss += 1
        last = min(last, r1 + 2.2 * max(pw, 20 * sc))
        feet.append((_x_at(fit, last) + x0, float(last) + y0))
    (lx, ly), (rx, ry) = feet
    gap = np.hypot(rx - lx, ry - ly)
    if gap < 20 * sc or rx <= lx:
        return None, False
    hl, hr = ly - r1 - y0, ry - r1 - y0
    sure = bool(min(hl, hr) > 0.2 * gap and max(hl, hr) / max(min(hl, hr), 1) < 1.8 and abs(ly - ry) < 0.55 * gap)
    return [(float(lx), float(ly)), (float(rx), float(ry))], sure


def _walk_down(fit, y, F, S, BV, sc):
    """Follow a pole (x = fit(y)) down from y to where it meets the snow: coloured or darker than the snow."""
    last, miss = y, 0
    while y + 1 < F.shape[0] and miss <= 3 * sc + 2:
        y += 1
        x = int(round(_x_at(fit, y)))
        if not (1 <= x < F.shape[1] - 1):
            break
        yi = int(y)
        if S[yi, x - 1:x + 2].max() > 55 or BV[yi, x - 1:x + 2].max() > 28:
            last, miss = y, 0
        else:
            miss += 1
    return last


def _feet_poles(frame, tap):
    """No panel found (white panels, blue panels against blue sky, shadow on the snow): look for the gate's two poles
    directly — a pair of thin, nearly upright, similar lines with the tap between them (on the panel or at the feet)."""
    H, W = frame.shape[:2]
    sc = min(H, W) / 1080
    R = int(230 * sc)
    x0, y0 = max(int(tap[0]) - R, 0), max(int(tap[1]) - int(1.3 * R), 0)
    x1, y1 = min(int(tap[0]) + R, W), min(int(tap[1]) + R, H)
    roi = frame[y0:y1, x0:x1]
    F, S, BV = _features(roi, sc)
    tx, ty = tap[0] - x0, tap[1] - y0
    B = (F > 16).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(B, 1, np.pi / 360, threshold=int(12 * sc) + 4, minLineLength=int(14 * sc) + 4,
                           maxLineGap=int(5 * sc) + 2)
    if segs is None:
        return None, False
    # join the pieces of one pole: same line (x at two heights within a few px)
    poles = []
    for x_1, y_1, x_2, y_2 in segs.reshape(-1, 4).astype(float):
        if y_2 < y_1:
            x_1, y_1, x_2, y_2 = x_2, y_2, x_1, y_1
        if y_2 - y_1 <= 0 or abs(x_2 - x_1) > 0.58 * (y_2 - y_1):        # within 30° of upright
            continue
        fit = np.polyfit([y_1, y_2], [x_1, x_2], 1)
        for pl in poles:
            if all(abs(_x_at(pl["fit"], yy) - _x_at(fit, yy)) < 3 * sc + 2 for yy in (y_1, y_2)) and \
                    y_1 < pl["y1"] + 25 * sc and y_2 > pl["y0"] - 25 * sc:
                pl["pts"] += [(x_1, y_1), (x_2, y_2)]
                pl["y0"], pl["y1"] = min(pl["y0"], y_1), max(pl["y1"], y_2)
                q = np.array(pl["pts"])
                pl["fit"] = np.polyfit(q[:, 1], q[:, 0], 1)
                break
        else:
            poles.append({"fit": fit, "y0": y_1, "y1": y_2, "pts": [(x_1, y_1), (x_2, y_2)]})
    poles = [p for p in poles if p["y1"] - p["y0"] >= 22 * sc]
    for p in poles:
        p["foot"] = _walk_down(p["fit"], p["y1"], F, S, BV, sc)
        p["xm"] = _x_at(p["fit"], (p["y0"] + p["foot"]) / 2)
    best = None
    for L in poles:
        for Rp in poles:
            if Rp is L or Rp["xm"] <= L["xm"]:
                continue
            gap = np.hypot(_x_at(Rp["fit"], Rp["foot"]) - _x_at(L["fit"], L["foot"]), Rp["foot"] - L["foot"])
            hl, hr = L["foot"] - L["y0"], Rp["foot"] - Rp["y0"]
            if not (14 * sc <= gap <= 260 * sc) or abs(L["fit"][0] - Rp["fit"][0]) > 0.6:          # poles often splay
                continue
            if abs(L["foot"] - Rp["foot"]) > 0.8 * gap or max(hl, hr) / max(min(hl, hr), 1) > 2.5 or min(hl, hr) < 0.5 * gap:
                continue
            top = max(L["y0"], Rp["y0"])
            if min(L["foot"], Rp["foot"]) - top < 0.4 * min(hl, hr):                   # side by side, not one above
                continue
            # how far the tap is from the gate (the area between the poles, from the top down to the feet + a bit)
            yb = max(L["foot"], Rp["foot"]) + 0.25 * gap
            xl, xr = _x_at(L["fit"], ty), _x_at(Rp["fit"], ty)
            dx = max(xl - tx, 0, tx - xr)
            dy = max(top - ty, 0, ty - yb)
            d = np.hypot(dx, dy) + 0.02 * gap + 0.3 * abs(L["foot"] - Rp["foot"])     # tie: feet level, smaller gate
            if best is None or d < best[0]:
                best = (d, L, Rp, gap)
    if best is None or best[0] > 45 * sc:
        return None, False
    d, L, Rp, gap = best
    diff = abs(L["foot"] - Rp["foot"])
    sure = bool(d < 10 * sc + 0.02 * gap + 0.3 * diff and diff < 0.25 * gap)
    if diff > 0.25 * gap:      # one pole faded out above the snow: carry it on along its own line to the other's foot
        lo = max(L["foot"], Rp["foot"])
        L, Rp = [dict(p, foot=lo) if p["foot"] < lo else p for p in (L, Rp)]
    feet = [(_x_at(L["fit"], L["foot"]) + x0, float(L["foot"]) + y0), (_x_at(Rp["fit"], Rp["foot"]) + x0, float(Rp["foot"]) + y0)]
    return feet, sure


def find_feet(frame, tap):
    """frame (a clean plate is best), tap px anywhere on the gate → ([(xa, ya), (xb, yb)] left/right pole foot px,
    sure) or (None, False). The panel first (coloured GS panels), else the pair of poles around the tap."""
    feet, sure = _feet_panel(frame, tap)
    if feet is not None:
        return feet, sure
    return _feet_poles(frame, tap)


def feet_at(path, t, tap, frames=None):
    """Pole feet for a tap (px) in the frame at t of the video: (feet, sure, frame used)."""
    if frames is None:
        times = [t + d for d in np.linspace(-PLATE_S, PLATE_S, PLATE_N)] + [t]
        frames, _ = read_frames(path, times)
    t = round(float(t), 4)
    if t not in frames:
        return None, False, None
    pl, _ = plate(frames, t, tap)
    feet, sure = find_feet(pl, tap)
    return feet, sure, frames[t]


def ref_gates(path, gates, log=print):
    """gates [{label, t, tap: [x, y] 0..1}] (or with a, b already set by the coach's drag) → [{label, t, a, b (0..1),
    sure, auto}] plus the reference picture of each gate for locate()."""
    R = Reader(path)
    W, H = int(R.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(R.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    times = []
    for g in gates:
        times += [g["t"] + d for d in np.linspace(-PLATE_S, PLATE_S, PLATE_N)] + [g["t"]]
    frames, _ = read_frames(path, times)
    out = []
    for g in gates:
        t = round(float(g["t"]), 4)
        if t not in frames:
            raise OverlayError(f"The {g['label']} gate is past the end of the video.")
        if g.get("a") and g.get("b"):           # the coach dragged the feet: keep them
            feet = [(g["a"][0] * W, g["a"][1] * H), (g["b"][0] * W, g["b"][1] * H)]
            sure, auto = True, False
        else:
            tap = (g["tap"][0] * W, g["tap"][1] * H)
            sub = {k: v for k, v in frames.items() if abs(k - t) <= PLATE_S + 1e-3}
            pl, used = plate(sub, t, tap)
            feet, sure = find_feet(pl, tap)
            auto = True
            log(f"course: {g['label']} gate at {t:.2f}s tap {tap[0]:.0f},{tap[1]:.0f} -> feet {feet} sure {sure} ({used} frames)")
            if feet is None:
                # a guess the coach can drag into place (••• → Adjust course gates, orange = not sure): a level line
                # one pole gap wide at the tap — better than no course at all (beta 07.11: the whole course failed)
                half = 28 * min(W, H) / 1080
                feet, sure = [(tap[0] - half, tap[1]), (tap[0] + half, tap[1])], False
                log(f"course: {g['label']} gate: poles not found, level guess at the tap")
        out.append({"label": g["label"], "t": t, "a": [feet[0][0] / W, feet[0][1] / H], "b": [feet[1][0] / W, feet[1][1] / H],
                    "sure": sure, "auto": auto, "_frame": frames[t]})
    return out


# ───────────────────────── the same gates in another video ─────────────────────────

def _ref_feats(refs):
    out = []
    for r in refs:
        f = r["_frame"]
        s = MATCH_WORK / max(f.shape[:2])
        g = prep_gray(f, s)
        out.append((sift_feats(g, skier_mask(g.shape)), g.shape, s))
    return out


def locate(path, refs, log=print):
    """Each reference gate in this video: [{label, t (best matching moment), a, b (0..1, refined on the poles), inl,
    span, refined}] — None for a gate not seen. Also returns the video's size and fps."""
    feats = _ref_feats(refs)
    R = Reader(path)
    fps = R.fps
    W, H = int(R.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(R.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    every = max(1, int(round(MATCH_STEP_S * fps)))
    prof = [[] for _ in refs]
    k = 0
    while True:
        ok = R.cap.grab()
        if not ok:
            break
        if k % every == 0:
            ok, f = R.cap.retrieve()
            if ok:
                s = MATCH_WORK / max(f.shape[:2])
                g = prep_gray(f, s)
                fb = sift_feats(g, skier_mask(g.shape))
                for j, (fr, shp, sr) in enumerate(feats):
                    Hm, inl = match_h(fb, fr)
                    good = Hm is not None and h_sane(Hm, g.shape[1], g.shape[0], shp[1], shp[0])
                    prof[j].append((k / fps, inl if good else 0, Hm if good else None, s, sr))
        k += 1
    found = []
    for j, r in enumerate(refs):
        p = prof[j]
        if not p:
            found.append(None)
            continue
        bi = max(range(len(p)), key=lambda q: p[q][1])
        t, inl, Hm, s, sr = p[bi]
        log(f"course: {r['label']} gate best match at {t:.1f}s, {inl} points")
        if Hm is None or inl < MIN_INLIERS:
            found.append(None)
            continue
        rW, rH = r["_frame"].shape[1], r["_frame"].shape[0]
        Hf = np.linalg.inv(scale_mat(s)) @ np.linalg.inv(Hm) @ scale_mat(sr)      # reference px → this video px
        A = cv2.perspectiveTransform(np.float32([[[r["a"][0] * rW, r["a"][1] * rH]]]), Hf)[0, 0]
        B = cv2.perspectiveTransform(np.float32([[[r["b"][0] * rW, r["b"][1] * rH]]]), Hf)[0, 0]
        found.append({"label": r["label"], "t": round(t, 3), "inl": int(inl), "A": (float(A[0]), float(A[1])), "B": (float(B[0]), float(B[1]))})
    # refine on the poles seen in this video (frames around each matched moment)
    times = []
    for g in found:
        if g:
            times += [g["t"] + d for d in np.linspace(-PLATE_S, PLATE_S, PLATE_N)] + [g["t"]]
    frames, _ = read_frames(path, times) if times else ({}, fps)
    out = []
    for g in found:
        if g is None:
            out.append(None)
            continue
        A, B = np.array(g["A"]), np.array(g["B"])
        gap = float(np.linalg.norm(B - A))
        mid = (A + B) / 2
        sub = {k: v for k, v in frames.items() if abs(k - round(g["t"], 4)) <= PLATE_S + 1e-3}
        refined = False
        if round(g["t"], 4) in sub:
            pl, _ = plate(sub, round(g["t"], 4), (mid[0], mid[1] - 0.9 * gap))
            tap = (mid[0], mid[1] - 0.9 * gap)
            # panel method only: the pole-pair search (no panel found) moved good carried feet on Hintertux 0230
            feet, sure = _feet_panel(pl, tap)
            near = 0.6
            if feet is not None:
                dev = max(np.hypot(*(np.array(feet[0]) - A)), np.hypot(*(np.array(feet[1]) - B)))
                g2 = np.hypot(*(np.array(feet[1]) - np.array(feet[0])))
                if dev < near * max(gap, g2) and 0.6 < g2 / max(gap, 1) < 1.6:
                    A, B, refined = np.array(feet[0]), np.array(feet[1]), True
        out.append({"label": g["label"], "t": g["t"], "a": [A[0] / W, A[1] / H], "b": [B[0] / W, B[1] / H],
                    "inl": g["inl"], "refined": refined})
    return out, (W, H), fps


def set_windows(a, marks, start, max_out_s=0.6):
    """mark["win"] = [t0, t1]: while the gate (carried with the camera from its matched frame) is in the picture —
    short stretches outside (the camera tracking jumping for a moment, seen on Hintertux 0230) don't end it — at most
    MAX_WIN_S from the matched moment, not past the neighbouring gates' moments."""
    fps, Hs = a["fps"], a["Hs"]
    W, H = a["size"]
    n = len(Hs)
    for j, mk in enumerate(marks):
        i = max(0, min(n - 1, int(round((mk["t"] - start) * fps))))
        A0 = np.array([mk["a"][0] * W, mk["a"][1] * H])
        B0 = np.array([mk["b"][0] * W, mk["b"][1] * H])
        lo_lim = marks[j - 1]["t"] if j > 0 else -1e9
        hi_lim = marks[j + 1]["t"] if j + 1 < len(marks) else 1e9

        def inside(k):
            M = chain(Hs, i, k)
            for P in (apply(M, A0), apply(M, B0)):
                if not (-0.05 * W <= P[0] <= 1.05 * W and -0.05 * H <= P[1] <= 1.05 * H):
                    return False
            return True
        gap = int(max_out_s * fps)

        def walk(d):
            k, last_in, out = i, i, 0
            while 0 <= k + d < n and abs(k + d - i) / fps < MAX_WIN_S and lo_lim < start + (k + d) / fps < hi_lim:
                k += d
                if inside(k):
                    last_in, out = k, 0
                else:
                    out += 1
                    if out > gap:
                        break
            return last_in
        mk["win"] = [start + walk(-1) / fps, start + walk(1) / fps]


def gate_path(a, mk, start, every_s=0.1):
    """The gate's pole feet over time (while in the picture) for the player: [[t, ax, ay, bx, by], ...] 0..1."""
    fps, Hs = a["fps"], a["Hs"]
    W, H = a["size"]
    n = len(Hs)
    i = max(0, min(n - 1, int(round((mk["t"] - start) * fps))))
    A0 = np.array([mk["a"][0] * W, mk["a"][1] * H])
    B0 = np.array([mk["b"][0] * W, mk["b"][1] * H])
    step = max(1, int(round(every_s * fps)))
    pts = []
    for k in range(0, n, step):
        M = chain(Hs, i, k)
        A, B = apply(M, A0), apply(M, B0)
        if all(-0.1 * W <= P[0] <= 1.1 * W and -0.1 * H <= P[1] <= 1.1 * H for P in (A, B)):
            pts.append([round(start + k / fps, 3), round(A[0] / W, 4), round(A[1] / H, 4), round(B[0] / W, 4), round(B[1] / H, 4)])
    return pts


def _trusted(found, ref_times):
    """The gates whose times agree with each other like in the reference run (racers differ, but not by seconds over a
    few gates): the largest such group, the start gate last (the crowded start area fools the analysis most)."""
    labs = [l for l in found if l in ref_times]
    best = []
    for m in range(1, 1 << len(labs)):
        sub = [labs[j] for j in range(len(labs)) if m >> j & 1]
        ok = all(abs((found[y] - found[x]) - (ref_times[y] - ref_times[x])) < 0.3 * abs(ref_times[y] - ref_times[x]) + 0.5
                 for x in sub for y in sub if x < y)
        if ok and (len(sub), "start" not in sub) > (len(best), "start" not in best):
            best = sub
    return best


def run_video(path, gates, seg, *, track=None, ref_times=None, discipline="", log=print, progress=None, analysis=None,
              is_ref=False):
    """gates [{label, t, a, b}] for THIS video (reference gates, or locate()'s result) → result for videos.course.
    ref_times {label: t} = the reference run's crossing times: a gate whose time doesn't fit the others is looked for
    again around the moment the others say (Hintertux 0230: a start-area bystander crossed the start line first).
    is_ref: the coach's own video, each gate's moment is where the coach paused with the racer at it — the crossing
    is looked for within REF_WIN_S of that (beta IMG_5815: a racer higher up crossed the start line's 5 m extension
    2.5 s early)."""
    marks = [dict(g) for g in gates if g]
    if not marks:
        return {"v": VERSION, "status": "nomatch"}
    marks.sort(key=lambda m: m["t"])
    R = Reader(path)
    fps = R.fps
    dur = R.duration or (marks[-1]["t"] + MAX_WIN_S)
    start = max(0.0, marks[0]["t"] - MAX_WIN_S - PAD_S)
    end = min(dur, marks[-1]["t"] + MAX_WIN_S + PAD_S)
    start = round(start * fps) / fps
    total = max(1, int((end - start) * fps))
    # pull towards the gates whose matched moment is when the racer is there (the camera follows the racer): split
    # and finish. The start gate's best match is usually while the racer still stands at the start → no pull there.
    pull_marks = [m for m in marks if m["label"] != "start"] or marks
    a = analysis or analyse(path, start, 1.0 / fps, total, seg, G.marks_track(pull_marks, track), pull=True,
                            progress=progress, log=log)
    set_windows(a, marks, start)
    if is_ref:
        for mk in marks:
            mk["win"] = [max(mk["win"][0], mk["t"] - REF_WIN_S), min(mk["win"][1], mk["t"] + REF_WIN_S)]
    res = G.passes(a, marks, start)
    by = {r["mark"]["label"]: r for r in res}
    if ref_times:
        found = {l: r["t"] for l, r in by.items() if r["t"] is not None and r["status"] in ("ok", "unsure")}
        trust = _trusted(found, ref_times)
        if trust and not (len(trust) == 1 and trust[0] == "start"):
            for mk in marks:
                lab = mk["label"]
                if lab in trust or lab not in ref_times:
                    continue
                est = float(np.median([found[l] + ref_times[lab] - ref_times[l] for l in trust]))
                span = min(abs(ref_times[lab] - ref_times[l]) for l in trust)
                w = 0.25 * span + 0.5
                m2 = {**mk, "win": [est - w, est + w], "near_t": est}
                r2 = G.passes(a, [m2], start)[0]
                if r2["t"] is None and (est + w < 0.3 or est - w > dur - 0.3):
                    r2["status"] = "outside"      # the racer passed it before the video starts / after it ends
                log(f"course: {lab} again around {est:.2f}s (±{w:.1f}): {r2['status']} {r2['t']} (first {by[lab]['t'] if lab in by else None})")
                mk["win"], mk["near_t"] = m2["win"], est
                by[lab] = r2
    out_g = []
    for g in gates:
        if g is None:
            continue
        r = by.get(g["label"])
        mk = next(m for m in marks if m["label"] == g["label"])
        e = {"label": g["label"], "status": r["status"] if r else "missed", "t": r["t"] if r else None,
             "err": r["err"] if r else None, "dist": r["dist"] if r else None, "a": mk["a"], "b": mk["b"], "at_t": mk["t"],
             "win": [round(x, 3) for x in mk["win"]], "pts": gate_path(a, mk, start)}
        if r and r.get("between"):
            e["between"] = True
        if r and r.get("at") and r.get("k") is not None:
            # which pole the racer passed nearer to: the player draws the valid 5 m segment on that side
            W, H = a["size"]
            i = max(0, min(len(a["Hs"]) - 1, int(round((mk["t"] - start) * a["fps"]))))
            M = chain(a["Hs"], i, r["k"])
            A = np.array(apply(M, (mk["a"][0] * W, mk["a"][1] * H)))
            B = np.array(apply(M, (mk["b"][0] * W, mk["b"][1] * H)))
            d = B - A
            along = float((np.array(r["at"]) - A) @ d) / max(1e-6, float(d @ d))
            e["side"] = "a" if along < 0.5 else "b"
            e["along"] = round(along, 2)
        if g.get("refined") is not None:
            e["refined"] = g["refined"]
        out_g.append(e)
    T = {e["label"]: e for e in out_g if e["t"] is not None and e["status"] in ("ok", "unsure")}

    def span(x, y):
        if x in T and y in T and T[y]["t"] > T[x]["t"]:
            return {"s": round(T[y]["t"] - T[x]["t"], 3), "err": round(float(np.hypot(T[x]["err"], T[y]["err"])), 3)}
        return None
    return {"v": VERSION, "status": "done", "fps": round(fps, 3), "gates": out_g,
            "split": span("start", "split"), "to_finish": span("split", "finish"), "total": span("start", "finish"),
            "found_pct": round(100 * a["found"] / max(1, total), 1)}
