"""BIB (start number) reading, beta v281 — picture side. The reading itself is done by Claude (api/bib.js, Vercel).

Tested on 82 race clips from production (HL Grong + NM Kvitfjell, scripts/bib-test, 2026-11-08): with the session's
start list as a check, three tries found the right bib in 60 of 75 clips where it is visible (80 %), 1 wrong.

One pass over the clip at ~8 frames/s finds the racer (biggest person who isn't a spectator right next to the camera,
YOLO11s-seg) and keeps the person box at full resolution. From those, three "sheets" of up to 8 crops (4 x 2, each
450 px high) — one per try, asked in this order until an answer is on the start list:
  1. best    the 8 biggest + sharpest crops in the whole clip (spread out in time)
  2. middle  the best crop in each of 8 equal slots from 25 % to 75 % of the clip (owner's idea: the racer passes by)
  3. late    the 8 best crops in the second half (owner's idea: the racer is seen from behind — back numbers read best)

  python bib.py clip.mp4 outdir      (local test: writes try1.jpg … try3.jpg)
"""
import os
import sys

import cv2
import numpy as np

from seg import letterbox

FPS = 8.0
K = 8
TILE_H, TILE_W = 450, 340
MAX_CANDS = 400          # crops kept in memory (a 60 s clip at 8 fps = 480 frames)


def _sharp(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (128, max(1, int(128 * g.shape[0] / max(1, g.shape[1])))))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def racer_crops(path, seg, fps=FPS, log=print):
    """([(score, height px, time fraction 0..1, crop)] — one per sampled frame where a racer-sized person was found,
    number of frames sampled)."""
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(vfps / fps)))
    out, i = [], 0
    while cap.grab():
        if i % step:
            i += 1
            continue
        ok, f = cap.retrieve()
        i += 1
        if not ok:
            continue
        lb, s = letterbox(f)
        ps = seg.persons(lb, conf=0.4)
        if not ps:
            continue
        H, W = f.shape[:2]
        boxes = [p[0] / s for p in ps]
        # spectators / coaches right next to the camera are not the racer
        boxes = [b for b in boxes if not ((b[3] - b[1]) > 0.5 * H or (b[3] > 0.97 * H and (b[3] - b[1]) > 0.3 * H))]
        if not boxes:
            continue
        b = max(boxes, key=lambda b: b[3] - b[1])
        h, w = b[3] - b[1], b[2] - b[0]
        if h < 60:
            continue
        x0, y0 = int(max(0, b[0] - 0.08 * w)), int(max(0, b[1] - 0.04 * h))
        x1, y1 = int(min(W, b[2] + 0.08 * w)), int(min(H, b[3] + 0.04 * h))
        crop = f[y0:y1, x0:x1].copy()
        out.append((h * min(1.0, _sharp(crop) / 150.0), int(h), i - 1, crop))
    cap.release()
    n = max(total, i, 1)
    out = [(sc, h, fi / n, c) for sc, h, fi, c in out]
    samples = max(1, i // step)
    if len(out) > MAX_CANDS:          # keep the best ones, in time order
        keep = sorted(range(len(out)), key=lambda k: -out[k][0])[:MAX_CANDS]
        out = [out[k] for k in sorted(keep)]
    return out, samples


def _best(cands, gap, k=K):
    """the k best crops, no two within `gap` (fraction of the clip) of each other"""
    picks = []
    for c in sorted(cands, key=lambda c: -c[0]):
        if all(abs(c[2] - p[2]) > gap for p in picks):
            picks.append(c)
        if len(picks) >= k:
            break
    return picks


def _slots(cands, start, end, k=K):
    best = {}
    for c in cands:
        if start <= c[2] < end:
            sl = min(k - 1, int((c[2] - start) * k / (end - start)))
            if sl not in best or c[0] > best[sl][0]:
                best[sl] = c
    return [best[s] for s in sorted(best)]


def tries(cands, samples):
    """The three crop sets, in the order they are asked. Empty sets and repeats of an earlier set are left out."""
    gap = 4.0 / samples           # no two picks within 4 sampled frames (as in the test)
    sets = [("best", _best(cands, gap)), ("middle", _slots(cands, 0.25, 0.75)), ("late", _best([c for c in cands if c[2] >= 0.5], gap))]
    out, seen = [], set()
    for name, picks in sets:
        key = tuple(sorted(id(p[3]) for p in picks))
        if picks and key not in seen:
            seen.add(key)
            out.append((name, picks))
    return out


def sheet(picks):
    """Up to 8 crops as one JPEG (4 per row). Returns bytes."""
    tiles = []
    for _, _, _, im in picks[:K]:
        s = TILE_H / im.shape[0]
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        if im.shape[1] > TILE_W:
            x = (im.shape[1] - TILE_W) // 2
            im = im[:, x:x + TILE_W]
        tiles.append(cv2.copyMakeBorder(im, 4, 4, 4, 4 + TILE_W - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    img = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def on_list(answer, start_list):
    """answer "144" / "V3" / None → the start-list rows it matches ([] = not on the list). No list → [] as well."""
    if not answer or not str(answer).isdigit():
        return []
    return [r for r in start_list if r.get("bib") == int(answer)]


if __name__ == "__main__":
    from seg import Segmenter
    here = os.path.dirname(os.path.abspath(__file__))
    seg = Segmenter(os.environ.get("OVERLAY_MODEL") or os.path.join(here, "model", "yolo11s-seg.onnx"))
    cands, samples = racer_crops(sys.argv[1], seg)
    os.makedirs(sys.argv[2], exist_ok=True)
    for n, (name, picks) in enumerate(tries(cands, samples), 1):
        open(os.path.join(sys.argv[2], f"try{n}_{name}.jpg"), "wb").write(sheet(picks))
        print(n, name, len(picks), [round(p[2], 2) for p in picks])
