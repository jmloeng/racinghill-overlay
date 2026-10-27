# Racinghill overlay engine

The server that makes **OVERLAY** ("ghost run") and **race line** videos for [Racinghill.com](https://racinghill.com), a web app where alpine ski racing teams upload, store and share training and race videos.

- **Overlay:** two runs filmed from the same spot. The second skier is cut out and laid over the first as a see-through "ghost".
- **Race lines:** both skiers' paths on the snow are drawn while the camera pans and zooms (run 1 red, ghost blue).

It uses [Ultralytics YOLO11s-seg](https://github.com/ultralytics/ultralytics) (AGPL-3.0), exported to ONNX, for person segmentation. That is why this part of Racinghill is published here under the **GNU Affero General Public License v3.0** (see `LICENSE`). The rest of Racinghill is not part of this repository.

## Files

| File | What it does |
|---|---|
| `overlay.py` | The overlay renderer: aligns run 2 to run 1 (SIFT + optical flow), cuts out the ghost, checks that both runs were filmed from the same spot, writes the MP4 with ffmpeg |
| `raceline.py` | Race line: finds the skier's feet in every frame, follows the camera motion (keyframe SIFT + flow) and draws the line glued to the slope |
| `seg.py` | Person segmentation with the ONNX model (640 px, plus an optional 320 px model for fast crops) |
| `server.py` | Small HTTP worker (Google Cloud Run): takes a job, downloads the videos, renders, uploads the result as the logged-in user |
| `Dockerfile` | Builds the container; stage 1 exports the YOLO11s-seg weights to ONNX at 640 and 320 px |

## Run locally

```bash
pip install -r requirements.txt
yolo export model=yolo11s-seg.pt format=onnx imgsz=640   # needs ultralytics; produces yolo11s-seg.onnx
OVERLAY_MODEL=yolo11s-seg.onnx python overlay.py run1.mp4 run2.mp4 out.mp4 --offset 1.36 --start 2 --lines
```

`--offset` = time in run 2 minus time in run 1 at the same moment (for example both skiers at the same gate). ffmpeg must be on PATH.

## Contact

Inflection Code Project, operated by Inflection Advisory AS (Norway). Email: jorgen@inflection.no
