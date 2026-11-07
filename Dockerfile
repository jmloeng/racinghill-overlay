# Racinghill OVERLAY worker (ghost run) — Python 3.13, OpenCV, ONNX Runtime, ffmpeg. Google Cloud Run.
# Stage 1 exports Ultralytics' official YOLO11s-seg weights to ONNX (PyTorch is only needed for that and stays out of
# the final image). Note: Ultralytics YOLO11 is AGPL-3.0 (see docs/ROADMAP.md "OVERLAY" — licence question).
FROM python:3.13-slim AS model
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ultralytics==8.4.188 onnx onnxslim onnxruntime==1.31.0
WORKDIR /m
ENV YOLO_CONFIG_DIR=/tmp/yolo
RUN yolo export model=yolo11s-seg.pt format=onnx imgsz=320 simplify=True && mv yolo11s-seg.onnx yolo11s-seg-320.onnx \
 && yolo export model=yolo11s-seg.pt format=onnx imgsz=640 simplify=True && ls -la /m

FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=model /m/yolo11s-seg.onnx /m/yolo11s-seg-320.onnx /app/model/
COPY overlay.py seg.py raceline.py gates.py course.py server.py ./
RUN useradd -m worker && chown -R worker /app
USER worker
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "server.py"]
