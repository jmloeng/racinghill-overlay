"""Person segmentation with a YOLO11-seg model exported to ONNX (no PyTorch in the container).

Output of the exported model (640x640 input):
  output0: (1, 4 + 80 + 32, 8400)  box cx,cy,w,h · 80 class scores · 32 mask coefficients
  output1: (1, 32, 160, 160)       mask prototypes
Only class 0 (person) is used.
An optional second export at 320x320 (same weights, `<model>-320.onnx` next to the 640 one) is about 4x faster; race-line
analysis uses it for the crop around the racer. persons() picks the model by the size of the image it gets.
"""
import os
import cv2
import numpy as np
import onnxruntime as ort

SIZE = 640


class Segmenter:
    def __init__(self, model_path, threads=None):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = {}                    # input size -> session
        small = os.path.splitext(model_path)[0] + "-320.onnx"
        for path in (model_path, small):
            if os.path.exists(path):
                sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
                self.sess[sess.get_inputs()[0].shape[2]] = sess
        if not self.sess:
            raise FileNotFoundError(model_path)
        self.small = 320 if 320 in self.sess else SIZE   # crop size for the fast model (SIZE when there is none)

    def persons(self, img, conf=0.15, iou=0.5):
        """img: BGR, SIZE x SIZE (or self.small x self.small). Returns list of (box[x0,y0,x1,y1], score, mask uint8 0/255,
        same size as img)."""
        SIZE = img.shape[0]
        sess = self.sess[SIZE]
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        out0, out1 = sess.run(None, {sess.get_inputs()[0].name: x})
        p = out0[0]                       # (116, 8400 at 640)
        scores = p[4]                     # class 0 = person
        keep = scores > conf
        if not keep.any():
            return []
        cand = p[:, keep]
        cx, cy, w, h = cand[0], cand[1], cand[2], cand[3]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        sc = cand[4]
        coefs = cand[84:].T               # (n, 32)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), conf, iou)
        if len(idx) == 0:
            return []
        idx = np.array(idx).reshape(-1)
        pm = out1.shape[2]                # 160 at 640
        protos = out1[0].reshape(32, -1)  # (32, pm*pm)
        res = []
        for k in idx:
            m = 1 / (1 + np.exp(-(coefs[k] @ protos)))
            m = cv2.resize(m.reshape(pm, pm), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            x0, y0, bw, bh = boxes[k]
            x1, y1 = x0 + bw, y0 + bh
            mm = np.zeros((SIZE, SIZE), np.uint8)
            xa, ya, xb, yb = max(0, int(x0)), max(0, int(y0)), min(SIZE, int(np.ceil(x1))), min(SIZE, int(np.ceil(y1)))
            if xb <= xa or yb <= ya:
                continue
            mm[ya:yb, xa:xb] = (m[ya:yb, xa:xb] > 0.5).astype(np.uint8) * 255
            res.append((np.array([x0, y0, x1, y1], np.float32), float(sc[k]), mm))
        return res


def crop_at(frame, cx, cy, size=SIZE):
    """SIZE x SIZE crop around (cx, cy), shifted to stay inside the frame (padded when the frame is smaller)."""
    H, W = frame.shape[:2]
    x0 = int(np.clip(cx - size / 2, 0, max(0, W - size)))
    y0 = int(np.clip(cy - size / 2, 0, max(0, H - size)))
    crop = frame[y0:y0 + size, x0:x0 + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        pad = np.zeros((size, size, 3), np.uint8)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    return crop, x0, y0


def letterbox(frame, size=SIZE):
    """Whole frame scaled into SIZE x SIZE (for finding the racer when there is no AI ZOOM track)."""
    H, W = frame.shape[:2]
    s = size / max(H, W)
    small = cv2.resize(frame, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    pad = np.full((size, size, 3), 114, np.uint8)
    pad[:small.shape[0], :small.shape[1]] = small
    return pad, s
