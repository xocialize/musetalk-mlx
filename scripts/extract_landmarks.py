"""Gate D-visual STEP 1 (runs in .venv / py3.12) — faithful DWPose+S3FD crop extraction.

Uses the EXACT upstream DWPose weights via onnxruntime (rtmlib, dw-ll_ucoco_384.onnx +
yolox_l.onnx) for landmarks + the bundled S3FD face detector (torch) for the bbox — no
mmcv/mmpose. Replicates musetalk/utils/preprocessing.get_landmark_and_bbox exactly.

Usage:  python scripts/extract_landmarks.py <video> <frames_dir> <coords.pkl>
"""
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "refs" / "MuseTalk" / "musetalk" / "utils"))  # face_detection

from face_detection import FaceAlignment, LandmarksType   # noqa: E402  (S3FD, torch)
from rtmlib import Wholebody                                # noqa: E402  (DWPose onnx)

video, frames_dir, coords_pkl = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(frames_dir, exist_ok=True)
COORD_PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)

# extract frames
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
frames, paths = [], []
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    p = os.path.join(frames_dir, f"{i:08d}.png")
    cv2.imwrite(p, fr)
    frames.append(fr); paths.append(p); i += 1
cap.release()
print(f"extracted {len(frames)} frames @ {fps:.2f} fps", flush=True)

# faithful DWPose (exact upstream onnx weights) + S3FD
dw = ROOT / "weights" / "dwpose"
pose = Wholebody(det=str(dw / "yolox_l.onnx"), pose=str(dw / "dw-ll_ucoco_384.onnx"),
                 pose_input_size=(288, 384), backend="onnxruntime", device="cpu")
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device="cpu")

coords = []
for fr in frames:
    kpts, _ = pose(fr)                                  # (N,133,2) COCO-wholebody
    bbox = fa.get_detections_for_batch(np.asarray([fr]))[0]   # S3FD bbox or None
    if bbox is None or len(kpts) == 0:
        coords.append(COORD_PLACEHOLDER); continue
    flm = kpts[0][23:91].astype(np.int32)              # 68 face landmarks (== upstream)
    half = flm[29].copy()
    half_dist = np.max(flm[:, 1]) - half[1]
    upper = max(0, half[1] - half_dist)
    fl = (np.min(flm[:, 0]), int(upper), np.max(flm[:, 0]), np.max(flm[:, 1]))
    x1, y1, x2, y2 = fl
    coords.append(tuple(bbox) if (y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0) else fl)

with open(coords_pkl, "wb") as f:
    pickle.dump({"coords": coords, "fps": fps, "frames": paths, "n": len(paths)}, f)
n_ok = sum(1 for c in coords if c != COORD_PLACEHOLDER)
print(f"saved {len(coords)} coords ({n_ok} with face) -> {coords_pkl}", flush=True)
