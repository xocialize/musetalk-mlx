"""Gate D-visual STEP 2 (runs in .venv / py3.12) — assemble the lip-synced video.

Consumes DWPose coords (from extract_landmarks.py), runs the MLX generation core
(torch-free audio + VAE + UNet) and the upstream bisenet blending, muxes audio.
Optionally also renders a PyTorch-UNet reference video (same crops+blend) for A/B + SyncNet.

Usage:
    python build_video.py <coords.pkl> <audio.wav> <out.mp4> [--variant fp16] [--torch-ref]
"""
import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "refs" / "MuseTalk"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "musetalk" / "utils"))

# legacy bisenet checkpoint needs weights_only=False (torch 2.6+ default flip)
import functools  # noqa: E402
import torch  # noqa: E402
torch.load = functools.partial(torch.load, weights_only=False)

ap = argparse.ArgumentParser()
ap.add_argument("coords"); ap.add_argument("audio"); ap.add_argument("out")
ap.add_argument("--variant", default="fp16")
ap.add_argument("--extra-margin", type=int, default=10)
ap.add_argument("--parsing-mode", default="jaw")
ap.add_argument("--torch-ref", action="store_true")
args = ap.parse_args()

mx.set_default_device(mx.gpu)
from musetalk_mlx.pipeline_mlx import MuseTalkPipeline  # noqa: E402
from musetalk.utils.blending import get_image           # noqa: E402  (bisenet face-parse blend)
from musetalk.utils.face_parsing import FaceParsing      # noqa: E402

with open(args.coords, "rb") as f:
    meta = pickle.load(f)
coords, fps, frame_paths = meta["coords"], meta["fps"], meta["frames"]
frames = [cv2.imread(p) for p in frame_paths]
print(f"frames={len(frames)} fps={fps:.2f}", flush=True)

pipe = MuseTalkPipeline.from_pretrained_mlx(ROOT / "dist" / f"MuseTalk-1.5-MLX-{args.variant}")
# FaceParsing loads from hardcoded ./models/... paths -> run from the upstream dir (symlinked)
os.chdir(UPSTREAM)
fp = FaceParsing()
os.chdir(ROOT)

# audio -> per-frame whisper chunks (torch-free MLX path)
chunks = pipe.encode_audio_from_wav(args.audio, fps=int(round(fps)))
video_num = chunks.shape[0]
print(f"audio chunks={video_num}", flush=True)

# precompute per-frame crops + 8-ch latents (v15 extra_margin)
PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)
crops, boxes, latents = [], [], []
for bbox, frame in zip(coords, frames):
    if bbox == PLACEHOLDER:
        crops.append(None); boxes.append(None); latents.append(None); continue
    x1, y1, x2, y2 = bbox
    y2 = min(y2 + args.extra_margin, frame.shape[0])
    crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
    crops.append(crop); boxes.append((x1, y1, x2, y2))
    latents.append(pipe.get_latents_for_unet(crop))

# cycle frames/coords/latents to cover the audio length
def cyc(lst, i):
    c = lst + lst[::-1]
    return c[i % len(c)]

# batched MLX generation
lat_stack = mx.concatenate([cyc(latents, i) for i in range(video_num)], axis=0)
recon = pipe.run_batched(lat_stack.astype(mx.float16), chunks.astype(mx.float16), batch_size=8)
print(f"generated {len(recon)} faces", flush=True)

# blend back + write frames
tmp = ROOT / "outputs" / "dvisual_frames"; tmp.mkdir(parents=True, exist_ok=True)
for p in tmp.glob("*.png"):
    p.unlink()
for i in range(video_num):
    box = cyc(boxes, i); ori = cyc(frames, i).copy()
    if box is None:
        cv2.imwrite(str(tmp / f"{i:08d}.png"), ori); continue
    x1, y1, x2, y2 = box
    res = cv2.resize(recon[i].astype(np.uint8), (x2 - x1, y2 - y1))
    combined = get_image(ori, res, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
    cv2.imwrite(str(tmp / f"{i:08d}.png"), combined)

# encode video + mux audio
silent = ROOT / "outputs" / "dvisual_silent.mp4"
subprocess.run(["ffmpeg", "-y", "-r", str(fps), "-i", str(tmp / "%08d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["ffmpeg", "-y", "-i", str(silent), "-i", args.audio, "-c:v", "copy",
                "-c:a", "aac", "-shortest", args.out], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"WROTE {args.out}", flush=True)
