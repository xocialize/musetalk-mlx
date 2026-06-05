"""Gate D-visual rigor — MLX vs PyTorch on REAL crops (full VAE+UNet generation).

For a sample of real DWPose crops + real audio chunks, run both the MLX published model
and the upstream torch UNet/VAE (posterior mean, deterministic) and compare decoded faces.
Proves the port matches on real data, not just synthetic goldens.
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel

ROOT = Path(__file__).resolve().parents[1]
mx.set_default_device(mx.cpu)
from musetalk_mlx.pipeline_mlx import MuseTalkPipeline, preprocess_img  # noqa: E402
from musetalk_mlx.whisper.audio2feature import apply_pe  # noqa: E402

SCALE = 0.18215
N_SAMPLE = 16

# ---- load real crops + audio chunks ----
meta = pickle.load(open(ROOT / "outputs" / "yongen_coords.pkl", "rb"))
frames = [cv2.imread(p) for p in meta["frames"]]
mlx_pipe = MuseTalkPipeline.from_pretrained_mlx(ROOT / "dist" / "MuseTalk-1.5-MLX-fp16")
chunks = np.array(mlx_pipe.encode_audio_from_wav(
    str(ROOT / "refs/MuseTalk/data/audio/yongen.wav"), fps=25))

idxs = np.linspace(0, min(len(frames), chunks.shape[0]) - 1, N_SAMPLE).astype(int)
crops = []
for i in idxs:
    x1, y1, x2, y2 = meta["coords"][i]
    y2 = min(int(y2) + 10, frames[i].shape[0])
    crops.append(cv2.resize(frames[i][int(y1):y2, int(x1):int(x2)], (256, 256),
                            interpolation=cv2.INTER_LANCZOS4))

# ---- torch reference (mean latents + UNet t=0 + decode) ----
vae = AutoencoderKL.from_pretrained(str(ROOT / "weights/sd-vae-ft-mse")).eval()
cfg = json.loads((ROOT / "weights/MuseTalk/musetalkV15/musetalk.json").read_text())
cfg.pop("_class_name", None); cfg.pop("_diffusers_version", None)
unet = UNet2DConditionModel(**cfg).eval()
unet.load_state_dict(torch.load(ROOT / "weights/MuseTalk/musetalkV15/unet.pth",
                                map_location="cpu", weights_only=True))
norm = T.Normalize([0.5] * 3, [0.5] * 3)


def t_prep(img, hm):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
    if hm:
        m = torch.zeros(256, 256); m[:128] = 1
        x = x * (m > 0.5)
    return norm(x)[None]


import math  # noqa: E402
d = 384
pe = torch.zeros(50, d)
pos = torch.arange(0, 50).unsqueeze(1).float()
div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)

diffs = []
for k, i in enumerate(idxs):
    crop = crops[k]
    chunk = chunks[i:i + 1].astype(np.float32)
    # MLX
    lat = mlx_pipe.get_latents_for_unet(crop)
    face_mlx = mlx_pipe.generate_faces(lat, mx.array(chunk))[0]      # BGR uint8
    # torch
    with torch.no_grad():
        ml = SCALE * vae.encode(t_prep(crop, True)).latent_dist.mean
        rl = SCALE * vae.encode(t_prep(crop, False)).latent_dist.mean
        latents = torch.cat([ml, rl], dim=1)
        af = torch.from_numpy(chunk) + pe[None]
        pred = unet(latents, torch.tensor([0]), encoder_hidden_states=af).sample
        dec = vae.decode(pred / SCALE).sample
        dec = (dec / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        face_torch = (dec * 255).round().astype(np.uint8)[0][..., ::-1]
    diff = np.abs(face_mlx.astype(np.int16) - face_torch.astype(np.int16))
    diffs.append((int(diff.max()), float(diff.mean())))

maxes = [m for m, _ in diffs]; means = [a for _, a in diffs]
print(f"\nMLX vs torch on {N_SAMPLE} REAL crops (decoded faces, uint8 0-255):")
print(f"  per-frame max|Δ|:  worst={max(maxes)}  median={int(np.median(maxes))}")
print(f"  per-frame mean|Δ|: worst={max(means):.3f}  median={np.median(means):.3f}")
print(f"  -> port matches reference on real data" if max(means) < 1.5 else "  -> CHECK")
