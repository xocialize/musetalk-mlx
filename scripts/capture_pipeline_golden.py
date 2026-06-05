"""Capture the face-level e2e golden (Gate D-core) from the upstream torch path.

crop 256² (deterministic) -> get_latents_for_unet (posterior MEAN) -> UNet(t=0, pe(audio))
-> decode_latents -> recon BGR. Validates the full neural assembly (mask, normalize, BGR/RGB,
concat, decode) against MLX. Uses the posterior mean so it's deterministic across frameworks.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel

ROOT = Path(__file__).resolve().parents[1]
VAE_DIR = ROOT / "weights" / "sd-vae-ft-mse"
UNET_DIR = ROOT / "weights" / "MuseTalk" / "musetalkV15"
OUT = ROOT / "goldens"

SCALE = 0.18215
norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
vae = AutoencoderKL.from_pretrained(str(VAE_DIR)).eval()
cfg = json.loads((UNET_DIR / "musetalk.json").read_text())
cfg.pop("_class_name", None); cfg.pop("_diffusers_version", None)
unet = UNet2DConditionModel(**cfg).eval()
unet.load_state_dict(torch.load(UNET_DIR / "unet.pth", map_location="cpu", weights_only=True))


def mask_tensor(size=256):
    m = torch.zeros((size, size)); m[: size // 2, :] = 1.0
    return m


def preprocess(img_bgr, half_mask):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
    if half_mask:
        x = x * (mask_tensor() > 0.5)
    return norm(x)[None]


# deterministic 256² "face" (identical bytes injected into MLX)
img = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
audio_chunk = np.load(OUT / "audio_golden.npz")["chunks"][:1].astype(np.float32)   # (1,50,384)

# sinusoidal PE (musetalk/models/unet.py PositionalEncoding)
import math
d = 384
pe = torch.zeros(50, d)
pos = torch.arange(0, 50).unsqueeze(1).float()
div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)

with torch.no_grad():
    ml = SCALE * vae.encode(preprocess(img, True)).latent_dist.mean
    rl = SCALE * vae.encode(preprocess(img, False)).latent_dist.mean
    latents = torch.cat([ml, rl], dim=1)                       # (1,8,32,32)
    audio_feat = torch.from_numpy(audio_chunk) + pe[None]
    pred = unet(latents, torch.tensor([0]), encoder_hidden_states=audio_feat).sample
    dec = vae.decode(pred / SCALE).sample
    dec = (dec / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    recon = (dec * 255).round().astype(np.uint8)[..., ::-1]    # BGR

np.savez(OUT / "pipeline_golden.npz",
         img=img, latents=latents.numpy(), pred=pred.numpy(), recon=np.ascontiguousarray(recon))
print("latents", latents.shape, "pred", pred.shape, "recon", recon.shape)
print("saved goldens/pipeline_golden.npz")
