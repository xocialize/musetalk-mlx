"""Capture the PyTorch VAE oracle (sd-vae-ft-mse) for Gate A parity.

Deterministic: we compare the DiagonalGaussian *moments* (mean, logvar) on encode,
not a sampled latent (MLX RNG != torch RNG). Decode is fully deterministic.

Outputs:
  goldens/vae_golden.npz        — input img, enc mean/logvar, dec latent, dec out
  goldens/vae_key_inventory.txt — full state_dict key list (to mirror MLX module names)
"""
from pathlib import Path
import json
import numpy as np
import torch
from diffusers import AutoencoderKL

ROOT = Path(__file__).resolve().parents[1]
VAE_DIR = ROOT / "weights" / "sd-vae-ft-mse"
OUT = ROOT / "goldens"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
vae = AutoencoderKL.from_pretrained(str(VAE_DIR)).eval()

# dump key inventory + config
keys = list(vae.state_dict().keys())
(OUT / "vae_key_inventory.txt").write_text("\n".join(keys))
print(f"state_dict: {len(keys)} tensors")
print("scaling_factor:", vae.config.scaling_factor)

# fixed inputs (numpy -> inject into both frameworks)
rng = np.random.default_rng(1234)
img = rng.standard_normal((1, 3, 256, 256)).astype(np.float32)        # normalized image space [-ish]
dec_latent = rng.standard_normal((1, 4, 32, 32)).astype(np.float32)   # latent space

with torch.no_grad():
    posterior = vae.encode(torch.from_numpy(img)).latent_dist
    enc_mean = posterior.mean.cpu().numpy()
    enc_logvar = posterior.logvar.cpu().numpy()
    dec_out = vae.decode(torch.from_numpy(dec_latent)).sample.cpu().numpy()

np.savez(
    OUT / "vae_golden.npz",
    img=img, enc_mean=enc_mean, enc_logvar=enc_logvar,
    dec_latent=dec_latent, dec_out=dec_out,
    scaling_factor=np.array(vae.config.scaling_factor, dtype=np.float32),
)
print("enc_mean", enc_mean.shape, "| enc_logvar", enc_logvar.shape, "| dec_out", dec_out.shape)
print("saved goldens/vae_golden.npz")
