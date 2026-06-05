"""Phase 6 — export self-contained MLX variants (bf16 / q8 / q4) for mlx-community.

Each variant dir is torch-free and standalone:
  {vae,whisper_encoder}.safetensors  (bf16, hi-precision — never quantized)
  unet.safetensors                   (bf16 or int8/int4 quantized Linears)
  config.json + README.md (model card)

Quantization is UNet-Linear-only; VAE + whisper encoder stay bf16.
"""
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
from musetalk_mlx.config import VAE_SCALING_FACTOR  # noqa: E402
from musetalk_mlx.models.unet import UNet2DConditionModel  # noqa: E402
from musetalk_mlx.models.vae import AutoencoderKL  # noqa: E402
from musetalk_mlx.pipeline_mlx import _tree_cast  # noqa: E402
from musetalk_mlx.whisper.audio2feature import apply_pe  # noqa: E402
from musetalk_mlx.whisper.whisper_encoder import WhisperEncoder  # noqa: E402
from musetalk_mlx.utils.weights import (  # noqa: E402
    load_unet_weights, load_vae_weights, load_whisper_encoder_weights, save_native,
)

mx.set_default_device(mx.gpu)
W = ROOT / "weights"


def _fp16(m):
    m.update(_tree_cast(m.parameters(), mx.float16)); mx.eval(m.parameters()); return m


# load fp32, cast vae+whisper to bf16 (shared across variants)
vae = _fp16(load_vae_weights(AutoencoderKL(), W / "sd-vae-ft-mse").eval())
enc = _fp16(load_whisper_encoder_weights(WhisperEncoder(), W / "whisper-tiny").eval())

# fp16 UNet reference output for cosine
g = np.load(ROOT / "goldens" / "unet_golden.npz")
lat = mx.array(g["latent"]).astype(mx.float16)
aud = apply_pe(mx.array(g["audio"]).astype(mx.float16))


def unet_out(m):
    o = m(lat, mx.array([0]), aud); mx.eval(o)
    return np.array(o.astype(mx.float32)).ravel().astype(np.float64)


ref = unet_out(_fp16(load_unet_weights(UNet2DConditionModel(), W / "MuseTalk" / "musetalkV15" / "unet.pth").eval()))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


VARIANTS = [
    ("fp16", None),
    ("q8", {"group_size": 64, "bits": 8}),
    ("q4", {"group_size": 64, "bits": 4}),
]

for name, q in VARIANTS:
    d = DIST / f"MuseTalk-1.5-MLX-{name}"
    d.mkdir(parents=True, exist_ok=True)
    unet = _fp16(load_unet_weights(UNet2DConditionModel(), W / "MuseTalk" / "musetalkV15" / "unet.pth").eval())
    cosine = 1.0
    if q:
        nn.quantize(unet, group_size=q["group_size"], bits=q["bits"])
        cosine = cos(ref, unet_out(unet))
    save_native(vae, d / "vae.safetensors")
    save_native(enc, d / "whisper_encoder.safetensors")
    save_native(unet, d / "unet.safetensors")
    cfg = {
        "model": "MuseTalk-1.5", "framework": "mlx", "dtype": "float16",
        "scaling_factor": VAE_SCALING_FACTOR, "cross_attention_dim": 384,
        "quantization": q, "unet_cosine_vs_fp16": round(cosine, 5),
    }
    (d / "config.json").write_text(json.dumps(cfg, indent=2))
    sz = sum(f.stat().st_size for f in d.glob("*.safetensors")) / 1e9
    print(f"{name:>4}: unet_cos={cosine:.5f}  size={sz:.2f} GB  -> {d.name}")

print("\nexported variants to", DIST)
