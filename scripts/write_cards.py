"""Write HF model cards (README.md) for each exported MLX variant."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

QUALITY = {  # recon mean|Δ| vs fp32 torch golden (measured at export-verify)
    "fp16": "0.32", "q8": "0.41", "q4": "2.74",
}


def card(name, cfg):
    q = cfg.get("quantization")
    qline = (f"UNet Linears quantized to **int{q['bits']}** (group_size {q['group_size']}); "
             f"VAE + Whisper encoder kept fp16. Per-pass cosine vs fp16 = "
             f"**{cfg['unet_cosine_vs_fp16']}**." if q else
             "Full **fp16** (VAE + UNet + Whisper encoder).")
    return f"""---
license: mit
library_name: mlx
tags:
- mlx
- lip-sync
- talking-head
- musetalk
- apple-silicon
base_model: TMElyralab/MuseTalk
pipeline_tag: image-to-image
---

# MuseTalk 1.5 — MLX ({name})

Apple-MLX port of **[MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk)** (TMElyralab / Tencent
Music) — realtime, high-quality lip-sync via **single-step latent-space inpainting** (not diffusion).
Runs natively on Apple Silicon. MIT-licensed, commercial use OK.

**This variant:** {qline}
Decoded-face error vs the PyTorch reference: mean |Δ| ≈ **{QUALITY[name]}/255**.

## Components (all in this repo, self-contained, torch-free)

| File | What |
|------|------|
| `unet.safetensors` | SD1.x `UNet2DConditionModel` (in=8, out=4, cross_attn=384), single-step t=0 |
| `vae.safetensors` | `sd-vae-ft-mse` AutoencoderKL (fp16) |
| `whisper_encoder.safetensors` | whisper-tiny audio encoder (fp16) |
| `config.json` | dtype / quantization / scaling factor |

## Performance

Realtime on an M-series GPU: **~34 generated 256² faces/sec** at batch 8 (>25 fps video rate),
~7 GB peak. fp16 inference.

## Usage

```python
from musetalk_mlx.pipeline_mlx import MuseTalkPipeline
pipe = MuseTalkPipeline.from_pretrained_mlx("MuseTalk-1.5-MLX-{name}")
# crop_bgr: a 256x256 face crop; chunks: (N,50,384) whisper audio features
latents = pipe.get_latents_for_unet(crop_bgr)
faces = pipe.generate_faces(latents, audio_chunks)   # BGR uint8 lip-synced faces
```

Face detection / cropping / paste-back blending use the upstream
([MuseTalk](https://github.com/TMElyralab/MuseTalk)) CPU preprocessing.

## Parity (vs PyTorch, cpu fp32)

VAE encode 1.7e-5 · decode 3.4e-5 · UNet forward **1.4e-6** · whisper encoder 1.6e-5 ·
face-level e2e recon ≤ 2/255.

## License

MIT (mirrors upstream MuseTalk). Dependency models keep their own permissive licenses.
Port by MVS Collective (xocialize-code).
"""


for name in ("fp16", "q8", "q4"):
    d = DIST / f"MuseTalk-1.5-MLX-{name}"
    cfg = json.loads((d / "config.json").read_text())
    (d / "README.md").write_text(card(name, cfg))
    print("wrote", (d / "README.md").relative_to(ROOT))
