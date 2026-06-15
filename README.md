# musetalk-mlx

Apple-MLX port of **[MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk)** (TMElyralab / Tencent Music) —
realtime, high-quality lip-sync via **single-step latent-space inpainting** (not diffusion).
MIT-licensed, commercial-OK.

> **Status:** COMPLETE + PUBLISHED — `mlx-community/MuseTalk-1.5-{fp16,q8,q4}`.
> All neural components ported + parity-locked (UNet fwd 1.41e-6, face e2e recon ≤2/255), realtime
> ~34 faces/s @ bs=8. See [`_research/PREFLIGHT.md`](_research/PREFLIGHT.md) for the full plan,
> CONFIRM-gate results, per-phase parity, and lessons M1–M10. *(Deferred: real-clip SyncNet visual
> gate, torch-free mel, MLX-Swift mirror.)*

## What it does

Given a face video and a driving audio track, MuseTalk regenerates the lower-face / mouth region of
each frame to match the speech — in a **single UNet forward per frame** (no denoising loop), so it's
fast enough for realtime on Apple Silicon.

```
audio ─► Whisper-tiny encoder (384-d) ─► PositionalEncoding ─┐ (cross-attn)
face  ─► crop 256² ─► VAE.encode(masked) ⊕ VAE.encode(ref) ─► 8-ch latent ─► UNet(t=0) ─► VAE.decode ─► paste back
```

## Components

| Component | Source | Port |
|-----------|--------|------|
| UNet (8→4ch, cross_attn=384) | `TMElyralab/MuseTalk` · `musetalkV15/unet.pth` | manual Tier-2 (SD1.x topology) |
| VAE | `stabilityai/sd-vae-ft-mse` | manual Tier-2 (not in DiffusionKit) |
| Audio encoder | `openai/whisper-tiny` | in-package MLX whisper encoder |
| Face detect / DWPose / parsing | S3FD · DWPose · bisenet | CPU, run as-is (caller-supplied) |

## Quick start

Install, then load a published MLX variant — torch-free and self-contained:

```bash
pip install -e .
```

```python
from musetalk_mlx.pipeline_mlx import MuseTalkPipeline

# A downloaded mlx-community/MuseTalk-1.5-{fp16,q8,q4} snapshot
# (config.json + unet.safetensors + vae.safetensors + whisper_encoder.safetensors).
pipe = MuseTalkPipeline.from_pretrained_mlx("MuseTalk-1.5-fp16")

# Per-frame face generation (neural core); face crop/blend/paste-back are
# upstream CPU preprocessing wired in by the caller.
faces = pipe.run_batched(latent_stack, chunk_stack, batch_size=8)
```

`MuseTalkPipeline` also exposes `from_pretrained(weights_root)` (loads original PyTorch/HF weights;
needs torch), `get_latents_for_unet` (256² BGR → 8-ch latent), `decode_latents`, `generate_faces`,
and `run_batched`. Helper scripts live in `scripts/` (`download_weights.py`, `export_mlx.py`,
`build_video.py`, `bench_realtime.py`, `publish.py`, `write_cards.py`, golden-capture scripts).

## Dev / parity

```bash
pip install -e ".[parity]"   # pulls torch + diffusers>=0.27 + transformers for golden capture
pytest tests/parity          # PT vs MLX, run on mx.cpu, gated on RELATIVE error
pytest tests/smoke           # shape / config / realtime / published-variant smoke
```

PyTorch is an **optional** dev dependency — end users running the MLX port never need it.

> Note: upstream `musetalkV15/musetalk.json` pins `diffusers==0.6.0.dev0`, which won't build on
> py3.12. The parity extra therefore uses `diffusers>=0.27` (the UNet/VAE block decomposition is
> numerically stable across these versions); port the 0.6.0-era block order, not a newer one.

## License

MIT (mirrors upstream). Dependency models keep their own permissive licenses.
