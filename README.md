# musetalk-mlx

Apple-MLX port of **[MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk)** (TMElyralab / Tencent Music) —
realtime, high-quality lip-sync via **single-step latent-space inpainting** (not diffusion).
MIT-licensed, commercial-OK.

> **Status:** ✅ **COMPLETE + PUBLISHED** — `mlx-community/MuseTalk-1.5-{fp16,q8,q4}`.
> All neural components ported + parity-locked (UNet fwd 1.4e-6, face e2e recon ≤2/255), realtime
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
| Audio encoder | `openai/whisper-tiny` | reuse existing MLX whisper |
| Face detect / DWPose / parsing | S3FD · DWPose · bisenet | CPU, run as-is |

## Quick start

*(pipeline lands in Phase 4 — `musetalk_mlx/pipeline_mlx.py`)*

## Dev / parity

```bash
pip install -e ".[parity]"   # pulls torch + diffusers==0.6.0 + transformers for golden capture
pytest tests/parity          # PT vs MLX, run on mx.cpu, gated on relative error
pytest tests/smoke           # shape / config / e2e noise-path
```

PyTorch is an **optional** dev dependency — end users running the MLX port never need it.

## License

MIT (mirrors upstream). Dependency models keep their own permissive licenses.
