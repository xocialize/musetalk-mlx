"""Smoke: fp16 batched generation runs and stays near fp32 (realtime path)."""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "goldens" / "pipeline_golden.npz"
WEIGHTS = ROOT / "weights"

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="run capture_pipeline_golden.py first")


def test_fp16_batched_generation():
    mx.set_default_device(mx.gpu)
    from musetalk_mlx.pipeline_mlx import MuseTalkPipeline

    g = np.load(GOLDEN)
    chunk = np.load(ROOT / "goldens" / "audio_golden.npz")["chunks"][:1].astype(np.float32)
    pipe = MuseTalkPipeline.from_pretrained(WEIGHTS).astype(mx.float16)

    n = 6
    latents = mx.array(np.repeat(g["latents"], n, 0)).astype(mx.float16)
    chunks = mx.array(np.repeat(chunk, n, 0)).astype(mx.float16)
    recon = pipe.run_batched(latents, chunks, batch_size=4)
    assert recon.shape == (n, 256, 256, 3)
    assert recon.dtype == np.uint8
    # fp16 GPU matmul is batch-size dependent (different kernels) -> per-pixel max varies
    # with bs; gate on MEAN abs error (robust), which stays ~0.4/255.
    d = np.abs(recon[0].astype(np.int16) - g["recon"][0].astype(np.int16))
    assert d.mean() < 1.0, f"fp16 recon drifted: mean|Δ|={d.mean():.3f}"
