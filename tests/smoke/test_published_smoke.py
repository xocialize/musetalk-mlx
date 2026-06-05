"""Smoke: published MLX variants reload torch-free and generate valid faces."""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
GOLDEN = ROOT / "goldens" / "pipeline_golden.npz"

pytestmark = pytest.mark.skipif(
    not (DIST / "MuseTalk-1.5-MLX-fp16" / "unet.safetensors").exists() or not GOLDEN.exists(),
    reason="run export_mlx.py first",
)


@pytest.mark.parametrize("variant,max_mean", [("fp16", 1.0), ("q8", 1.5), ("q4", 5.0)])
def test_variant_reload_and_generate(variant, max_mean):
    mx.set_default_device(mx.gpu)
    from musetalk_mlx.pipeline_mlx import MuseTalkPipeline

    g = np.load(GOLDEN)
    chunk = np.load(ROOT / "goldens" / "audio_golden.npz")["chunks"][:1].astype(np.float32)
    pipe = MuseTalkPipeline.from_pretrained_mlx(DIST / f"MuseTalk-1.5-MLX-{variant}")
    recon = pipe.generate_faces(
        mx.array(g["latents"]).astype(mx.float16), mx.array(chunk).astype(mx.float16)
    )
    assert recon.shape == (1, 256, 256, 3) and recon.dtype == np.uint8
    d = np.abs(recon[0].astype(np.int16) - g["recon"][0].astype(np.int16))
    assert d.mean() < max_mean, f"{variant} recon drifted: mean|Δ|={d.mean():.3f}"
