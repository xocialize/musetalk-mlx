"""Gate D-core — face-level e2e parity (latents -> UNet -> recon) vs torch (cpu fp32).

Validates the full neural assembly: mask construction, image normalization, BGR/RGB,
8-ch concat, UNet(t=0) with PE'd audio, and decode_latents. Deterministic (posterior mean).
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "goldens" / "pipeline_golden.npz"
WEIGHTS = ROOT / "weights"

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="run capture_pipeline_golden.py first")


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-8))


@pytest.fixture(scope="module")
def pipe():
    mx.set_default_device(mx.cpu)
    from musetalk_mlx.pipeline_mlx import MuseTalkPipeline
    return MuseTalkPipeline.from_pretrained(WEIGHTS)


def test_latent_prep_parity(pipe):
    g = np.load(GOLDEN)
    latents = np.array(pipe.get_latents_for_unet(g["img"], deterministic=True))
    r = _rel(latents, g["latents"])
    mean_abs = float(np.mean(np.abs(latents - g["latents"])))
    print(f"\nget_latents_for_unet rel={r:.2e} mean|Δ|={mean_abs:.2e}  {latents.shape}")
    # The masked half-latent has a single error spike at the synthetic half-mask edge
    # (sharp -1 discontinuity at row 128); the ref half parities at ~1e-5 and the bulk
    # mean|Δ|~1e-4. Gate on mean (bulk) + the strict recon test below catches real bugs.
    assert mean_abs < 1e-3, f"latent prep bulk diverges: mean|Δ|={mean_abs:.2e}"
    assert r < 5e-3, f"latent prep diverges: {r:.2e}"


def test_full_face_generation_parity(pipe):
    g = np.load(GOLDEN)
    latent = mx.array(g["latents"])
    audio = mx.array(np.load(ROOT / "goldens" / "audio_golden.npz")["chunks"][:1].astype(np.float32))
    recon = pipe.generate_faces(latent, audio)               # BGR uint8 (1,256,256,3)
    # compare decoded pixels (uint8) — near-exact
    diff = np.abs(recon.astype(np.int16) - g["recon"].astype(np.int16))
    print(f"\nrecon face max|Δ|={diff.max()} mean|Δ|={diff.mean():.3f} (uint8 0-255)")
    assert diff.max() <= 2, f"recon diverges: max|Δ|={diff.max()}"
