"""Gate B — MuseTalk UNet full-pass parity vs PyTorch oracle (cpu fp32).

Forward at t=0 with seeded 8-ch latent + 384-d audio cross-attn states.
Gate on relative error.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "goldens" / "unet_golden.npz"
UNET_PTH = ROOT / "weights" / "MuseTalk" / "musetalkV15" / "unet.pth"

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="run capture_unet_golden.py first")


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-8))


def test_unet_forward_parity():
    mx.set_default_device(mx.cpu)
    from musetalk_mlx.models.unet import UNet2DConditionModel
    from musetalk_mlx.utils.weights import load_unet_weights

    m = UNet2DConditionModel()
    load_unet_weights(m, UNET_PTH)
    m.eval()

    g = np.load(GOLDEN)
    latent = mx.array(g["latent"])              # (1,8,32,32) NCHW
    audio = mx.array(g["audio"])                # (1,50,384)
    ts = mx.array([0])
    out = np.array(m(latent, ts, audio))        # NCHW
    r = _rel(out, g["out"])
    print(f"\nUNet forward rel={r:.2e}  (out {out.shape})")
    assert r < 5e-3, f"UNet diverges: {r:.2e}"
