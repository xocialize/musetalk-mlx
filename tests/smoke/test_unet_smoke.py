"""Smoke: UNet GPU forward (no NaN), batched, NCHW public API."""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
UNET_PTH = ROOT / "weights" / "MuseTalk" / "musetalkV15" / "unet.pth"

pytestmark = pytest.mark.skipif(not UNET_PTH.exists(), reason="download_weights.py first")


def test_unet_forward_gpu_batched():
    mx.set_default_device(mx.gpu)
    from musetalk_mlx.models.unet import UNet2DConditionModel
    from musetalk_mlx.utils.weights import load_unet_weights

    m = UNet2DConditionModel()
    load_unet_weights(m, UNET_PTH)
    m.eval()

    bs = 4
    latent = mx.array(np.random.default_rng(0).standard_normal((bs, 8, 32, 32)).astype(np.float32))
    audio = mx.array(np.random.default_rng(1).standard_normal((bs, 50, 384)).astype(np.float32))
    out = m(latent, mx.array([0]), audio)
    mx.eval(out)
    assert out.shape == (bs, 4, 32, 32)
    assert not bool(mx.isnan(out).any())
