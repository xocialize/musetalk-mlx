"""Smoke: VAE shapes + GPU roundtrip (no NaN), NCHW public API."""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
VAE_DIR = ROOT / "weights" / "sd-vae-ft-mse"

pytestmark = pytest.mark.skipif(not VAE_DIR.exists(), reason="download_weights.py first")


def test_vae_roundtrip_gpu():
    mx.set_default_device(mx.gpu)
    from musetalk_mlx.models.vae import AutoencoderKL
    from musetalk_mlx.utils.weights import load_vae_weights

    m = AutoencoderKL()
    load_vae_weights(m, VAE_DIR)
    m.eval()

    x = mx.array(np.random.default_rng(0).standard_normal((1, 3, 256, 256)).astype(np.float32))
    post = m.encode(x)
    assert post.mean.shape == (1, 4, 32, 32)            # NCHW latent
    latent = post.mean * m.scaling_factor
    dec = m.decode(latent / m.scaling_factor)
    mx.eval(dec)
    assert dec.shape == (1, 3, 256, 256)
    assert not bool(mx.isnan(dec).any())
