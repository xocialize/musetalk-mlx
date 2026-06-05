"""Gate A — ft-mse VAE encode/decode parity vs the PyTorch oracle.

Run on mx.cpu for true fp32 (Apple-GPU fp32 matmul is tf32-like). Gate on
RELATIVE error (VAE activations have large magnitude).

Prereqs: scripts/capture_vae_golden.py has produced goldens/vae_golden.npz, and
weights/sd-vae-ft-mse/ exists (scripts/download_weights.py).
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "goldens" / "vae_golden.npz"
VAE_DIR = ROOT / "weights" / "sd-vae-ft-mse"

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="run capture_vae_golden.py first")


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-8))


@pytest.fixture(scope="module")
def vae():
    mx.set_default_device(mx.cpu)
    from musetalk_mlx.models.vae import AutoencoderKL
    from musetalk_mlx.utils.weights import load_vae_weights

    m = AutoencoderKL()
    load_vae_weights(m, VAE_DIR)
    m.eval()
    return m


def test_encode_moments_parity(vae):
    g = np.load(GOLDEN)
    x = mx.array(g["img"])                                  # (1,3,256,256) NCHW
    post = vae.encode(x)                                    # DiagonalGaussian (NCHW)
    mean = np.array(post.mean)
    logvar = np.array(post.logvar)
    r_mean = _rel(mean, g["enc_mean"])
    r_logvar = _rel(logvar, g["enc_logvar"])
    print(f"\nVAE encode  mean rel={r_mean:.2e}  logvar rel={r_logvar:.2e}")
    assert r_mean < 1e-4, f"encode mean diverges: {r_mean:.2e}"


def test_decode_parity(vae):
    g = np.load(GOLDEN)
    z = mx.array(g["dec_latent"])                           # (1,4,32,32) NCHW
    out = np.array(vae.decode(z))                           # NCHW
    r = _rel(out, g["dec_out"])
    print(f"\nVAE decode  rel={r:.2e}")
    assert r < 1e-4, f"decode diverges: {r:.2e}"
