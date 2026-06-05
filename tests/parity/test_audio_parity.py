"""Gate C — whisper-tiny encoder + chunking parity vs PyTorch oracle (cpu fp32)."""
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "goldens" / "audio_golden.npz"
WH_DIR = ROOT / "weights" / "whisper-tiny"

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="run capture_audio_golden.py first")


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-8))


def test_encoder_parity():
    mx.set_default_device(mx.cpu)
    from musetalk_mlx.whisper.whisper_encoder import WhisperEncoder
    from musetalk_mlx.utils.weights import load_whisper_encoder_weights

    enc = WhisperEncoder()
    load_whisper_encoder_weights(enc, WH_DIR)
    enc.eval()

    g = np.load(GOLDEN)
    mel = mx.array(g["mel0"])                       # (1,80,3000)
    stacked = np.array(enc(mel))                    # (1,1500,5,384)
    r = _rel(stacked, g["stacked"])
    print(f"\nwhisper encoder stacked rel={r:.2e}  {stacked.shape}")
    assert r < 1e-3, f"encoder diverges: {r:.2e}"


def test_chunk_parity():
    mx.set_default_device(mx.cpu)
    from musetalk_mlx.whisper.audio2feature import get_whisper_chunk

    g = np.load(GOLDEN)
    stacked = mx.array(g["stacked"])
    librosa_length = 128000                         # yongen.wav @16k (see capture script)
    chunks = np.array(get_whisper_chunk(stacked, librosa_length))
    assert chunks.shape == tuple(g["chunks"].shape), (chunks.shape, g["chunks"].shape)
    r = _rel(chunks, g["chunks"])
    print(f"\nchunk rel={r:.2e}  {chunks.shape}")
    assert r < 1e-5, f"chunk logic diverges: {r:.2e}"


def test_positional_encoding_formula():
    from musetalk_mlx.whisper.audio2feature import positional_encoding

    seq, d = 50, 384
    pe = np.array(positional_encoding(seq, d))[0]    # (50,384)
    # numpy reference (matches torch PositionalEncoding)
    pos = np.arange(seq)[:, None]
    div = np.exp(np.arange(0, d, 2) * (-math.log(10000.0) / d))
    ref = np.zeros((seq, d), dtype=np.float32)
    ref[:, 0::2] = np.sin(pos * div)
    ref[:, 1::2] = np.cos(pos * div)
    r = float(np.max(np.abs(pe - ref)))
    print(f"\nPE max-abs vs ref={r:.2e}")
    assert r < 1e-5
