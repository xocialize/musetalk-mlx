"""Gate C-mel — torch-free MLX log-mel vs HF WhisperFeatureExtractor golden.

The golden (goldens/mel_hf_golden.npy) is HF's input_features for the first 30s of
yongen.wav. This test needs no torch/transformers — it validates the self-contained
MLX frontend against the captured reference.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "goldens" / "mel_hf_golden.npy"
WAV = ROOT / "refs" / "MuseTalk" / "data" / "audio" / "yongen.wav"

pytestmark = pytest.mark.skipif(not GOLDEN.exists() or not WAV.exists(),
                                reason="capture mel golden + upstream wav needed")


def test_log_mel_parity():
    mx.set_default_device(mx.cpu)
    import librosa

    from musetalk_mlx.whisper.log_mel import log_mel_spectrogram

    wav, _ = librosa.load(str(WAV), sr=16000)
    mel = np.array(log_mel_spectrogram(mx.array(wav[: 30 * 16000])))
    hf = np.load(GOLDEN)
    d = np.abs(mel - hf)
    print(f"\nlog-mel max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e}")
    assert mel.shape == hf.shape == (1, 80, 3000)
    assert d.max() < 1e-3, f"mel diverges: {d.max():.3e}"
