"""Smoke: the captured config must stay byte-identical to upstream musetalk.json.

Guards against silent drift in the oracle values during the port.
"""
import json
from pathlib import Path

from musetalk_mlx import config

GOLDEN = Path(__file__).resolve().parents[2] / "goldens" / "musetalk_unet_config.json"


def test_unet_config_matches_upstream_json():
    upstream = json.loads(GOLDEN.read_text())
    for k, v in upstream.items():
        assert config.UNET_CONFIG[k] == v, f"UNET_CONFIG[{k}] drifted: {config.UNET_CONFIG.get(k)} != {v}"


def test_critical_invariants():
    c = config.UNET_CONFIG
    assert c["in_channels"] == 8            # masked ⊕ reference latent
    assert c["out_channels"] == 4
    assert c["cross_attention_dim"] == 384  # whisper-tiny feat dim
    assert c["attention_head_dim"] == 8     # heads-style misnomer
    assert config.UNET_TIMESTEP == 0        # single-step inpainting
    assert config.VAE_SCALING_FACTOR == 0.18215
