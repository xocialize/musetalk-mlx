"""Capture the PyTorch UNet oracle (MuseTalk v1.5) for Gate B parity.

Loads diffusers UNet2DConditionModel(**musetalk.json) + musetalkV15/unet.pth, runs a
deterministic forward at t=0 with seeded 8-ch latent + 384-d audio cross-attn states.
Also introspects per-attention head counts (resolves the attention_head_dim=8 misnomer)
and dumps the RAW state_dict key inventory (lesson M1).
"""
from pathlib import Path
import json
import numpy as np
import torch
from diffusers import UNet2DConditionModel

ROOT = Path(__file__).resolve().parents[1]
UNET_DIR = ROOT / "weights" / "MuseTalk" / "musetalkV15"
OUT = ROOT / "goldens"
OUT.mkdir(exist_ok=True)

cfg = json.loads((UNET_DIR / "musetalk.json").read_text())
cfg.pop("_class_name", None)
cfg.pop("_diffusers_version", None)

unet = UNet2DConditionModel(**cfg).eval()
sd = torch.load(UNET_DIR / "unet.pth", map_location="cpu", weights_only=True)
missing, unexpected = unet.load_state_dict(sd, strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print("  missing[:5]:", missing[:5])
if unexpected:
    print("  unexpected[:5]:", unexpected[:5])

# RAW key inventory (from the .pth, not post-load)
(OUT / "unet_key_inventory.txt").write_text("\n".join(sd.keys()))
print(f"raw state_dict: {len(sd)} tensors")

# introspect attention head counts (the misnomer resolver)
head_report = []
for name, mod in unet.named_modules():
    if mod.__class__.__name__ == "Attention" and hasattr(mod, "heads"):
        head_report.append((name, mod.heads))
(OUT / "unet_head_report.txt").write_text(
    "\n".join(f"{n}\theads={h}" for n, h in head_report)
)
# summarize unique head counts at each resolution
print("attention heads (first 6):", [h for _, h in head_report[:6]])
print("unique head counts:", sorted(set(h for _, h in head_report)))

# deterministic forward at t=0
rng = np.random.default_rng(1234)
latent = rng.standard_normal((1, 8, 32, 32)).astype(np.float32)        # masked⊕ref
audio = rng.standard_normal((1, 50, 384)).astype(np.float32)           # encoder_hidden_states
ts = torch.tensor([0], dtype=torch.long)

with torch.no_grad():
    out = unet(torch.from_numpy(latent), ts, encoder_hidden_states=torch.from_numpy(audio)).sample
    out = out.cpu().numpy()

np.savez(OUT / "unet_golden.npz", latent=latent, audio=audio, out=out)
print("out", out.shape, "| saved goldens/unet_golden.npz")
