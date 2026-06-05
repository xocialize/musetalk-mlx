"""Fetch all upstream weights into weights/ (HF hub). Idempotent."""
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

W = Path(__file__).resolve().parents[1] / "weights"
W.mkdir(exist_ok=True)

# 1. ft-mse VAE (Phase 1) — small, ~335 MB
print("[1/3] sd-vae-ft-mse ...", flush=True)
snapshot_download(
    "stabilityai/sd-vae-ft-mse", local_dir=str(W / "sd-vae-ft-mse"),
    allow_patterns=["*.json", "*.safetensors"],
)

# 2. MuseTalk v1.5 UNet (Phase 2) — unet.pth + musetalk.json
print("[2/3] MuseTalk musetalkV15 ...", flush=True)
for f in ["musetalkV15/unet.pth", "musetalkV15/musetalk.json"]:
    hf_hub_download("TMElyralab/MuseTalk", f, local_dir=str(W / "MuseTalk"))

# 3. whisper-tiny (Phase 3 audio) — HF transformers WhisperModel
print("[3/3] whisper-tiny ...", flush=True)
snapshot_download(
    "openai/whisper-tiny", local_dir=str(W / "whisper-tiny"),
    allow_patterns=["*.json", "*.safetensors", "*.txt", "vocab.json", "merges.txt"],
)

print("DONE — weights in", W, flush=True)
