"""Download Gate D-visual aux weights: DWPose, face-parse bisenet, SyncNet."""
from pathlib import Path
from huggingface_hub import hf_hub_download
import urllib.request

W = Path(__file__).resolve().parents[1] / "weights"
(W / "face-parse-bisent").mkdir(parents=True, exist_ok=True)

print("[1/4] DWPose dw-ll_ucoco_384 ...", flush=True)
hf_hub_download("yzd-v/DWPose", "dw-ll_ucoco_384.pth", local_dir=str(W / "dwpose"))

print("[2/4] LatentSync SyncNet ...", flush=True)
hf_hub_download("ByteDance/LatentSync", "latentsync_syncnet.pt", local_dir=str(W / "syncnet"))

print("[3/4] bisenet resnet18 ...", flush=True)
urllib.request.urlretrieve(
    "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    str(W / "face-parse-bisent" / "resnet18-5c106cde.pth"),
)

print("[4/4] bisenet face-parse 79999_iter ...", flush=True)
try:
    hf_hub_download("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth",
                    local_dir=str(W / "face-parse-bisent"))
except Exception as e:
    print("  HF mirror failed, trying gdown:", e)
    import gdown
    gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812",
                   output=str(W / "face-parse-bisent" / "79999_iter.pth"))

print("DONE", flush=True)
