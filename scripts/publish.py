"""Publish exported MLX variants to mlx-community. Run explicitly (pushes public repos).

  python scripts/publish.py            # all variants
  python scripts/publish.py q4 q8      # subset
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ORG = "mlx-community"

# local dist dir -> HF repo name
REPOS = {
    "fp16": "MuseTalk-1.5-fp16",
    "q8": "MuseTalk-1.5-q8",
    "q4": "MuseTalk-1.5-q4",
}

api = HfApi()
want = sys.argv[1:] or list(REPOS)
for v in want:
    local = DIST / f"MuseTalk-1.5-MLX-{v}"
    repo_id = f"{ORG}/{REPOS[v]}"
    assert local.exists(), f"missing {local} (run export_mlx.py)"
    print(f"-> {repo_id}  ({local.name})")
    create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(local), repo_id=repo_id, repo_type="model",
                      commit_message="Add MuseTalk 1.5 MLX port (xocialize-code / MVS Collective)")
    print(f"   published https://huggingface.co/{repo_id}")
