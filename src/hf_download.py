"""Download WildChat-1M, WildChat-4.8M, LMSYS-Chat-1M from HuggingFace.

Requires HF_TOKEN env var. Each dataset is gated; you must accept terms in the
browser before running this.

Run:  python -m src.hf_download
Outputs: data/raw/<dataset_slug>/  (HF cache symlinks)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

DATA_RAW = Path("data/raw")
DATASETS = [
    ("allenai/WildChat-1M",   "wildchat-1m"),
    ("allenai/WildChat-4.8M", "wildchat-4.8m"),
    ("lmsys/lmsys-chat-1m",   "lmsys-chat-1m"),
]


def _check_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        # Fall back to a local file (gitignored)
        local = Path(".hf_token")
        if local.exists():
            tok = local.read_text().strip()
    if not tok:
        sys.exit("ERROR: set HF_TOKEN env var or write token to .hf_token")
    return tok


def main() -> None:
    from huggingface_hub import snapshot_download

    token = _check_token()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    for repo_id, slug in DATASETS:
        target = DATA_RAW / slug
        target.mkdir(exist_ok=True)
        print(f"\n=== {repo_id} -> {target} ===")
        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target),
                local_dir_use_symlinks=False,  # actual files in data/raw
                token=token,
                allow_patterns=["*.parquet", "*.json", "*.md", "README*"],
            )
            print(f"  done: {path}")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
