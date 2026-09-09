"""Download the XDen-1K dataset (zhangjxx/XDen-1K, CC-BY-4.0, ~16.1 GB) from
Hugging Face into data/xden1k/, entirely on D: (both the HF cache and the
final files) since C: has almost no free space on this machine.

Known trap, documented in the project plan: the dataset's own metadata.json
uses zero-based string keys and the field 'size' (cm bounding-box extents),
not '01'-style keys or a field called 'scale' as the README claims -- see
ballast.data.xden for the code that actually reads it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / ".caches"
DATA_DIR = REPO_ROOT / "data" / "xden1k"

os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
os.environ.setdefault("TMPDIR", str(CACHE_ROOT / "tmp"))
os.environ.setdefault("TEMP", str(CACHE_ROOT / "tmp"))
os.environ.setdefault("TMP", str(CACHE_ROOT / "tmp"))

for p in (CACHE_ROOT / "hf", CACHE_ROOT / "tmp", DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> None:
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"Downloading zhangjxx/XDen-1K to {DATA_DIR} ...")
    path = snapshot_download(
        repo_id="zhangjxx/XDen-1K",
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        max_workers=4,
    )
    print(f"Done: {path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
