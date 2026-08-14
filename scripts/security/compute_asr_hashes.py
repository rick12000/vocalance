"""Compute SHA-256 hashes for X-ASR ONNX model files.

Run this script after the ASR model has been downloaded to
``vocalance/app/assets/asr/chunk-480ms-model/`` (either by launching the app once
and letting it auto-download, or by manually running the download snippet from the
project documentation), then paste the output into the ``sha256`` field of
``_ASR_MODEL_ARTIFACT`` in ``vocalance/app/config/app_config.py``.

Usage:
    conda activate vocalance_env_dev
    python scripts/security/compute_asr_hashes.py
"""

import hashlib
import os
import sys
from pathlib import Path

FILENAMES = [
    "encoder-480ms.onnx",
    "decoder-480ms.onnx",
    "joiner-480ms.onnx",
    "tokens.txt",
]

MODEL_DIR = Path(__file__).resolve().parents[2] / "vocalance" / "app" / "assets" / "asr" / "chunk-480ms-model"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 512), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print(f"Looking for ASR model files in:\n  {MODEL_DIR}\n")

    if not MODEL_DIR.exists():
        print(
            "Directory not found. Launch Vocalance once to trigger the automatic download,\n"
            "or download the files manually and place them at the path above."
        )
        sys.exit(1)

    missing = []
    results = []

    for filename in FILENAMES:
        path = MODEL_DIR / filename
        if not path.exists():
            missing.append(filename)
            print(f"  [MISSING] {filename}")
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Hashing {filename} ({size_mb:.2f} MB)...", end="", flush=True)
        digest = sha256_of_file(path)
        print(" done")
        results.append((filename, digest))

    if missing:
        print(f"\n{len(missing)} file(s) not found. Download them first.\n")

    if not results:
        sys.exit(1)

    print("\n--- Paste into _ASR_MODEL_ARTIFACT in app_config.py ---\n")
    print("        sha256={")
    for filename, digest in results:
        print(f'            "{filename}": "{digest}",')
    print("        },")

    print("\n--- Quick reference ---\n")
    for filename, digest in results:
        print(f"{filename}: {digest}")


if __name__ == "__main__":
    main()
