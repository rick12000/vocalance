"""Compute SHA-256 hashes for locally cached LLM model files.

Run this script after downloading every whitelisted LLM model, then paste the
output into the ``gguf_sha256`` field of each ``LocalLLMArtifact`` in
``vocalance/app/config/app_config.py``.

Usage:
    conda activate vocalance_env_dev
    python scripts/security/compute_llm_hashes.py
"""

import hashlib
import os
import sys

from vocalance.app.config.app_config import local_llm_allowlist


def models_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.expanduser("~")
    return os.path.join(base, "Vocalance", "llm_models")


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 512), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    base = models_dir()
    print(f"Looking for models in: {base}\n")

    missing = []
    results = []

    for artifact in local_llm_allowlist().artifacts:
        for filename in artifact.gguf_filenames:
            path = os.path.join(base, filename)
            if not os.path.exists(path):
                missing.append(filename)
                print(f"  [MISSING] {filename}")
                continue
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Hashing {filename} ({size_mb:.0f} MB)...", end="", flush=True)
            digest = sha256_of_file(path)
            print(" done")
            results.append((artifact.id, filename, digest))

    if missing:
        print(f"\n{len(missing)} model(s) not yet downloaded. Download them through the app first.\n")

    if not results:
        sys.exit(1)

    print("\n--- Paste into app_config.py ---\n")
    for artifact_id, filename, digest in results:
        print(f"# {artifact_id}")
        print(f'"{filename}": "{digest}",')

    print("\n--- Or as a quick reference dict ---\n")
    print("gguf_sha256 = {")
    for _artifact_id, filename, digest in results:
        print(f'    "{filename}": "{digest}",')
    print("}")


if __name__ == "__main__":
    main()
