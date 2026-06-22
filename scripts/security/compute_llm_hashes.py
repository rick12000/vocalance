"""Compute SHA-256 hashes for locally cached LLM model files.

Run this script after downloading all three LLM models through the app, then
paste the output into the ``gguf_sha256`` field of each ``LocalLLMArtifact``
in ``vocalance/app/config/app_config.py``.

Usage:
    conda activate vocalance_env_dev
    python scripts/security/compute_llm_hashes.py
"""

import hashlib
import os
import sys

MODELS = [
    {
        "artifact_id": "qwen2.5-1.5b-q5km",
        "filename": "qwen2.5-1.5b-instruct-q5_k_m.gguf",
    },
    {
        "artifact_id": "qwen3-4b-q5km",
        "filename": "Qwen3-4B-Q5_K_M.gguf",
    },
    {
        "artifact_id": "qwen3-8b-q5km",
        "filename": "Qwen3-8B-Q5_K_M.gguf",
    },
]


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

    for model in MODELS:
        path = os.path.join(base, model["filename"])
        if not os.path.exists(path):
            missing.append(model)
            print(f"  [MISSING] {model['filename']}")
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Hashing {model['filename']} ({size_mb:.0f} MB)...", end="", flush=True)
        digest = sha256_of_file(path)
        print(" done")
        results.append((model, digest))

    if missing:
        print(f"\n{len(missing)} model(s) not yet downloaded. Download them through the app first.\n")

    if not results:
        sys.exit(1)

    print("\n--- Paste into app_config.py ---\n")
    for model, digest in results:
        print(
            f'LocalLLMArtifact(  # {model["artifact_id"]}\n'
            f"    gguf_sha256={{\n"
            f'        "{model["filename"]}": "{digest}",\n'
            f"    }},\n"
            f")"
        )

    print("\n--- Or as a quick reference dict ---\n")
    print("gguf_sha256 = {")
    for model, digest in results:
        print(f'    "{model["filename"]}": "{digest}",')
    print("}")


if __name__ == "__main__":
    main()
