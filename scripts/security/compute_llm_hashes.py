"""Compute SHA-256 hashes for whitelisted LLM GGUF files (and optionally download them).

Usage:
    conda activate vocalance_env_dev
    python scripts/security/compute_llm_hashes.py
    python scripts/security/compute_llm_hashes.py --download
    python scripts/security/compute_llm_hashes.py --download --only ministral-3-8b-iq3xs qwen3-8b-iq3m
"""

import argparse
import hashlib
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

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


def download_artifact(repo_id: str, filename: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    final_path = os.path.join(dest_dir, filename)
    temp_dir = os.path.join(dest_dir, f".{filename}.download")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=temp_dir,
        local_dir_use_symlinks=False,
    )
    if not os.path.isfile(downloaded) or os.path.getsize(downloaded) == 0:
        raise RuntimeError(f"Download failed or empty: {filename}")
    if os.path.exists(final_path):
        os.remove(final_path)
    shutil.move(downloaded, final_path)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and/or hash whitelisted LLM GGUF files.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing (or all with --force) files from Hugging Face into the app models directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the file already exists.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="MODEL_ID",
        help="Limit to these allowlist model ids.",
    )
    args = parser.parse_args()

    base = models_dir()
    os.makedirs(base, exist_ok=True)
    print(f"Models directory: {base}\n")

    only_ids = set(args.only) if args.only else None
    missing: list[str] = []
    results: list[tuple[str, str, str]] = []

    for artifact in local_llm_allowlist().artifacts:
        if only_ids is not None and artifact.id not in only_ids:
            continue
        for filename in artifact.gguf_filenames:
            path = os.path.join(base, filename)
            if args.download and (args.force or not os.path.exists(path)):
                print(f"  Downloading {artifact.id}: {filename} ...", flush=True)
                download_artifact(artifact.repo_id, filename, base)
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
        print(f"\n{len(missing)} model(s) not on disk. Run with --download first.\n")

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
