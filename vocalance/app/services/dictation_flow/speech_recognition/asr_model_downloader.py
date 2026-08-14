from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status

from vocalance.app.config.app_config import GlobalAppConfig, asr_model_artifact
from vocalance.app.lifecycle.worker import run_blocking

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 512
PROGRESS_INTERVAL_BYTES = 10 * 1024 * 1024
TRUSTED_HF_HOSTS = frozenset(
    {
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.huggingface.co",
        "hf.co",
        "cdn-lfs.hf.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs-eu-1.hf.co",
        "xethub.hf.co",
        "cas-bridge.xethub.hf.co",
    }
)


class IntegrityError(Exception):
    """Raised when a downloaded file's SHA-256 does not match the expected digest."""


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_hf_redirect(response: httpx.Response) -> None:
    if not response.is_redirect:
        return
    host = urlparse(response.headers.get("location", "")).hostname or ""
    if not host:
        return
    if not (host in TRUSTED_HF_HOSTS or any(host.endswith(f".{h}") for h in TRUSTED_HF_HOSTS)):
        raise ValueError(f"Blocked redirect to untrusted host: {host!r}")


class ASRModelDownloader:
    """Downloads X-ASR ONNX model files from Hugging Face into the package assets directory.

    Files are downloaded one at a time to a temporary directory adjacent to the
    destination, verified against a pinned SHA-256 digest, then atomically moved into
    place. The entire model directory is removed on any integrity or I/O failure so a
    partial download is never silently left behind.
    """

    def __init__(self, config: GlobalAppConfig) -> None:
        artifact = asr_model_artifact()
        assets_root = config.asset_paths.assets_root
        if assets_root is None:
            raise RuntimeError("Cannot resolve assets root; ASR model download is not possible.")
        asr_assets_dir = assets_root / "asr"
        self.artifact = artifact
        self.model_dir = asr_assets_dir / artifact.local_folder_name
        self.temp_dir = asr_assets_dir / "_download_temp"

    def assets_ready(self) -> bool:
        """Return True if every required ASR model file is present on disk."""
        return all((self.model_dir / fn).is_file() for fn in self.artifact.filenames)

    def sync_download_file(
        self,
        filename: str,
        cancel_event: Optional[threading.Event],
        progress_cb: Optional[Callable[[str], None]],
    ) -> Path:
        """Synchronously download one model file; returns final path on success.

        Raises:
            IntegrityError: SHA-256 mismatch after download.
            RuntimeError: Download cancelled or file missing/empty.
        """
        remote_path = f"{self.artifact.remote_folder}/{filename}"
        final_path = self.model_dir / filename
        temp_file_dir = self.temp_dir / filename
        partial_path = temp_file_dir / filename

        if temp_file_dir.exists():
            shutil.rmtree(temp_file_dir)
        temp_file_dir.mkdir(parents=True, exist_ok=True)

        if cancel_event and cancel_event.is_set():
            shutil.rmtree(temp_file_dir, ignore_errors=True)
            raise RuntimeError(f"Download cancelled: {filename}")

        url = hf_hub_url(
            self.artifact.repo_id,
            remote_path,
            repo_type="model",
            revision=self.artifact.revision,
        )
        timeout = httpx.Timeout(600.0, connect=60.0)
        logger.info("Downloading ASR model file: %s", filename)

        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            event_hooks={"response": [validate_hf_redirect]},
        ) as client:
            with client.stream("GET", url, headers=build_hf_headers()) as response:
                hf_raise_for_status(response)
                total = int(response.headers.get("content-length", "0") or "0")
                downloaded = 0
                last_progress_at = 0

                with open(partial_path, "wb") as out:
                    for chunk in response.iter_bytes(chunk_size=CHUNK_BYTES):
                        if cancel_event and cancel_event.is_set():
                            shutil.rmtree(temp_file_dir, ignore_errors=True)
                            raise RuntimeError(f"Download cancelled: {filename}")
                        if chunk:
                            out.write(chunk)
                            downloaded += len(chunk)
                            if progress_cb and total > 0 and downloaded - last_progress_at >= PROGRESS_INTERVAL_BYTES:
                                pct = min(100.0, 100.0 * downloaded / total)
                                progress_cb(f"Downloading speech-to-text model ({filename}: {pct:.0f}%)")
                                last_progress_at = downloaded

        if not partial_path.exists() or partial_path.stat().st_size == 0:
            shutil.rmtree(temp_file_dir, ignore_errors=True)
            raise RuntimeError(f"ASR download produced empty file: {filename}")

        expected = self.artifact.sha256.get(filename)
        if expected:
            actual = sha256_of_file(partial_path)
            if actual != expected.lower():
                shutil.rmtree(temp_file_dir, ignore_errors=True)
                raise IntegrityError(
                    f"SHA-256 mismatch for {filename!r}: expected {expected} got {actual}"
                )
            logger.info("SHA-256 verified: %s", filename)
        else:
            logger.error(
                "No SHA-256 hash configured for %s — cannot verify integrity. "
                "Run scripts/security/compute_asr_hashes.py and populate sha256 in app_config.py.",
                filename,
            )

        if final_path.exists():
            os.remove(final_path)
        shutil.move(str(partial_path), str(final_path))
        shutil.rmtree(temp_file_dir, ignore_errors=True)

        logger.info("ASR model file ready: %s", filename)
        return final_path

    async def download(
        self,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Download all ASR model files. Returns True on success, False on cancellation.

        Already-present files are skipped. On any failure the partial model directory
        is removed so no corrupt state is left on disk.

        Args:
            cancel_event: Threading event; when set, download aborts at the next checkpoint.
            progress_cb: Called with a human-readable status string during download.

        Returns:
            True when every file in the artifact is present on disk.

        Raises:
            IntegrityError: A downloaded file's SHA-256 did not match the pinned digest.
            RuntimeError: A non-cancellation download failure occurred.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)

        for i, filename in enumerate(self.artifact.filenames):
            if cancel_event and cancel_event.is_set():
                return False

            if (self.model_dir / filename).is_file():
                logger.info("ASR model file already present, skipping: %s", filename)
                continue

            if progress_cb:
                progress_cb(f"Downloading speech-to-text model ({i + 1}/{len(self.artifact.filenames)})")

            try:
                await run_blocking(
                    self.sync_download_file,
                    filename,
                    cancel_event,
                    progress_cb,
                    name=f"asr-download-{filename}",
                )
            except RuntimeError as exc:
                self.cleanup_partial()
                if "cancelled" in str(exc).lower():
                    return False
                raise
            except Exception:
                self.cleanup_partial()
                raise

        return self.assets_ready()

    def cleanup_partial(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir, ignore_errors=True)
        logger.info("Cleaned up partial ASR model download")
