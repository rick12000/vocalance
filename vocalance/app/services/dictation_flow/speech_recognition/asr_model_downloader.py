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

from vocalance.app.config.app_config import GlobalAppConfig, asr_model_artifact, bpe_vocab_artifact
from vocalance.app.services.dictation_flow.hotwords import export_bpe_vocab, vocab_matches_tokens
from vocalance.app.lifecycle.worker import run_blocking

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 512
PROGRESS_INTERVAL_BYTES = 10 * 1024 * 1024
TRUSTED_GITHUB_HOSTS = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
        "objects.githubusercontent.com",
        "codeload.github.com",
    }
)
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


def host_is_trusted(host: str, trusted: frozenset[str]) -> bool:
    return host in trusted or any(host.endswith(f".{name}") for name in trusted)


def validate_hf_redirect(response: httpx.Response) -> None:
    if not response.is_redirect:
        return
    host = urlparse(response.headers.get("location", "")).hostname or ""
    if not host:
        return
    if not host_is_trusted(host, TRUSTED_HF_HOSTS):
        raise ValueError(f"Blocked redirect to untrusted host: {host!r}")


def validate_github_redirect(response: httpx.Response) -> None:
    if not response.is_redirect:
        return
    host = urlparse(response.headers.get("location", "")).hostname or ""
    if not host:
        return
    if not host_is_trusted(host, TRUSTED_GITHUB_HOSTS):
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

    def bpe_model_path(self) -> Path:
        return self.model_dir / bpe_vocab_artifact().model_filename

    def bpe_vocab_path(self) -> Path:
        return self.model_dir / bpe_vocab_artifact().vocab_filename

    def bpe_ready(self) -> bool:
        """Return True when both the SentencePiece model and exported vocab are present."""
        return self.bpe_model_path().is_file() and self.bpe_vocab_path().is_file()

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

    def sync_download_bpe_vocab(
        self,
        cancel_event: Optional[threading.Event],
        progress_cb: Optional[Callable[[str], None]],
    ) -> None:
        """Download the pinned SentencePiece model and export sherpa's vocab file.

        Existing ONNX files are left in place. A failed download removes only the
        temporary BPE files.

        Raises:
            IntegrityError: SHA-256 mismatch, or the vocab pieces differ from tokens.txt.
            RuntimeError: Download cancelled or the file was empty.
        """
        artifact = bpe_vocab_artifact()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_final = self.bpe_model_path()
        vocab_final = self.bpe_vocab_path()
        if model_final.is_file() and sha256_of_file(model_final) != artifact.sha256.lower():
            model_final.unlink()
            if vocab_final.is_file():
                vocab_final.unlink()

        temp_dir = self.model_dir.parent / "_bpe_download_temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Download cancelled: bpe vocab")
            source_model = model_final
            if not model_final.is_file():
                partial = temp_dir / artifact.model_filename
                if progress_cb:
                    progress_cb("Downloading hotword tokenizer")
                self.download_github_file(artifact.url, partial, cancel_event)
                actual = sha256_of_file(partial)
                if actual != artifact.sha256.lower():
                    raise IntegrityError(
                        f"SHA-256 mismatch for {artifact.model_filename!r}: expected {artifact.sha256} got {actual}"
                    )
                source_model = partial
            vocab_partial = temp_dir / artifact.vocab_filename
            export_bpe_vocab(source_model, vocab_partial)
            tokens_path = self.model_dir / "tokens.txt"
            if tokens_path.is_file():
                mismatch = vocab_matches_tokens(vocab_partial, tokens_path)
                if mismatch:
                    raise IntegrityError(mismatch)
            if source_model != model_final:
                if model_final.exists():
                    model_final.unlink()
                shutil.move(str(source_model), str(model_final))
            if vocab_final.exists():
                vocab_final.unlink()
            shutil.move(str(vocab_partial), str(vocab_final))
            logger.info("Hotword tokenizer ready: %s", vocab_final)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def download_github_file(
        self,
        url: str,
        destination: Path,
        cancel_event: Optional[threading.Event],
    ) -> None:
        host = urlparse(url).hostname or ""
        if not host_is_trusted(host, TRUSTED_GITHUB_HOSTS):
            raise ValueError(f"Blocked download from untrusted host: {host!r}")
        timeout = httpx.Timeout(600.0, connect=60.0)
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            event_hooks={"response": [validate_github_redirect]},
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(destination, "wb") as out:
                    for chunk in response.iter_bytes(chunk_size=CHUNK_BYTES):
                        if cancel_event and cancel_event.is_set():
                            raise RuntimeError("Download cancelled: bpe vocab")
                        if chunk:
                            out.write(chunk)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("Hotword tokenizer download produced an empty file")

    async def download_bpe_vocab(
        self,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Download and export the hotword tokenizer. Returns False when cancelled."""
        try:
            await run_blocking(
                self.sync_download_bpe_vocab,
                cancel_event,
                progress_cb,
                name="bpe-vocab-download",
            )
        except RuntimeError as exc:
            if "cancelled" in str(exc).lower():
                return False
            raise
        return self.bpe_ready()

    def cleanup_partial(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir, ignore_errors=True)
        logger.info("Cleaned up partial ASR model download")
