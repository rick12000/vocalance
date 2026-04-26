from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx
from huggingface_hub import hf_hub_url
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.lifecycle.worker import run_blocking

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1024 * 512
_PROGRESS_INTERVAL_BYTES = 50 * 1024 * 1024


class LLMModelDownloader:
    """Hugging Face GGUF downloads: hub client (retries) or HTTP stream (cancellable)."""

    def __init__(self, config: GlobalAppConfig) -> None:
        self._config = config
        self._download_lock = threading.RLock()
        self._models_dir = os.path.join(config.storage.user_data_root, "llm_models")
        self._temp_dir = os.path.join(config.storage.user_data_root, "llm_models_temp")
        os.makedirs(self._models_dir, exist_ok=True)
        os.makedirs(self._temp_dir, exist_ok=True)
        logger.debug("LLM models directory: %s", self._models_dir)
        logger.debug("LLM temp directory: %s", self._temp_dir)

    def get_models_directory(self) -> str:
        return self._models_dir

    def shutdown(self) -> None:
        """No-op: download workers are daemon threads owned by the lifecycle."""
        return None

    def model_exists(self, filename: str) -> bool:
        model_path = os.path.join(self._models_dir, filename)
        return os.path.exists(model_path) and os.path.getsize(model_path) > 0

    def model_bundle_complete(self, filenames: Sequence[str]) -> bool:
        return all(self.model_exists(fn) for fn in filenames)

    def get_model_path(self, filename: str) -> str:
        return os.path.join(self._models_dir, filename)

    def remove_bundle_artifacts(self, filenames: Sequence[str]) -> None:
        """Remove committed GGUF files and known temp dirs for these filenames (best-effort)."""
        for fn in filenames:
            final_path = self.get_model_path(fn)
            if os.path.isfile(final_path):
                try:
                    os.remove(final_path)
                except OSError as e:
                    logger.warning("Could not remove %s: %s", final_path, e)
            self._scrub_temp_dirs_for_filename(fn)

    def _scrub_temp_dirs_for_filename(self, fn: str) -> None:
        for suffix in (f"{fn}_download", f"{fn}_stream_dl"):
            temp_dir = os.path.join(self._temp_dir, suffix)
            if os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except OSError as e:
                    logger.warning("Could not remove temp dir %s: %s", temp_dir, e)

    def revert_partial_bundle(self, filenames: Sequence[str], had_on_disk_before: Dict[str, bool]) -> None:
        """Remove only files that were not present before a download attempt; scrub temp dirs for all parts."""
        for fn in filenames:
            if had_on_disk_before.get(fn, False):
                continue
            final_path = self.get_model_path(fn)
            if os.path.isfile(final_path):
                try:
                    os.remove(final_path)
                except OSError as e:
                    logger.warning("Could not revert %s: %s", final_path, e)
        for fn in filenames:
            self._scrub_temp_dirs_for_filename(fn)

    def _sync_stream_download_file(
        self,
        repo_id: str,
        filename: str,
        cancel_event: threading.Event,
        progress_message_cb: Optional[Callable[[str], None]],
    ) -> Optional[str]:
        """Stream one file to models dir; returns final path or None on failure/cancel."""
        final_path = self.get_model_path(filename)
        stream_temp_root = os.path.join(self._temp_dir, f"{filename}_stream_dl")
        partial_path = os.path.join(stream_temp_root, filename)

        try:
            if os.path.isdir(stream_temp_root):
                shutil.rmtree(stream_temp_root)
            os.makedirs(stream_temp_root, exist_ok=True)

            url = hf_hub_url(repo_id, filename, repo_type="model")
            headers = build_hf_headers()
            timeout = httpx.Timeout(600.0, connect=60.0)

            if cancel_event.is_set():
                return None

            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                with client.stream("GET", url, headers=headers) as response:
                    hf_raise_for_status(response)
                    try:
                        total = int(response.headers.get("content-length", "0") or "0")
                    except ValueError:
                        total = 0
                    downloaded = 0
                    last_progress_at = 0

                    with open(partial_path, "wb") as out:
                        for chunk in response.iter_bytes(chunk_size=_CHUNK_BYTES):
                            if cancel_event.is_set():
                                logger.info("Download cancelled: %s", filename)
                                return None
                            if chunk:
                                out.write(chunk)
                                downloaded += len(chunk)
                                if progress_message_cb and total > 0 and downloaded - last_progress_at >= _PROGRESS_INTERVAL_BYTES:
                                    pct = min(100.0, 100.0 * downloaded / total)
                                    progress_message_cb(f"{filename}: {pct:.0f}%")
                                    last_progress_at = downloaded

            if not os.path.exists(partial_path) or os.path.getsize(partial_path) == 0:
                logger.error("Stream download produced empty file")
                return None

            if cancel_event.is_set():
                return None

            if os.path.exists(final_path):
                os.remove(final_path)
            shutil.move(partial_path, final_path)
            shutil.rmtree(stream_temp_root, ignore_errors=True)
            return final_path

        except Exception as e:
            logger.error("Stream download failed: %s", e, exc_info=True)
            if os.path.isdir(stream_temp_root):
                shutil.rmtree(stream_temp_root, ignore_errors=True)
            if os.path.exists(final_path) and os.path.getsize(final_path) == 0:
                try:
                    os.remove(final_path)
                except OSError:
                    pass
            return None

    def _sync_download_atomic(self, repo_id: str, filename: str) -> Optional[str]:
        """Download model atomically via huggingface_hub (non-cancellable, with hub retries inside)."""
        final_path = self.get_model_path(filename)
        temp_download_dir = os.path.join(self._temp_dir, f"{filename}_download")

        try:
            if os.path.exists(temp_download_dir):
                shutil.rmtree(temp_download_dir)
            os.makedirs(temp_download_dir, exist_ok=True)

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=temp_download_dir,
                local_dir_use_symlinks=False,
                resume_download=False,
            )

            if not os.path.exists(downloaded_path) or os.path.getsize(downloaded_path) == 0:
                logger.error("Download failed: file missing or empty")
                return None

            if os.path.exists(final_path):
                os.remove(final_path)

            shutil.move(downloaded_path, final_path)
            shutil.rmtree(temp_download_dir)

            return final_path

        except Exception as e:
            logger.error("Download failed: %s", e, exc_info=True)
            self._cleanup_failed_download(temp_download_dir, final_path)
            return None

    def _cleanup_failed_download(self, temp_dir: str, final_path: str) -> None:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error("Error cleaning temp directory: %s", e)

        if os.path.exists(final_path) and os.path.getsize(final_path) == 0:
            try:
                os.remove(final_path)
            except Exception as e:
                logger.error("Error removing empty file: %s", e)

    async def download_model(
        self,
        repo_id: str,
        filename: str,
        force_download: bool = False,
        max_retries: int = 3,
        retry_delay_seconds: int = 5,
        cancel_event: Optional[threading.Event] = None,
        progress_message_cb: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Download a single GGUF with optional cancel/progress hooks and hub retries."""
        if not force_download and self.model_exists(filename):
            model_path = self.get_model_path(filename)
            logger.info("Model already exists: %s", filename)
            return model_path

        with self._download_lock:
            if cancel_event is not None:

                def _run_stream() -> Optional[str]:
                    return self._sync_stream_download_file(repo_id, filename, cancel_event, progress_message_cb)

                return await run_blocking(_run_stream, name=f"llm-stream-{filename}")

            for attempt in range(1, max_retries + 1):
                try:
                    logger.info("Downloading model %s from %s (attempt %s/%s)...", filename, repo_id, attempt, max_retries)
                    downloaded_path = await run_blocking(
                        self._sync_download_atomic, repo_id, filename, name=f"llm-download-{filename}"
                    )

                    if downloaded_path:
                        logger.info("Model downloaded successfully: %s", filename)
                        return downloaded_path

                    logger.error("Download failed (attempt %s/%s)", attempt, max_retries)
                    if attempt < max_retries:
                        logger.info("Retrying in %s seconds...", retry_delay_seconds)
                        await asyncio.sleep(retry_delay_seconds)

                except Exception as e:
                    logger.error("Download error (attempt %s/%s): %s", attempt, max_retries, e, exc_info=True)
                    if attempt < max_retries:
                        logger.info("Retrying in %s seconds...", retry_delay_seconds)
                        await asyncio.sleep(retry_delay_seconds)

            logger.error("Failed to download model after %s attempts", max_retries)
            return None

    async def download_model_bundle(
        self,
        repo_id: str,
        filenames: List[str],
        force_download: bool = False,
        max_retries: int = 3,
        retry_delay_seconds: int = 5,
        cancel_event: Optional[threading.Event] = None,
        progress_message_cb: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Download every file in order; returns path to the first file if all succeed."""
        for i, fn in enumerate(filenames):
            if cancel_event is not None and cancel_event.is_set():
                return None
            label = f"File {i + 1}/{len(filenames)}: {fn}"
            if progress_message_cb:
                progress_message_cb(label)
            path = await self.download_model(
                repo_id=repo_id,
                filename=fn,
                force_download=force_download,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                cancel_event=cancel_event,
                progress_message_cb=progress_message_cb,
            )
            if not path:
                return None
        return self.get_model_path(filenames[0])

    def get_download_status(self) -> Dict[str, Any]:
        status = {"models_directory": self._models_dir, "available_models": [], "total_size_mb": 0}

        try:
            if os.path.exists(self._models_dir):
                for filename in os.listdir(self._models_dir):
                    if filename.endswith(".gguf"):
                        file_path = os.path.join(self._models_dir, filename)
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        status["available_models"].append({"filename": filename, "size_mb": round(size_mb, 2)})
                        status["total_size_mb"] += size_mb

            status["total_size_mb"] = round(status["total_size_mb"], 2)

        except Exception as e:
            logger.error("Error getting download status: %s", e)

        return status
