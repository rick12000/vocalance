import asyncio
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from huggingface_hub import hf_hub_download

from vocalance.app.config.app_config import GlobalAppConfig

logger = logging.getLogger(__name__)


class LLMModelDownloader:
    """Service for downloading and managing LLM models from Hugging Face

    Implements atomic downloads to prevent partial files:
    - Downloads to temporary location first
    - Only moves to final location on complete success
    - Cleans up partial files on any failure
    - Retries up to 3 times with 5s delay between attempts
    """

    def __init__(self, config: GlobalAppConfig):
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LLMDownloader")
        self._download_lock = threading.RLock()
        self._models_dir = os.path.join(config.storage.user_data_root, "llm_models")
        self._temp_dir = os.path.join(config.storage.user_data_root, "llm_models_temp")
        os.makedirs(self._models_dir, exist_ok=True)
        os.makedirs(self._temp_dir, exist_ok=True)
        logger.debug(f"LLM models directory: {self._models_dir}")
        logger.debug(f"LLM temp directory: {self._temp_dir}")

    def get_models_directory(self) -> str:
        return self._models_dir

    def model_exists(self, filename: str) -> bool:
        model_path = os.path.join(self._models_dir, filename)
        return os.path.exists(model_path) and os.path.getsize(model_path) > 0

    def get_model_path(self, filename: str) -> str:
        return os.path.join(self._models_dir, filename)

    async def download_model(
        self, repo_id: str, filename: str, force_download: bool = False, max_retries: int = 3, retry_delay_seconds: int = 5
    ) -> Optional[str]:
        """
        Download model with atomic guarantees and retry logic.

        Args:
            repo_id: HuggingFace repository ID
            filename: Model filename
            force_download: Force re-download even if exists
            max_retries: Maximum number of retry attempts
            retry_delay_seconds: Delay between retries in seconds

        Returns:
            Path to model if successful, None if failed
        """
        if (not force_download and self.model_exists(filename)) or (not force_download and self.model_exists(filename)):
            model_path = self.get_model_path(filename)
            logger.info(f"Model already exists: {filename}")
            return model_path

        with self._download_lock:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Downloading model {filename} from {repo_id} (attempt {attempt}/{max_retries})...")
                    loop = asyncio.get_event_loop()
                    downloaded_path = await loop.run_in_executor(self._executor, self._sync_download_atomic, repo_id, filename)

                    if downloaded_path:
                        logger.info(f"Model downloaded successfully: {filename}")
                        return downloaded_path

                    logger.error(f"Download failed (attempt {attempt}/{max_retries})")
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay_seconds} seconds...")
                        await asyncio.sleep(retry_delay_seconds)

                except Exception as e:
                    logger.error(f"Download error (attempt {attempt}/{max_retries}): {e}", exc_info=True)
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay_seconds} seconds...")
                        await asyncio.sleep(retry_delay_seconds)

            logger.error(f"Failed to download model after {max_retries} attempts")

            return None

    def _sync_download_atomic(self, repo_id: str, filename: str) -> Optional[str]:
        """
        Download model atomically - downloads to temp, only commits on success.
        Cleans up partial files on any failure.
        """
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
            logger.error(f"Download failed: {e}", exc_info=True)
            self._cleanup_failed_download(temp_download_dir, final_path)
            return None

    def _cleanup_failed_download(self, temp_dir: str, final_path: str) -> None:
        """Clean up temp directory and any empty final files after failed download."""
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning temp directory: {e}")

        if os.path.exists(final_path) and os.path.getsize(final_path) == 0:
            try:
                os.remove(final_path)
            except Exception as e:
                logger.error(f"Error removing empty file: {e}")

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
            logger.error(f"Error getting download status: {e}")

        return status
