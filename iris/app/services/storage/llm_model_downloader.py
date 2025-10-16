"""
LLM Model Downloader Service

Handles downloading and caching of LLM models from Hugging Face Hub.
"""

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from huggingface_hub import hf_hub_download

from iris.app.config.app_config import GlobalAppConfig

logger = logging.getLogger(__name__)


class LLMModelDownloader:
    """Service for downloading and managing LLM models from Hugging Face"""

    def __init__(self, config: GlobalAppConfig):
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LLMDownloader")
        self._download_lock = threading.RLock()
        self._models_dir = os.path.join(config.storage.user_data_root, "llm_models")
        os.makedirs(self._models_dir, exist_ok=True)
        logger.info(f"LLM models directory: {self._models_dir}")

    def get_models_directory(self) -> str:
        return self._models_dir

    def model_exists(self, filename: str) -> bool:
        model_path = os.path.join(self._models_dir, filename)
        return os.path.exists(model_path) and os.path.getsize(model_path) > 0

    def get_model_path(self, filename: str) -> str:
        return os.path.join(self._models_dir, filename)

    async def download_model(self, repo_id: str, filename: str, force_download: bool = False) -> Optional[str]:
        model_path = self.get_model_path(filename)

        if not force_download and self.model_exists(filename):
            logger.info(f"Model already exists: {filename}")
            return model_path

        with self._download_lock:
            if not force_download and self.model_exists(filename):
                return model_path

            try:
                logger.info(f"Downloading model {filename} from {repo_id}...")
                loop = asyncio.get_event_loop()
                downloaded_path = await loop.run_in_executor(self._executor, self._sync_download, repo_id, filename)

                if downloaded_path:
                    logger.info(f"Model downloaded successfully: {filename}")
                    return downloaded_path
                logger.error(f"Failed to download model: {filename}")
                return None

            except Exception as e:
                logger.error(f"Error downloading model {filename}: {e}", exc_info=True)
                return None

    def _sync_download(self, repo_id: str, filename: str) -> Optional[str]:
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id, filename=filename, local_dir=self._models_dir, local_dir_use_symlinks=False, resume_download=True
            )
            return downloaded_path
        except Exception as e:
            logger.error(f"Sync download error: {e}")
            return None

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
