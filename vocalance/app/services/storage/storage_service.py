import asyncio
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.storage.storage_models import (
    AgenticPromptsData,
    CommandHistoryData,
    CommandsData,
    GridClicksData,
    MarksData,
    SettingsData,
    SoundMappingsData,
    StorageData,
)

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with timestamp-based expiration tracking."""

    def __init__(self, data: Any, timestamp: float) -> None:
        self.data = data
        self.timestamp = timestamp

    def is_expired(self, ttl: float) -> bool:
        return time.time() - self.timestamp > ttl


class StorageService:
    """Type-safe storage service with caching and atomic writes.

    Provides persistent JSON storage for Pydantic models with in-memory caching,
    atomic file writes, and thread-safe operations for reliable data persistence.
    """

    def __init__(self, config: GlobalAppConfig) -> None:
        self._config = config
        self._base_dir = Path(config.storage.user_data_root)
        self._cache_ttl = config.storage.cache_ttl_seconds

        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Storage")

        # Cache
        self._cache: Dict[str, CacheEntry] = {}

        # Map model types to file paths relative to base_dir
        self._path_map: Dict[Type[StorageData], str] = {
            MarksData: os.path.join(config.storage.marks_dir, "marks.json"),
            SettingsData: os.path.join(config.storage.settings_dir, "user_settings.json"),
            CommandsData: os.path.join(config.storage.settings_dir, "custom_commands.json"),
            GridClicksData: os.path.join(config.storage.click_tracker_dir, "click_history.json"),
            AgenticPromptsData: os.path.join(config.storage.user_data_root, "dictation", "agentic_prompts.json"),
            SoundMappingsData: os.path.join(config.storage.sound_model_dir, "sound_mappings.json"),
            CommandHistoryData: os.path.join(config.storage.command_history_dir, "command_history.json"),
        }

        # Ensure directories exist
        self._ensure_directories()

        logger.debug(f"StorageService initialized with base directory: {self._base_dir}")

    def _ensure_directories(self) -> None:
        for filepath in self._path_map.values():
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _get_path(self, model_type: Type[StorageData]) -> Path:
        if model_type not in self._path_map:
            raise ValueError(f"Unknown storage model type: {model_type.__name__}")
        return Path(self._path_map[model_type])

    def _get_cache_key(self, model_type: StorageData) -> str:
        return model_type.__name__

    async def read(self, model_type: StorageData) -> StorageData:
        """Read typed data from storage with caching and automatic defaults.

        Args:
            model_type: Pydantic model class to read

        Returns:
            Instance of model with data from storage or defaults
        """
        cache_key = self._get_cache_key(model_type)

        # Check cache
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if not entry.is_expired(self._cache_ttl):
                    logger.debug(f"Cache hit for {cache_key}")
                    return entry.data
                else:
                    del self._cache[cache_key]

        # Read from disk
        path = self._get_path(model_type)

        if not path.exists():
            logger.debug(f"File does not exist: {path}, creating default instance")
            result = model_type()
            with self._lock:
                self._cache[cache_key] = CacheEntry(data=result, timestamp=time.time())
            return result

        try:
            loop = asyncio.get_event_loop()
            data_dict = await loop.run_in_executor(self._executor, self._read_json, path)

            instance = model_type.model_validate(data_dict)

            with self._lock:
                self._cache[cache_key] = CacheEntry(data=instance, timestamp=time.time())

            logger.debug(f"Read {cache_key} from storage")
            return instance

        except ValidationError as e:
            logger.error(f"Validation error reading {cache_key}: {e}")
            result = model_type()
            return result
        except Exception as e:
            logger.error(f"Error reading {cache_key}: {e}")
            result = model_type()
            return result

    async def write(self, data: StorageData) -> bool:
        """Write typed data to storage with atomic file writes and cache update.

        Args:
            data: Pydantic model instance to write

        Returns:
            True if write succeeded
        """
        model_type = type(data)
        path = self._get_path(model_type)
        cache_key = self._get_cache_key(model_type)

        try:
            data_dict = data.model_dump()

            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(self._executor, self._write_json, path, data_dict)

            if success:
                with self._lock:
                    self._cache[cache_key] = CacheEntry(data=data, timestamp=time.time())
                logger.debug(f"Wrote {cache_key} to storage")
                return True

            return False

        except Exception as e:
            logger.error(f"Error writing {cache_key}: {e}")
            return False

    def _make_serializable(self, data: Any) -> Any:
        if isinstance(data, BaseModel):
            return data.model_dump(mode="json")
        elif isinstance(data, dict):
            return {key: self._make_serializable(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_serializable(item) for item in data]
        else:
            return data

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: Dict[str, Any]) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            temp_path = path.with_suffix(f".tmp.{uuid.uuid4().hex}")

            serializable_data = self._make_serializable(data)

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

            if path.exists():
                backup_path = path.with_suffix(".backup")
                os.replace(path, backup_path)
                try:
                    os.replace(temp_path, path)
                    if backup_path.exists():
                        os.remove(backup_path)
                except Exception:
                    if backup_path.exists():
                        os.replace(backup_path, path)
                    raise
            else:
                os.replace(temp_path, path)

            return True

        except Exception as e:
            logger.error(f"Error writing JSON to {path}: {e}")
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return False

    def clear_cache(self, model_type: Optional[StorageData] = None) -> None:
        with self._lock:
            if model_type:
                cache_key = self._get_cache_key(model_type)
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.debug(f"Cleared cache for {cache_key}")
            else:
                self._cache.clear()
                logger.debug("Cleared all cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"entries": len(self._cache), "models": list(self._cache.keys()), "ttl_seconds": self._cache_ttl}

    async def shutdown(self) -> None:
        try:
            self._executor.shutdown(wait=True)
            logger.info("StorageService shutdown complete")
        except Exception as e:
            logger.error(f"Error during StorageService shutdown: {e}")

    @property
    def storage_config(self):
        """Expose storage config for path access (backward compatibility)"""
        return self._config.storage
