import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.storage.atomic_json import JsonReadError, JsonWriteError, read_json_dict, write_json_atomic
from vocalance.app.services.storage.storage_models import (
    AgenticPromptsData,
    AppUserConfigDocument,
    CommandsData,
    DictationAliasData,
    GridClicksData,
    MarksData,
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
    """Type-safe JSON persistence with in-memory TTL cache and atomic writes."""

    def __init__(self, config: GlobalAppConfig) -> None:
        self.config = config
        self.base_dir = Path(config.storage.user_data_root)
        self.cache_ttl = config.storage.cache_ttl_seconds
        self.lock = threading.RLock()
        self.cache: Dict[str, CacheEntry] = {}
        self.path_map: Dict[Type[StorageData], str] = {
            MarksData: os.path.join(config.storage.marks_dir, "marks.json"),
            AppUserConfigDocument: os.path.join(config.storage.settings_dir, "app_user_config.json"),
            CommandsData: os.path.join(config.storage.settings_dir, "custom_commands.json"),
            GridClicksData: os.path.join(config.storage.click_tracker_dir, "click_history.json"),
            AgenticPromptsData: os.path.join(config.storage.user_data_root, "dictation", "agentic_prompts.json"),
            SoundMappingsData: os.path.join(config.storage.sound_model_dir, "sound_mappings.json"),
            DictationAliasData: os.path.join(config.storage.user_data_root, "dictation", "aliases.json"),
        }

        self.ensure_directories()
        logger.debug("StorageService initialized with base directory: %s", self.base_dir)

    def ensure_directories(self) -> None:
        for filepath in self.path_map.values():
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def get_path(self, model_type: Type[StorageData]) -> Path:
        if model_type not in self.path_map:
            raise ValueError(f"Unknown storage model type: {model_type.__name__}")
        return Path(self.path_map[model_type])

    def get_cache_key(self, model_type: Type[StorageData]) -> str:
        return model_type.__name__

    async def read(self, model_type: Type[StorageData]) -> StorageData:
        cache_key = self.get_cache_key(model_type)

        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if not entry.is_expired(self.cache_ttl):
                    logger.debug("Cache hit for %s", cache_key)
                    return entry.data
                del self.cache[cache_key]

        path = self.get_path(model_type)

        if not path.exists():
            logger.debug("File does not exist: %s, creating default instance", path)
            result = model_type()
            with self.lock:
                self.cache[cache_key] = CacheEntry(data=result, timestamp=time.time())
            return result

        try:
            data_dict = await asyncio.to_thread(self.read_dict_from_disk, path)
            instance = model_type.model_validate(data_dict)
            with self.lock:
                self.cache[cache_key] = CacheEntry(data=instance, timestamp=time.time())
            logger.debug("Read %s from storage", cache_key)
            return instance

        except ValidationError as e:
            logger.error("Validation error reading %s: %s", cache_key, e)
            result = model_type()
            return result
        except JsonReadError as e:
            logger.error("Error reading %s: %s", cache_key, e)
            result = model_type()
            return result
        except Exception as e:
            logger.error("Error reading %s: %s", cache_key, e)
            result = model_type()
            return result

    async def write(self, data: StorageData) -> bool:
        model_type = type(data)
        path = self.get_path(model_type)
        cache_key = self.get_cache_key(model_type)

        try:
            data_dict = data.model_dump()
            success = await asyncio.to_thread(self.persist_dict_to_disk, path, data_dict)
            if success:
                with self.lock:
                    self.cache[cache_key] = CacheEntry(data=data, timestamp=time.time())
                logger.debug("Wrote %s to storage", cache_key)
                return True
            return False
        except Exception as e:
            logger.error("Error writing %s: %s", cache_key, e)
            return False

    def materialize_for_json(self, data: Any) -> Any:
        if isinstance(data, BaseModel):
            return data.model_dump(mode="json")
        if isinstance(data, dict):
            return {key: self.materialize_for_json(value) for key, value in data.items()}
        if isinstance(data, (list, tuple)):
            return [self.materialize_for_json(item) for item in data]
        return data

    def read_dict_from_disk(self, path: Path) -> Dict[str, Any]:
        return read_json_dict(path)

    def persist_dict_to_disk(self, path: Path, data: Dict[str, Any]) -> bool:
        try:
            write_json_atomic(path, self.materialize_for_json(data))
            return True
        except JsonWriteError as e:
            logger.error("Error writing JSON to %s: %s", path, e)
            return False

    def clear_cache(self, model_type: Optional[Type[StorageData]] = None) -> None:
        with self.lock:
            if model_type is not None:
                cache_key = self.get_cache_key(model_type)
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    logger.debug("Cleared cache for %s", cache_key)
            else:
                self.cache.clear()
                logger.debug("Cleared all cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {"entries": len(self.cache), "models": list(self.cache.keys()), "ttl_seconds": self.cache_ttl}

    async def shutdown(self) -> None:
        try:
            logger.info("Shutting down StorageService...")
            with self.lock:
                self.cache.clear()
                logger.debug("Storage cache cleared")
            logger.info("StorageService shutdown complete")
        except Exception as e:
            logger.error("Error during StorageService shutdown: %s", e, exc_info=True)

    @property
    def storage_config(self):
        return self.config.storage
