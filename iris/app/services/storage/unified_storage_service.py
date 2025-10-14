"""
Streamlined Unified Storage Service

Centralized, high-performance storage manager with simplified architecture.
Handles all persistent data with low latency and atomic operations.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

from iris.app.event_bus import EventBus
from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.automation_command_registry import AutomationCommandRegistry
from iris.app.config.command_types import AutomationCommand

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Storage data types"""
    SETTINGS = "settings"
    COMMANDS = "commands" 
    GRID_CLICKS = "grid_clicks"
    MARKS = "marks"
    AGENTIC_PROMPTS = "agentic_prompts"
    SOUND_MAPPINGS = "sound_mappings"
    COMMAND_HISTORY = "command_history"


@dataclass
class StorageKey:
    """Storage item identifier"""
    storage_type: StorageType
    key: str
    
    def __str__(self) -> str:
        return f"{self.storage_type.value}:{self.key}"


@dataclass
class CacheEntry:
    """Cached storage entry"""
    data: Any
    timestamp: float
    access_count: int = 0
    
    def is_expired(self, ttl: float) -> bool:
        return time.time() - self.timestamp > ttl


class UnifiedStorageService:
    """Streamlined storage service with centralized file operations"""
    
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig):
        self._event_bus = event_bus
        self._config = config
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Storage")
        
        # Cache with 5-minute TTL
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_ttl = 300.0
        
        # Storage paths
        self._paths = self._init_paths()
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.info("UnifiedStorageService initialized")
    
    @property
    def storage_config(self):
        """Expose storage config for path access"""
        return self._config.storage
    
    def _init_paths(self) -> Dict[StorageType, str]:
        """Initialize storage file paths"""
        storage = self._config.storage
        
        return {
            StorageType.SETTINGS: os.path.join(storage.settings_dir, "user_settings.json"),
            StorageType.COMMANDS: os.path.join(storage.settings_dir, "custom_commands.json"),
            StorageType.GRID_CLICKS: os.path.join(storage.click_tracker_dir, "click_history.json"),
            StorageType.MARKS: os.path.join(storage.marks_dir, "marks.json"),
            StorageType.AGENTIC_PROMPTS: os.path.join(storage.user_data_root, "dictation", "agentic_prompts.json"),
            StorageType.SOUND_MAPPINGS: os.path.join(storage.sound_model_dir, "sound_mappings.json"),
            StorageType.COMMAND_HISTORY: os.path.join(storage.command_history_dir, "command_history.json")
        }
    
    def _ensure_directories(self) -> None:
        """Ensure all storage directories exist"""
        for filepath in self._paths.values():
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    async def read(self, storage_key: StorageKey, default: Any = None) -> Any:
        """Read data with caching"""
        cache_key = str(storage_key)
        
        with self._lock:
            # Check cache first
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if not entry.is_expired(self._cache_ttl):
                    entry.access_count += 1
                    return entry.data
                else:
                    del self._cache[cache_key]
        
        # Read from storage
        filepath = self._paths[storage_key.storage_type]
        data = await self._read_file(filepath, default)
        
        # Cache the result
        with self._lock:
            self._cache[cache_key] = CacheEntry(data=data, timestamp=time.time())
        
        return data
    
    async def write(self, storage_key: StorageKey, data: Any, immediate: bool = True) -> bool:
        """Write data atomically"""
        filepath = self._paths[storage_key.storage_type]
        success = await self._write_file(filepath, data)
        
        if success:
            # Update cache
            cache_key = str(storage_key)
            with self._lock:
                self._cache[cache_key] = CacheEntry(data=data, timestamp=time.time())
        
        return success
    
    async def _read_file(self, filepath: str, default: Any = None) -> Any:
        """Read JSON file"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self._sync_read, filepath, default)
        except Exception as e:
            logger.error(f"Read error {filepath}: {e}")
            return default
    
    def _sync_read(self, filepath: str, default: Any) -> Any:
        """Synchronous file read"""
        if not os.path.exists(filepath):
            return default
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"JSON read error {filepath}: {e}")
            return default
    
    async def _write_file(self, filepath: str, data: Any) -> bool:
        """Write JSON file atomically"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self._sync_write, filepath, data)
        except Exception as e:
            logger.error(f"Write error {filepath}: {e}")
            return False
    
    def _sync_write(self, filepath: str, data: Any) -> bool:
        """Synchronous atomic file write"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Write to temp file first
            temp_path = f"{filepath}.tmp.{uuid.uuid4().hex}"
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            if os.path.exists(filepath):
                backup_path = f"{filepath}.backup"
                os.rename(filepath, backup_path)
                try:
                    os.rename(temp_path, filepath)
                    os.remove(backup_path)
                except Exception:
                    # Restore backup on failure
                    if os.path.exists(backup_path):
                        os.rename(backup_path, filepath)
                    raise
            else:
                os.rename(temp_path, filepath)
            
            return True
            
        except Exception as e:
            logger.error(f"Sync write error {filepath}: {e}")
            # Clean up temp file
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
            return False
    
    def clear_cache(self, storage_type: Optional[StorageType] = None) -> None:
        """Clear cache entries"""
        with self._lock:
            if storage_type:
                keys_to_remove = [k for k in self._cache.keys() if k.startswith(storage_type.value)]
                for key in keys_to_remove:
                    del self._cache[key]
            else:
                self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_accesses = sum(entry.access_count for entry in self._cache.values())
            return {
                'entries': len(self._cache),
                'total_accesses': total_accesses,
                'hit_rate': f"{len(self._cache) / max(total_accesses, 1) * 100:.1f}%"
            }
    
    async def shutdown(self) -> None:
        """Shutdown service"""
        try:
            self._executor.shutdown(wait=True)
            logger.info("UnifiedStorageService shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# Convenience functions for common operations
async def read_settings(storage: UnifiedStorageService, key: str = "user_settings", default: Any = None) -> Any:
    """Read user settings"""
    return await storage.read(StorageKey(StorageType.SETTINGS, key), default)

async def write_settings(storage: UnifiedStorageService, key: str, value: Any) -> bool:
    """Write user settings"""
    return await storage.write(StorageKey(StorageType.SETTINGS, key), value)

async def read_commands(storage: UnifiedStorageService, default: Any = None) -> Any:
    """Read commands"""
    return await storage.read(StorageKey(StorageType.COMMANDS, "all"), default)

async def write_commands(storage: UnifiedStorageService, value: Any) -> bool:
    """Write commands"""
    return await storage.write(StorageKey(StorageType.COMMANDS, "all"), value)

async def read_agentic_prompts(storage: UnifiedStorageService, default: Any = None) -> Any:
    """Read agentic prompts"""
    return await storage.read(StorageKey(StorageType.AGENTIC_PROMPTS, "prompts"), default)

async def write_agentic_prompts(storage: UnifiedStorageService, value: Any) -> bool:
    """Write agentic prompts"""
    return await storage.write(StorageKey(StorageType.AGENTIC_PROMPTS, "prompts"), value)

async def read_sound_mappings(storage: UnifiedStorageService, default: Any = None) -> Any:
    """Read sound mappings"""
    return await storage.read(StorageKey(StorageType.SOUND_MAPPINGS, "mappings"), default)

async def write_sound_mappings(storage: UnifiedStorageService, value: Any) -> bool:
    """Write sound mappings"""
    return await storage.write(StorageKey(StorageType.SOUND_MAPPINGS, "mappings"), value)

async def read_marks(storage: UnifiedStorageService, default: Any = None) -> Any:
    """Read marks"""
    return await storage.read(StorageKey(StorageType.MARKS, "marks"), default)

async def write_marks(storage: UnifiedStorageService, value: Any) -> bool:
    """Write marks"""
    return await storage.write(StorageKey(StorageType.MARKS, "marks"), value)

async def read_grid_clicks(storage: UnifiedStorageService, default: Any = None) -> Any:
    """Read grid clicks"""
    return await storage.read(StorageKey(StorageType.GRID_CLICKS, "clicks"), default)

async def write_grid_clicks(storage: UnifiedStorageService, value: Any) -> bool:
    """Write grid clicks"""
    return await storage.write(StorageKey(StorageType.GRID_CLICKS, "clicks"), value)

async def read_command_history(storage: UnifiedStorageService, default: Any = None) -> Any:
    """Read command history"""
    return await storage.read(StorageKey(StorageType.COMMAND_HISTORY, "history"), default)

async def write_command_history(storage: UnifiedStorageService, value: Any) -> bool:
    """Write command history"""
    return await storage.write(StorageKey(StorageType.COMMAND_HISTORY, "history"), value)


# Direct typed accessors for services (eliminates need for storage adapters)
class UnifiedStorageServiceExtensions:
    """Extension methods for direct typed access - eliminates storage adapter layer"""
    
    @staticmethod
    async def read_marks_dict(storage: UnifiedStorageService) -> Dict[str, Tuple[int, int]]:
        """Read marks as dict mapping name -> (x, y) coordinates"""
        marks_data = await read_marks(storage=storage, default={})
        marks = {}
        for name, coords in marks_data.items():
            if isinstance(coords, dict) and "x" in coords and "y" in coords:
                marks[name] = (coords["x"], coords["y"])
            elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                marks[name] = (coords[0], coords[1])
        return marks
    
    @staticmethod
    async def write_marks_dict(storage: UnifiedStorageService, marks: Dict[str, Tuple[int, int]]) -> bool:
        """Write marks from dict format"""
        storage_data = {}
        for name, coords in marks.items():
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                storage_data[name] = {"x": coords[0], "y": coords[1]}
        return await write_marks(storage=storage, value=storage_data)
    
    @staticmethod
    async def get_all_mark_names(storage: UnifiedStorageService) -> Set[str]:
        """Get all mark names as a set"""
        marks = await UnifiedStorageServiceExtensions.read_marks_dict(storage)
        return set(marks.keys())
    
    @staticmethod
    async def get_custom_commands(storage: UnifiedStorageService) -> Dict[str, Any]:
        """Get custom commands"""
        data = await read_commands(storage, {})
        custom_commands_data = data.get('custom_commands', {})
        
        custom_commands = {}
        for phrase, command_dict in custom_commands_data.items():
            if isinstance(command_dict, dict):
                try:
                    command_obj = AutomationCommand(**command_dict)
                    custom_commands[phrase] = command_obj
                except Exception:
                    continue
        return custom_commands
    
    @staticmethod
    async def save_custom_commands(storage: UnifiedStorageService, commands_data: Dict[str, Any]) -> bool:
        """Save custom commands"""
        serializable_commands = {}
        for phrase, command_data in commands_data.items():
            if hasattr(command_data, 'model_dump'):
                serializable_commands[phrase] = command_data.model_dump()
            elif hasattr(command_data, 'dict'):
                serializable_commands[phrase] = command_data.dict()
            elif isinstance(command_data, dict):
                serializable_commands[phrase] = command_data
        
        current_data = await read_commands(storage, {})
        current_data['custom_commands'] = serializable_commands
        return await write_commands(storage, current_data)
    
    @staticmethod
    async def get_action_map(storage: UnifiedStorageService) -> Dict[str, Any]:
        """Get action map for command lookup"""
        action_map = {}
        
        # Load custom commands
        custom_commands = await UnifiedStorageServiceExtensions.get_custom_commands(storage)
        for normalized_phrase, command_data in custom_commands.items():
            action_map[normalized_phrase] = command_data
        
        # Load default commands
        default_commands = AutomationCommandRegistry.get_default_commands()
        
        for command_data in default_commands:
            normalized_phrase = command_data.command_key.lower().strip()
            
            if normalized_phrase not in action_map:
                # Apply any phrase overrides
                data = await read_commands(storage, {})
                phrase_overrides = data.get('phrase_overrides', {})
                effective_phrase = phrase_overrides.get(command_data.command_key, command_data.command_key)
                
                if effective_phrase != command_data.command_key:
                    command_data = AutomationCommand(
                        command_key=effective_phrase,
                        action_type=command_data.action_type,
                        action_value=command_data.action_value,
                        short_description=command_data.short_description,
                        long_description=command_data.long_description,
                        is_custom=command_data.is_custom
                    )
                    normalized_phrase = effective_phrase.lower().strip()
                
                action_map[normalized_phrase] = command_data
        
        return action_map
    
    @staticmethod
    async def get_agentic_prompts(storage: UnifiedStorageService) -> Dict[str, Any]:
        """Get agentic prompts data"""
        return await read_agentic_prompts(storage=storage, default={"prompts": [], "current_prompt_id": None})
    
    @staticmethod
    async def save_agentic_prompts(storage: UnifiedStorageService, data: Dict[str, Any]) -> bool:
        """Save agentic prompts data"""
        return await write_agentic_prompts(storage=storage, value=data) 