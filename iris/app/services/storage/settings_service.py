"""
Simplified Settings Service

Handles settings override flow:
- Loads defaults from configuration
- Applies user overrides that persist between sessions
- Provides simple API for getting and setting values
- Centralizes all settings operations through unified storage
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from iris.app.config.app_config import GlobalAppConfig
from iris.app.event_bus import EventBus
from iris.app.events.core_events import SettingsResponseEvent
from iris.app.services.storage.storage_service import StorageService
from iris.app.services.storage.storage_models import SettingsData

logger = logging.getLogger(__name__)


class SettingsService:
    """
    Simplified settings service for configuration overrides
    
    Features:
    - Configuration defaults with user overrides
    - Persistent storage through unified storage
    - Simple API for getting/setting values
    """
    
    # Define which settings can be overridden by users
    OVERRIDEABLE_SETTINGS = {
        'llm.context_length', 
        'llm.max_tokens',
        'grid.default_rect_count',
        'sound_recognizer.confidence_threshold',
        'vad.energy_threshold',
        'audio.device'
    }
    
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService):
        self._event_bus = event_bus
        self._config = config
        self._storage = storage
        
        # Cache for performance
        self._user_overrides: Dict[str, Any] = {}
        self._effective_settings: Dict[str, Any] = {}
        
        logger.info("SettingsService initialized")
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions - simplified"""
        # No event subscriptions needed for simplified version
        logger.info("SettingsService subscriptions configured")
    
    async def initialize(self) -> bool:
        """Initialize service and load settings"""
        try:
            await self._load_user_overrides()
            await self._build_effective_settings()
            await self._publish_settings_response()
            
            logger.info("SettingsService initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Settings initialization error: {e}", exc_info=True)
            return False
    
    async def _load_user_overrides(self) -> None:
        """Load user setting overrides from storage"""
        try:
            settings_data = await self._storage.read(model_type=SettingsData)
            self._user_overrides = settings_data.user_overrides
            logger.debug(f"Loaded {len(self._user_overrides)} user setting categories")
        except Exception as e:
            logger.error(f"Failed to load user overrides: {e}")
            self._user_overrides = {}
    
    async def _build_effective_settings(self) -> None:
        """Build effective settings by applying overrides to defaults"""
        try:
            # Start with config defaults
            self._effective_settings = {
                'llm': {
                    'context_length': self._config.llm.context_length,
                    'max_tokens': self._config.llm.max_tokens
                },
                'grid': {
                    'default_rect_count': self._config.grid.default_rect_count
                },
                'sound_recognizer': {
                    'confidence_threshold': self._config.sound_recognizer.confidence_threshold
                },
                'vad': {
                    'energy_threshold': self._config.vad.energy_threshold
                },
                'audio': {
                    'device': self._config.audio.device,
                    'sample_rate': self._config.audio.sample_rate
                }
            }
            
            # Apply user overrides
            self._apply_overrides_to_effective_settings()
            
            logger.debug("Built effective settings with user overrides")
            
        except Exception as e:
            logger.error(f"Failed to build effective settings: {e}")
            self._effective_settings = {}
    
    def _apply_overrides_to_effective_settings(self) -> None:
        """Apply user overrides to effective settings"""
        for category, category_overrides in self._user_overrides.items():
            if category in self._effective_settings:
                for key, value in category_overrides.items():
                    setting_path = f"{category}.{key}"
                    if setting_path in self.OVERRIDEABLE_SETTINGS:
                        self._effective_settings[category][key] = value
                        logger.debug(f"Applied override: {setting_path} = {value}")
    
    async def get_effective_settings(self) -> Dict[str, Any]:
        """Get current effective settings (defaults + overrides)"""
        return self._effective_settings.copy()
    
    async def get_setting(self, setting_path: str, default: Any = None) -> Any:
        """Get a single setting value using dot notation"""
        try:
            settings = await self.get_effective_settings()
            
            keys = setting_path.split('.')
            current = settings
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
        except Exception as e:
            logger.error(f"Failed to get setting {setting_path}: {e}")
            return default
    
    async def update_multiple_settings(self, settings_updates: Dict[str, Any]) -> bool:
        """Update multiple settings atomically"""
        try:
            # Validate all settings first
            for setting_path, value in settings_updates.items():
                if setting_path not in self.OVERRIDEABLE_SETTINGS:
                    logger.warning(f"Setting {setting_path} is not overrideable")
                    return False
                
                if not self._validate_setting_value(setting_path, value):
                    logger.warning(f"Invalid value for {setting_path}: {value}")
                    return False
            
            # Apply all overrides
            for setting_path, value in settings_updates.items():
                category, key = setting_path.split('.', 1)
                
                if category not in self._user_overrides:
                    self._user_overrides[category] = {}
                
                self._user_overrides[category][key] = value
            
            # Save to storage
            settings_data = SettingsData(user_overrides=self._user_overrides)
            success = await self._storage.write(data=settings_data)
            
            if success:
                # Rebuild effective settings
                await self._build_effective_settings()
                await self._publish_settings_response()
                
                logger.info(f"Updated {len(settings_updates)} settings")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update multiple settings: {e}")
            return False
    
    async def reset_setting(self, setting_path: str) -> bool:
        """Reset a setting to its default value"""
        try:
            if setting_path not in self.OVERRIDEABLE_SETTINGS:
                return False
            
            category, key = setting_path.split('.', 1)
            
            # Remove from overrides
            if category in self._user_overrides and key in self._user_overrides[category]:
                del self._user_overrides[category][key]
                
                # Clean up empty categories
                if not self._user_overrides[category]:
                    del self._user_overrides[category]
            
            # Save to storage
            settings_data = SettingsData(user_overrides=self._user_overrides)
            success = await self._storage.write(data=settings_data)
            
            if success:
                # Rebuild effective settings
                await self._build_effective_settings()
                await self._publish_settings_response()
                
                logger.info(f"Reset setting to default: {setting_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to reset setting {setting_path}: {e}")
            return False
    
    def _validate_setting_value(self, setting_path: str, value: Any) -> bool:
        """Validate setting value based on setting type and constraints"""
        try:
            if setting_path == 'llm.context_length':
                return isinstance(value, int) and 128 <= value <= 32768
            elif setting_path == 'llm.max_tokens':
                return isinstance(value, int) and 1 <= value <= 4096
            elif setting_path == 'grid.default_rect_count':
                return isinstance(value, int) and value > 0
            elif setting_path == 'sound_recognizer.confidence_threshold':
                return isinstance(value, (int, float)) and 0.0 <= value <= 1.0
            elif setting_path == 'vad.energy_threshold':
                return isinstance(value, (int, float)) and value >= 0
            elif setting_path == 'audio.device':
                return value is None or isinstance(value, int)
            else:
                return True  # Unknown setting, allow it
                
        except Exception as e:
            logger.error(f"Validation error for {setting_path}: {e}")
            return False
    
    async def apply_settings_to_config(self, config: GlobalAppConfig) -> GlobalAppConfig:
        """Apply effective settings to a config object"""
        try:
            settings = await self.get_effective_settings()
            
            # Apply LLM settings
            if 'llm' in settings:
                llm = settings['llm']
                config.llm.context_length = llm.get('context_length', config.llm.context_length)
                config.llm.max_tokens = llm.get('max_tokens', config.llm.max_tokens)
            
            # Apply grid settings
            if 'grid' in settings:
                config.grid.default_rect_count = settings['grid'].get('default_rect_count', config.grid.default_rect_count)
            
            # Apply sound recognizer settings
            if 'sound_recognizer' in settings:
                config.sound_recognizer.confidence_threshold = settings['sound_recognizer'].get('confidence_threshold', config.sound_recognizer.confidence_threshold)
            
            # Apply VAD settings
            if 'vad' in settings:
                vad = settings['vad']
                config.vad.energy_threshold = vad.get('energy_threshold', config.vad.energy_threshold)
            
            # Apply audio settings
            if 'audio' in settings:
                config.audio.device = settings['audio'].get('device', config.audio.device)
            
            logger.info("Applied settings to configuration")
            return config
            
        except Exception as e:
            logger.error(f"Failed to apply settings to config: {e}")
            return config
    
    async def _publish_settings_response(self) -> None:
        """Publish current settings for UI and services"""
        try:
            settings = await self.get_effective_settings()
            event = SettingsResponseEvent(settings=settings)
            await self._event_bus.publish(event)
        except Exception as e:
            logger.error(f"Failed to publish settings response: {e}") 