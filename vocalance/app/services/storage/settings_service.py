"""
Simplified Settings Service

Handles settings override flow:
- Loads defaults from configuration
- Applies user overrides that persist between sessions
- Provides simple API for getting and setting values
- Centralizes all settings operations through unified storage
"""

import logging
from typing import Any, Dict, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import DynamicSettingsUpdatedEvent, SettingsResponseEvent
from vocalance.app.services.storage.settings_update_coordinator import SettingsUpdateCoordinator
from vocalance.app.services.storage.storage_models import SettingsData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class SettingsService:
    """
    Settings service for configuration overrides with real-time updates.

    This service:
    1. Loads/saves user overrides from storage
    2. Builds effective settings (defaults + overrides)
    3. Publishes updates to SettingsUpdateCoordinator for real-time propagation

    All settings updates flow through the coordinator to ensure consistency.
    """

    # Define which settings can be overridden by users
    OVERRIDEABLE_SETTINGS = {
        "llm.context_length",
        "llm.max_tokens",
        "grid.default_rect_count",
        "sound_recognizer.confidence_threshold",
        "sound_recognizer.vote_threshold",
        "vad.energy_threshold",
        "audio.device",
        "markov_predictor.confidence_threshold",
    }

    # Settings that update in real-time (default is restart required)
    REAL_TIME_SETTINGS = {
        "markov_predictor.confidence_threshold",
        "sound_recognizer.confidence_threshold",
        "sound_recognizer.vote_threshold",
        "grid.default_rect_count",
    }

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        storage: StorageService,
        coordinator: Optional[SettingsUpdateCoordinator] = None,
    ):
        self._event_bus = event_bus
        self._config = config
        self._storage = storage
        self._coordinator = coordinator

        # Cache for performance
        self._user_overrides: Dict[str, Any] = {}
        self._effective_settings: Dict[str, Any] = {}

        logger.debug("SettingsService initialized")

    def setup_subscriptions(self) -> None:
        """Setup event subscriptions for settings updates."""
        logger.debug("SettingsService subscriptions configured")

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
                "llm": {"context_length": self._config.llm.context_length, "max_tokens": self._config.llm.max_tokens},
                "grid": {"default_rect_count": self._config.grid.default_rect_count},
                "sound_recognizer": {
                    "confidence_threshold": self._config.sound_recognizer.confidence_threshold,
                    "vote_threshold": self._config.sound_recognizer.vote_threshold,
                },
                "vad": {"energy_threshold": self._config.vad.energy_threshold},
                "audio": {"device": self._config.audio.device, "sample_rate": self._config.audio.sample_rate},
                "markov_predictor": {"confidence_threshold": self._config.markov_predictor.confidence_threshold},
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

            keys = setting_path.split(".")
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
                category, key = setting_path.split(".", 1)

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

                # Publish dynamic settings update event only for real-time settings
                # The coordinator will handle propagating to services
                real_time_updates = {k: v for k, v in settings_updates.items() if k in self.REAL_TIME_SETTINGS}
                if real_time_updates:
                    await self._publish_dynamic_settings_update(settings_updates=real_time_updates)

                # Log whether restart is required (default unless in REAL_TIME_SETTINGS)
                all_real_time = all(setting in self.REAL_TIME_SETTINGS for setting in settings_updates.keys())
                if all_real_time:
                    logger.info(f"Updated {len(settings_updates)} settings (applied in real-time)")
                else:
                    logger.info(f"Updated {len(settings_updates)} settings (restart required for changes to take effect)")

                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update multiple settings: {e}")
            return False

    async def reset_setting(self, setting_path: str) -> bool:
        """Reset a setting to its default value (delegates to update_multiple_settings)"""
        if setting_path not in self.OVERRIDEABLE_SETTINGS:
            return False

        try:
            category, key = setting_path.split(".", 1)

            # Remove from overrides
            if category in self._user_overrides and key in self._user_overrides[category]:
                del self._user_overrides[category][key]
                if not self._user_overrides[category]:
                    del self._user_overrides[category]

            # Save and publish updates
            settings_data = SettingsData(user_overrides=self._user_overrides)
            success = await self._storage.write(data=settings_data)

            if success:
                await self._build_effective_settings()
                await self._publish_settings_response()

                # Get default value and publish update only for real-time settings
                default_value = self._get_default_value(setting_path=setting_path)
                if setting_path in self.REAL_TIME_SETTINGS:
                    await self._publish_dynamic_settings_update(settings_updates={setting_path: default_value})
                    logger.info(f"Reset setting to default (applied in real-time): {setting_path}")
                else:
                    logger.info(f"Reset setting to default (restart required): {setting_path}")

                return True

            return False

        except Exception as e:
            logger.error(f"Failed to reset setting {setting_path}: {e}")
            return False

    def _validate_setting_value(self, setting_path: str, value: Any) -> bool:
        """Validate setting value based on setting type and constraints"""
        validation_rules = {
            "llm.context_length": lambda v: isinstance(v, int) and 128 <= v <= 32768,
            "llm.max_tokens": lambda v: isinstance(v, int) and 1 <= v <= 4096,
            "grid.default_rect_count": lambda v: isinstance(v, int) and v > 0,
            "sound_recognizer.confidence_threshold": lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
            "sound_recognizer.vote_threshold": lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
            "markov_predictor.confidence_threshold": lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
            "vad.energy_threshold": lambda v: isinstance(v, (int, float)) and v >= 0,
            "audio.device": lambda v: v is None or isinstance(v, int),
        }

        try:
            validator = validation_rules.get(setting_path)
            return validator(value) if validator else True
        except Exception as e:
            logger.error(f"Validation error for {setting_path}: {e}")
            return False

    def _get_default_value(self, setting_path: str) -> Any:
        """Get default value for a setting from config"""
        category, key = setting_path.split(".", 1)

        category_map = {
            "llm": self._config.llm,
            "grid": self._config.grid,
            "sound_recognizer": self._config.sound_recognizer,
            "markov_predictor": self._config.markov_predictor,
            "vad": self._config.vad,
            "audio": self._config.audio,
        }

        config_obj = category_map.get(category)
        return getattr(config_obj, key) if config_obj else None

    async def _publish_settings_response(self) -> None:
        """Publish current settings for UI and services"""
        try:
            settings = await self.get_effective_settings()
            event = SettingsResponseEvent(settings=settings)
            await self._event_bus.publish(event)
        except Exception as e:
            logger.error(f"Failed to publish settings response: {e}")

    async def _publish_dynamic_settings_update(self, settings_updates: Dict[str, Any]) -> None:
        """
        Publish dynamic settings update event.
        The SettingsUpdateCoordinator subscribes to this and handles propagation.
        """
        try:
            event = DynamicSettingsUpdatedEvent(updated_settings=settings_updates)
            await self._event_bus.publish(event)
            logger.debug(f"Published DynamicSettingsUpdatedEvent for: {list(settings_updates.keys())}")
        except Exception as e:
            logger.error(f"Failed to publish dynamic settings update: {e}")

    async def apply_startup_settings_to_config(self) -> None:
        """
        Apply user overrides to config at startup.
        This publishes an update event that the coordinator handles.
        """
        try:
            settings_to_apply = {}

            for category, overrides in self._user_overrides.items():
                for key, value in overrides.items():
                    setting_path = f"{category}.{key}"
                    settings_to_apply[setting_path] = value

            if settings_to_apply:
                await self._publish_dynamic_settings_update(settings_updates=settings_to_apply)
                logger.info(f"Applied {len(settings_to_apply)} startup settings via coordinator")

        except Exception as e:
            logger.error(f"Failed to apply startup settings: {e}")
