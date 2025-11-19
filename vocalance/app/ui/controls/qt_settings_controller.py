"""Qt-based settings controller.

Manages application settings, configuration, and persistence.
"""

import asyncio
import logging
from typing import Any, Dict, Tuple

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.settings_events import SettingChangedEvent, SettingsResetEvent, SettingsUpdatedEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtSettingsController(QtBaseController):
    """Controller for application settings management.

    Handles:
    - Loading settings
    - Saving settings
    - Resetting to defaults
    - Settings change notifications
    - Full backend event subscription and handling
    """

    # Signals for settings operations
    settings_loaded = Signal(dict)  # Settings dict
    setting_changed = Signal(str, object)  # key, value
    all_settings_changed = Signal(dict)  # All settings
    settings_reset = Signal()
    operation_error = Signal(str)

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        settings_service,
        config: GlobalAppConfig,
        main_window,
    ):
        """Initialize settings controller.

        Args:
            event_bus: Event bus for pub/sub.
            event_loop: Asyncio event loop.
            settings_service: Settings service instance.
            config: Global app configuration.
            main_window: Main window reference.
        """
        super().__init__(
            event_bus=event_bus,
            event_loop=event_loop,
            logger=logging.getLogger("QtSettingsController"),
        )

        self.settings_service = settings_service
        self.config = config
        self.main_window = main_window
        self._cached_settings: Dict[str, Any] = {}

        # Subscribe to settings events
        self._subscribe_to_events()

        self.logger.debug("QtSettingsController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to settings-related events."""
        try:
            self.event_bus.subscribe(SettingsUpdatedEvent, self._handle_settings_updated)
            self.event_bus.subscribe(SettingChangedEvent, self._handle_setting_changed)
            self.event_bus.subscribe(SettingsResetEvent, self._handle_settings_reset)
            self.logger.debug("Subscribed to settings events")
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def _handle_settings_updated(self, event) -> None:
        """Handle settings updated event."""
        try:
            settings = getattr(event, "settings", {})
            self._cached_settings = settings
            self.logger.debug("Settings updated")
            self.all_settings_changed.emit(settings)
        except Exception as e:
            self.logger.error(f"Error handling settings updated: {e}", exc_info=True)

    def _handle_setting_changed(self, event) -> None:
        """Handle individual setting changed event."""
        try:
            key = getattr(event, "key", "")
            value = getattr(event, "value", None)
            self._cached_settings[key] = value
            self.logger.debug(f"Setting changed: {key} = {value}")
            self.setting_changed.emit(key, value)
        except Exception as e:
            self.logger.error(f"Error handling setting changed: {e}", exc_info=True)

    def _handle_settings_reset(self, event) -> None:
        """Handle settings reset event."""
        try:
            self.logger.info("Settings reset to defaults")
            self.settings_reset.emit()
            # Reload settings
            asyncio.run_coroutine_threadsafe(self.load_settings_async(), self.event_loop)
        except Exception as e:
            self.logger.error(f"Error handling settings reset: {e}", exc_info=True)

    async def load_settings_async(self) -> Dict[str, Any]:
        """Load all settings asynchronously."""
        try:
            settings = await self.settings_service.get_effective_settings()
            self._cached_settings = settings
            self.settings_loaded.emit(settings)
            return settings
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}", exc_info=True)
            self.operation_error.emit(f"Failed to load settings: {e}")
            return {}

    def load_settings(self) -> None:
        """Load all settings (creates async task)."""
        asyncio.run_coroutine_threadsafe(self.load_settings_async(), self.event_loop)

    async def update_setting_async(self, key: str, value: Any) -> Tuple[bool, str]:
        """Update a single setting asynchronously."""
        try:
            success = await self.settings_service.update_multiple_settings({key: value})
            if success:
                # Update cache
                parts = key.split(".")
                if len(parts) == 2:
                    category, setting_key = parts
                    if category not in self._cached_settings:
                        self._cached_settings[category] = {}
                    self._cached_settings[category][setting_key] = value
                self.logger.info(f"Setting updated: {key} = {value}")
                return True, f"Setting updated: {key}"
            else:
                message = f"Failed to update setting: {key}"
                self.operation_error.emit(message)
                return False, message
        except Exception as e:
            self.logger.error(f"Error updating setting: {e}", exc_info=True)
            self.operation_error.emit(str(e))
            return False, str(e)

    def update_setting(self, key: str, value: Any) -> None:
        """Update a single setting."""
        asyncio.run_coroutine_threadsafe(self.update_setting_async(key, value), self.event_loop)

    async def update_settings_async(self, settings: Dict[str, Any]) -> Tuple[bool, str]:
        """Update multiple settings asynchronously."""
        try:
            success, message = await asyncio.to_thread(self.settings_service.update_settings, settings)
            if success:
                self._cached_settings.update(settings)
                self.logger.info(f"Multiple settings updated: {len(settings)} items")
                self.all_settings_changed.emit(self._cached_settings)
            else:
                self.operation_error.emit(message)
            return success, message
        except Exception as e:
            self.logger.error(f"Error updating settings: {e}", exc_info=True)
            self.operation_error.emit(str(e))
            return False, str(e)

    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update multiple settings."""
        asyncio.run_coroutine_threadsafe(self.update_settings_async(settings), self.event_loop)

    async def reset_to_defaults_async(self) -> Tuple[bool, str]:
        """Reset all settings to defaults asynchronously."""
        try:
            success, message = await asyncio.to_thread(self.settings_service.reset_to_defaults)
            if success:
                self.logger.info("Settings reset to defaults")
                await self.load_settings_async()
            else:
                self.operation_error.emit(message)
            return success, message
        except Exception as e:
            self.logger.error(f"Error resetting settings: {e}", exc_info=True)
            self.operation_error.emit(str(e))
            return False, str(e)

    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        asyncio.run_coroutine_threadsafe(self.reset_to_defaults_async(), self.event_loop)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value from cache."""
        return self._cached_settings.get(key, default)

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all cached settings."""
        return dict(self._cached_settings)

    def cleanup(self) -> None:
        """Clean up controller resources."""
        try:
            self.event_bus.unsubscribe(SettingsUpdatedEvent, self._handle_settings_updated)
            self.event_bus.unsubscribe(SettingChangedEvent, self._handle_setting_changed)
            self.event_bus.unsubscribe(SettingsResetEvent, self._handle_settings_reset)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
