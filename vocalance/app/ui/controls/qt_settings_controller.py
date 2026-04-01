import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig, get_whitelisted_llm_model
from vocalance.app.event_bus import EventBus
from vocalance.app.events.settings_events import SettingChangedEvent, SettingsResetEvent, SettingsUpdatedEvent
from vocalance.app.services.storage.llm_model_downloader import LLMModelDownloader
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
    llm_download_progress = Signal(str)
    llm_cancellable_download_finished = Signal(bool, str)

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
        self._fallback_llm_downloader: Optional[LLMModelDownloader] = None

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

    def _llm_downloader(self) -> LLMModelDownloader:
        dictation = getattr(self.main_window, "_dictation_service", None)
        llm = getattr(dictation, "llm_service", None) if dictation else None
        if llm is not None:
            return llm.model_downloader
        if self._fallback_llm_downloader is None:
            self._fallback_llm_downloader = LLMModelDownloader(self.config)
        return self._fallback_llm_downloader

    def llm_bundle_on_disk(self, model_id: str) -> bool:
        spec = get_whitelisted_llm_model(model_id)
        if not spec:
            return False
        return self._llm_downloader().model_bundle_complete(spec.gguf_filenames)

    def schedule_llm_cancellable_download(self, model_id: str, cancel_event: threading.Event) -> None:
        async def _run() -> Tuple[bool, str]:
            def _progress(msg: str) -> None:
                self.llm_download_progress.emit(msg)

            dictation = getattr(self.main_window, "_dictation_service", None)
            llm = getattr(dictation, "llm_service", None) if dictation else None
            if not llm:
                return False, "Dictation service is not available yet."
            return await llm.download_whitelisted_model_cancellable(model_id, cancel_event, _progress)

        fut = asyncio.run_coroutine_threadsafe(_run(), self.event_loop)

        def _done(f: Future) -> None:
            try:
                ok, msg = f.result()
            except Exception as e:
                ok, msg = False, str(e)
            self.logger.info(
                "LLM cancellable download finished model_id=%s ok=%s msg=%s",
                model_id,
                ok,
                (msg[:120] + "…") if len(msg) > 120 else msg,
            )
            self.llm_cancellable_download_finished.emit(ok, msg)

        fut.add_done_callback(_done)

    async def update_settings_async(self, settings: Dict[str, Any]) -> Tuple[bool, str]:
        """Update multiple settings asynchronously."""
        try:
            success = await self.settings_service.update_multiple_settings(settings)
            if success:
                # Update cache with new values
                for key, value in settings.items():
                    parts = key.split(".")
                    if len(parts) == 2:
                        category, setting_key = parts
                        if category not in self._cached_settings:
                            self._cached_settings[category] = {}
                        self._cached_settings[category][setting_key] = value

                self.logger.info(f"Multiple settings updated: {len(settings)} items")
                self.all_settings_changed.emit(self._cached_settings)
                return True, "Settings updated successfully"
            else:
                message = "Failed to update settings"
                self.operation_error.emit(message)
                return False, message
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

    async def reset_section_settings_async(self, setting_keys: list) -> Tuple[bool, str]:
        """Reset specific settings to defaults asynchronously."""
        try:
            # Reset each setting in the section
            for setting_key in setting_keys:
                success = await self.settings_service.reset_setting(setting_key)
                if not success:
                    self.logger.warning(f"Failed to reset setting: {setting_key}")

            # Reload settings to get updated values
            await self.load_settings_async()
            self.logger.info(f"Section settings reset: {len(setting_keys)} settings")
            return True, "Section reset successfully"
        except Exception as e:
            self.logger.error(f"Error resetting section settings: {e}", exc_info=True)
            self.operation_error.emit(str(e))
            return False, str(e)

    def reset_section_settings(self, setting_keys: list) -> None:
        """Reset specific settings to defaults."""
        asyncio.run_coroutine_threadsafe(self.reset_section_settings_async(setting_keys), self.event_loop)

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
