import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig, get_whitelisted_llm_model
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import SettingsChangedEvent
from vocalance.app.services.storage.llm_model_downloader import LLMModelDownloader
from vocalance.app.services.storage.settings_service import SettingsService
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtSettingsController(QtBaseController):
    """Controller for application settings management."""

    settings_loaded = Signal(dict)
    setting_changed = Signal(str, object)
    all_settings_changed = Signal(dict)
    settings_reset = Signal()
    operation_error = Signal(str)
    llm_download_progress = Signal(str)
    llm_cancellable_download_finished = Signal(bool, str)

    def __init__(
        self,
        event_bus: EventBus,
        settings_service: SettingsService,
        config: GlobalAppConfig,
        main_window: Any,
    ) -> None:
        """Initialize settings controller.

        Args:
            event_bus: Event bus for pub/sub.
            settings_service: Settings service instance.
            config: Global app configuration.
            main_window: Main window reference (used to access the dictation service for LLM downloads).
        """
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtSettingsController"),
        )

        self.settings_service = settings_service
        self.config = config
        self.main_window = main_window
        self._cached_settings: Dict[str, Any] = {}
        self._fallback_llm_downloader: Optional[LLMModelDownloader] = None

        self._subscribe_to_events()
        self.logger.debug("QtSettingsController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to settings events."""
        self.event_bus.subscribe(
            event_type=SettingsChangedEvent,
            handler=self._handle_settings_changed,
        )

    def _handle_settings_changed(self, settings_change: SettingsChangedEvent) -> None:
        """Handle settings changed event."""
        try:
            self._cached_settings = settings_change.all_settings
            self.all_settings_changed.emit(self._cached_settings)
            for key, value in settings_change.updated_settings.items():
                self.setting_changed.emit(key, value)
        except Exception as e:
            self.logger.error(f"Error handling settings changed: {e}", exc_info=True)

    def load_settings_async(self) -> Dict[str, Any]:
        """Load all settings asynchronously and emit them to the view.

        Returns:
            Dict of current effective settings, or empty dict on failure.
        """
        try:
            settings = self.settings_service.get_effective_settings()
            self._cached_settings = settings
            self.settings_loaded.emit(settings)
            return settings
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}", exc_info=True)
            self.operation_error.emit(f"Failed to load settings: {e}")
            return {}

    def load_settings(self) -> None:
        """Load settings."""
        self.load_settings_async()

    async def update_setting_async(self, key: str, value: Any) -> Tuple[bool, str]:
        """Update a single setting asynchronously.

        Args:
            key: Dot-separated setting path (e.g. 'audio.sample_rate').
            value: New value for the setting.

        Returns:
            Tuple of (success, message).
        """
        try:
            success = await self.settings_service.update_multiple_settings({key: value})
            if success:
                parts = key.split(".")
                if len(parts) == 2:
                    category, setting_key = parts
                    self._cached_settings.setdefault(category, {})[setting_key] = value
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
        """Schedule an async update for a single setting.

        Args:
            key: Dot-separated setting path.
            value: New value for the setting.
        """
        asyncio.create_task(self.update_setting_async(key, value))

    def _llm_downloader(self) -> LLMModelDownloader:
        """Return the LLM downloader, preferring the one owned by the dictation service."""
        dictation = getattr(self.main_window, "_dictation_service", None)
        llm = getattr(dictation, "llm_service", None) if dictation else None
        if llm is not None:
            return llm.model_downloader
        if self._fallback_llm_downloader is None:
            self._fallback_llm_downloader = LLMModelDownloader(self.config)
        return self._fallback_llm_downloader

    def llm_bundle_on_disk(self, model_id: str) -> bool:
        """Return True if the model bundle for the given ID is fully downloaded.

        Args:
            model_id: Whitelisted LLM model identifier.
        """
        spec = get_whitelisted_llm_model(model_id)
        if not spec:
            return False
        return self._llm_downloader().model_bundle_complete(spec.gguf_filenames)

    def schedule_llm_cancellable_download(self, model_id: str, cancel_event: threading.Event) -> None:
        """Schedule a cancellable LLM model download and emit progress/completion signals.

        Args:
            model_id: Whitelisted LLM model identifier to download.
            cancel_event: Threading event that signals cancellation when set.
        """

        async def _run() -> Tuple[bool, str]:
            def _progress(msg: str) -> None:
                self.llm_download_progress.emit(msg)

            dictation = getattr(self.main_window, "_dictation_service", None)
            llm = getattr(dictation, "llm_service", None) if dictation else None
            if not llm:
                return False, "Dictation service is not available yet."
            return await llm.download_whitelisted_model_cancellable(model_id, cancel_event, _progress)

        fut: Future = asyncio.create_task(_run())

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
        """Update multiple settings asynchronously.

        Args:
            settings: Dict of dot-separated setting paths to new values.

        Returns:
            Tuple of (success, message).
        """
        try:
            success = await self.settings_service.update_multiple_settings(settings)
            if success:
                for key, value in settings.items():
                    parts = key.split(".")
                    if len(parts) == 2:
                        category, setting_key = parts
                        self._cached_settings.setdefault(category, {})[setting_key] = value
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
        """Schedule an async update for multiple settings.

        Args:
            settings: Dict of dot-separated setting paths to new values.
        """
        asyncio.create_task(self.update_settings_async(settings))

    async def _reset_to_defaults_async(self) -> None:
        try:
            success, message = await self.settings_service.reset_to_defaults()
            if success:
                self.load_settings_async()
            else:
                self.operation_error.emit(message)
        except Exception as e:
            self.logger.error("Error resetting settings: %s", e, exc_info=True)
            self.operation_error.emit(str(e))

    def reset_to_defaults(self) -> None:
        asyncio.create_task(self._reset_to_defaults_async())

    async def reset_section_settings_async(self, setting_keys: list) -> Tuple[bool, str]:
        """Reset specific settings to defaults asynchronously.

        Args:
            setting_keys: List of dot-separated setting paths to reset.

        Returns:
            Tuple of (success, message).
        """
        try:
            for setting_key in setting_keys:
                success = await self.settings_service.reset_setting(setting_key)
                if not success:
                    self.logger.warning(f"Failed to reset setting: {setting_key}")
            self.load_settings_async()
            return True, "Section reset successfully"
        except Exception as e:
            self.logger.error(f"Error resetting section settings: {e}", exc_info=True)
            self.operation_error.emit(str(e))
            return False, str(e)

    def reset_section_settings(self, setting_keys: list) -> None:
        """Schedule an async reset for specific settings.

        Args:
            setting_keys: List of dot-separated setting paths to reset.
        """
        asyncio.create_task(self.reset_section_settings_async(setting_keys))

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Return a cached setting value by key.

        Args:
            key: Setting key to look up.
            default: Value to return if the key is not cached.
        """
        return self._cached_settings.get(key, default)

    def get_all_settings(self) -> Dict[str, Any]:
        """Return a copy of all cached settings."""
        return dict(self._cached_settings)

    def cleanup(self) -> None:
        """Unsubscribe from all events and release resources."""
        super().cleanup()
