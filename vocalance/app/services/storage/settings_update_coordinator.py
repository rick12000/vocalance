"""Applies ``SettingsChangedEvent`` payload to ``GlobalAppConfig`` and forwards
per-setting deltas to registered callbacks.

Callbacks are registered directly (typed callables) rather than by string
service/method names, which eliminates the fragile string-based dispatch.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import SettingsChangedEvent
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)

# Settings that live only in config and need no service callback.
_CONFIG_ONLY_PATHS = frozenset(
    {
        "llm.selected_model_id",
        "llm.context_length",
        "llm.max_tokens",
        "grid.default_rect_count",
    }
)


class SettingsUpdateCoordinator(Service):
    """Apply ``SettingsChangedEvent`` to ``GlobalAppConfig`` and notify services.

    Services register typed callbacks for the specific setting paths they care about.

    The callback receives the new value as its sole argument.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        self._event_bus = event_bus
        self._config = config
        self._callbacks: Dict[str, list[Callable]] = {}

        self._category_map = {
            "sound_recognizer": self._config.sound_recognizer,
            "llm": self._config.llm,
            "grid": self._config.grid,
            "vad": self._config.vad,
            "audio": self._config.audio,
        }

        event_bus.subscribe(SettingsChangedEvent, self._handle_settings_updated)

    def register_callback(self, setting_path: str, callback: Callable[[Any], Any]) -> None:
        """Register a callback to be invoked when ``setting_path`` changes.

        The callback receives the new value as its single argument.  Both sync
        and async callables are supported.
        """
        self._callbacks.setdefault(setting_path, []).append(callback)

    async def _handle_settings_updated(self, event: SettingsChangedEvent) -> None:
        try:
            self._apply_to_config(event.updated_settings)
            await self._dispatch_callbacks(event.updated_settings)
        except Exception as e:
            logger.error("Error coordinating settings update: %s", e, exc_info=True)

    def _apply_to_config(self, updated_settings: Dict[str, Any]) -> None:
        for path, value in updated_settings.items():
            try:
                category, key = path.split(".", 1)
                config_obj = self._category_map.get(category)
                if config_obj and hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                else:
                    logger.warning("Unknown setting path: %s", path)
            except Exception as e:
                logger.error("Error updating config for %s: %s", path, e)

    async def _dispatch_callbacks(self, updated_settings: Dict[str, Any]) -> None:
        for path, value in updated_settings.items():
            if path in _CONFIG_ONLY_PATHS:
                continue
            for cb in self._callbacks.get(path, []):
                try:
                    if inspect.iscoroutinefunction(cb):
                        await cb(value)
                    else:
                        cb(value)
                except Exception as e:
                    logger.error("Callback error for setting %s: %s", path, e, exc_info=True)

    async def shutdown(self) -> None:
        self._event_bus.unsubscribe(SettingsChangedEvent, self._handle_settings_updated)
        self._callbacks.clear()
