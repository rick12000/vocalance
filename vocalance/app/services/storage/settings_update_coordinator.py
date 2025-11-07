import logging
from typing import Any, Dict

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import DynamicSettingsUpdatedEvent

logger = logging.getLogger(__name__)


class SettingsUpdateCoordinator:
    """
    Coordinates real-time settings updates across services.

    This coordinator handles settings that can be updated without restart:
    - markov_predictor.confidence_threshold
    - sound_recognizer.confidence_threshold
    - sound_recognizer.vote_threshold
    - grid.default_rect_count
    - vad.dictation_silent_chunks_for_end
    - vad.command_silent_chunks_for_end

    Other settings (LLM, audio device) require app restart.

    Flow:
    1. The GlobalAppConfig is updated (single source of truth)
    2. Real-time settings are propagated to registered services
    3. Changes are logged for debugging
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig):
        self._event_bus = event_bus
        self._config = config
        self._service_registry: Dict[str, Any] = {}

        logger.debug("SettingsUpdateCoordinator initialized")

    def setup_subscriptions(self) -> None:
        """Subscribe to settings update events"""
        self._event_bus.subscribe(event_type=DynamicSettingsUpdatedEvent, handler=self._handle_settings_updated)
        logger.debug("SettingsUpdateCoordinator subscriptions configured")

    def register_service(self, service_name: str, service_instance: Any) -> None:
        """
        Register a service that needs real-time settings updates.

        Services must implement specific update methods for their settings:
        - MarkovCommandService: update_confidence_threshold(float)
        - SoundRecognizer: update_confidence_threshold(float), update_vote_threshold(float)
        """
        self._service_registry[service_name] = service_instance
        logger.debug(f"Registered service for settings updates: {service_name}")

    async def _handle_settings_updated(self, event: DynamicSettingsUpdatedEvent) -> None:
        """Handle settings update event and coordinate updates across services"""
        try:
            updated_settings = event.updated_settings

            # Update the GlobalAppConfig first (single source of truth)
            self._update_config(updated_settings=updated_settings)

            # Then propagate to registered services
            await self._propagate_to_services(updated_settings=updated_settings)

        except Exception as e:
            logger.error(f"Error coordinating settings update: {e}", exc_info=True)

    def _update_config(self, updated_settings: Dict[str, Any]) -> None:
        """Update GlobalAppConfig with new values"""
        category_map = {
            "markov_predictor": self._config.markov_predictor,
            "sound_recognizer": self._config.sound_recognizer,
            "llm": self._config.llm,
            "grid": self._config.grid,
            "vad": self._config.vad,
            "audio": self._config.audio,
        }

        for setting_path, value in updated_settings.items():
            try:
                category, key = setting_path.split(".", 1)
                config_obj = category_map.get(category)

                if config_obj and hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                else:
                    logger.warning(f"Unknown setting path: {setting_path}")

            except Exception as e:
                logger.error(f"Error updating config for {setting_path}: {e}")

    async def _propagate_to_services(self, updated_settings: Dict[str, Any]) -> None:
        """Propagate real-time settings to registered services"""
        propagation_map = {
            "markov_predictor.confidence_threshold": ("markov_predictor", "on_confidence_threshold_updated"),
            "sound_recognizer.confidence_threshold": ("sound_recognizer", "on_confidence_threshold_updated"),
            "sound_recognizer.vote_threshold": ("sound_recognizer", "on_vote_threshold_updated"),
            "vad.dictation_silent_chunks_for_end": ("audio", "on_dictation_silent_chunks_updated"),
            "vad.command_silent_chunks_for_end": ("audio", "on_command_silent_chunks_updated"),
        }

        for setting_path, value in updated_settings.items():
            # GridService reads directly from config - no callback needed
            if setting_path == "grid.default_rect_count":
                continue

            service_info = propagation_map.get(setting_path)
            if service_info:
                service_name, method_name = service_info
                service = self._service_registry.get(service_name)

                if service and hasattr(service, method_name):
                    method = getattr(service, method_name)
                    if "threshold" in method_name:
                        await method(threshold=value)
                    elif "count" in method_name:
                        await method(count=value)
                    elif "chunks" in method_name:
                        await method(chunks=value)
                    else:
                        await method(value)
                else:
                    logger.warning(f"Service '{service_name}' not registered or method '{method_name}' not found")
            else:
                logger.warning(f"No propagation mapping found for real-time setting: {setting_path}")
