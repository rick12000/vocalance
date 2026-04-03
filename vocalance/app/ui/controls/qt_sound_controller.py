import asyncio
import logging
from typing import Dict, List, Optional

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.automation_command_registry import AutomationCommandRegistry
from vocalance.app.event_bus import EventBus
from vocalance.app.events.mark_events import MarkGetAllRequestEventData, MarksChangedEventData
from vocalance.app.events.sound_events import (
    AllSoundsResetEvent,
    DeleteSoundCommand,
    MapSoundToCommandPhraseCommand,
    RequestSoundListEvent,
    RequestSoundMappingsEvent,
    ResetAllSoundsCommand,
    SoundDeletedEvent,
    SoundListUpdatedEvent,
    SoundMappingsResponseEvent,
    SoundToCommandMappingUpdatedEvent,
    SoundTrainingCompleteEvent,
    SoundTrainingFailedEvent,
    SoundTrainingInitiatedEvent,
    SoundTrainingProgressEvent,
    SoundTrainingRequestEvent,
    SoundTrainingStatusEvent,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtSoundController(QtBaseController):
    """Business logic controller for sound functionality."""

    sounds_loaded = Signal(list)
    training_started = Signal(str, int)
    training_progress = Signal(str, int, int)
    training_completed = Signal(str)
    training_error = Signal(str, str)
    training_status = Signal(str, str)
    sound_deleted = Signal(str)
    all_sounds_deleted = Signal()
    sound_mapping_updated = Signal(str, str)
    operation_error = Signal(str)

    def __init__(
        self,
        event_bus: EventBus,
        sound_service,
        storage_service,
        config: GlobalAppConfig,
    ):
        """Initialize sound controller.

        Args:
            event_bus: Event bus for pub/sub.
            sound_service: Sound service instance.
            storage_service: Storage service instance.
            config: Global app configuration.
        """
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtSoundController"),
        )

        self.sound_service = sound_service
        self.storage_service = storage_service
        self.config = config

        self.available_sounds = []
        self._sound_mappings_cache = {}
        self._marks_cache = []

        self._subscribe_to_events()
        self.logger.debug("QtSoundController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to sound service events."""
        try:
            self.event_bus.subscribe(SoundListUpdatedEvent, self._on_sound_list_updated)
            self.event_bus.subscribe(SoundDeletedEvent, self._on_sound_deleted)
            self.event_bus.subscribe(AllSoundsResetEvent, self._on_all_sounds_reset)
            self.event_bus.subscribe(SoundToCommandMappingUpdatedEvent, self._on_sound_mapping_updated)
            self.event_bus.subscribe(SoundMappingsResponseEvent, self._on_sound_mappings_response)
            self.event_bus.subscribe(SoundTrainingInitiatedEvent, self._on_training_initiated)
            self.event_bus.subscribe(SoundTrainingProgressEvent, self._on_training_progress)
            self.event_bus.subscribe(SoundTrainingCompleteEvent, self._on_training_complete)
            self.event_bus.subscribe(SoundTrainingFailedEvent, self._on_training_failed)
            self.event_bus.subscribe(SoundTrainingStatusEvent, self._on_training_status)
            self.event_bus.subscribe(MarksChangedEventData, self._on_marks_changed)
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def on_view_ready(self):
        """Request initial data when the view is ready."""
        self.refresh_sound_mappings()
        self._request_marks_for_cache()

    async def _on_sound_list_updated(self, event):
        """Handle sound list updated event."""
        self.available_sounds = getattr(event, "sounds", [])
        self.sounds_loaded.emit(self.available_sounds)

    async def _on_sound_deleted(self, event):
        """Handle sound deleted event."""
        if getattr(event, "success", False):
            self.refresh_sound_list()

    async def _on_all_sounds_reset(self, event):
        """Handle all sounds reset event."""
        if getattr(event, "success", False):
            self.available_sounds = []
            self.refresh_sound_list()
            self.all_sounds_deleted.emit()

    async def _on_sound_mapping_updated(self, event):
        """Handle sound-to-command mapping update event."""
        if event.success:
            self._sound_mappings_cache[event.sound_label] = event.command_phrase
            self.sound_mapping_updated.emit(event.sound_label, event.command_phrase)
            self.refresh_sound_list()

    async def _on_sound_mappings_response(self, event):
        """Handle sound mappings response event."""
        if hasattr(event, "mappings"):
            self._update_sound_mappings_cache(event.mappings)
            self.refresh_sound_list()

    async def _on_training_initiated(self, event):
        """Handle training initiated event."""
        self.training_started.emit(getattr(event, "sound_name", "Unknown"), getattr(event, "total_samples", 0))

    async def _on_training_progress(self, event):
        """Handle training progress event."""
        self.training_progress.emit(
            getattr(event, "label", "Unknown"),
            getattr(event, "current_sample", 0),
            getattr(event, "total_samples", 0),
        )

    async def _on_training_complete(self, event):
        """Handle training complete event."""
        self.refresh_sound_list()
        self.training_completed.emit(getattr(event, "sound_name", "Unknown"))

    async def _on_training_failed(self, event):
        """Handle training failed event."""
        self.training_error.emit(getattr(event, "sound_name", "Unknown"), getattr(event, "reason", "Unknown error"))

    async def _on_training_status(self, event):
        """Handle training status event."""
        self.training_status.emit(getattr(event, "message", ""), getattr(event, "status_type", "info"))

    async def _on_marks_changed(self, event):
        """Handle marks changed event to update the local marks cache."""
        if hasattr(event, "marks") and event.marks:
            if isinstance(event.marks, dict):
                self._marks_cache = list(event.marks.keys())
            else:
                self._marks_cache = [
                    m if isinstance(m, str) else (m.name if hasattr(m, "name") else m.get("name", "")) for m in event.marks if m
                ]
        else:
            self._marks_cache = []

    def delete_individual_sound(self, sound_label: str) -> None:
        """Publish a delete-sound event.

        Args:
            sound_label: Label of the sound to delete.
        """
        asyncio.ensure_future(self.event_bus.publish(DeleteSoundCommand(label=sound_label)))

    def delete_all_sounds(self) -> None:
        """Publish a reset-all-sounds event."""
        asyncio.ensure_future(self.event_bus.publish(ResetAllSoundsCommand()))

    def train_sound(self, sound_name: str, num_samples: int) -> None:
        """Publish a sound training request.

        Args:
            sound_name: Name/label for the sound to train.
            num_samples: Number of training samples to collect.
        """
        asyncio.ensure_future(self.event_bus.publish(SoundTrainingRequestEvent(sound_label=sound_name, num_samples=num_samples)))

    def start_training(self, sound_name: str, num_samples: int) -> None:
        """Alias for train_sound.

        Args:
            sound_name: Name/label for the sound to train.
            num_samples: Number of training samples to collect.
        """
        self.train_sound(sound_name, num_samples)

    def map_sound_to_command(self, sound_label: str, command_phrase: str) -> None:
        """Publish a sound-to-command mapping event and schedule a refresh.

        Args:
            sound_label: Sound label to map.
            command_phrase: Command phrase to associate with the sound.
        """
        asyncio.ensure_future(
            self.event_bus.publish(MapSoundToCommandPhraseCommand(sound_label=sound_label, command_phrase=command_phrase))
        )
        asyncio.get_running_loop().call_later(0.5, self.refresh_sound_mappings)

    def refresh_sound_list(self) -> None:
        """Publish a request for the current sound list."""
        asyncio.ensure_future(self.event_bus.publish(RequestSoundListEvent()))

    def refresh_sound_mappings(self) -> None:
        """Publish a request for sound mappings."""
        asyncio.ensure_future(self.event_bus.publish(RequestSoundMappingsEvent()))

    def _request_marks_for_cache(self) -> None:
        """Publish a request for marks to populate the local marks cache."""
        asyncio.ensure_future(self.event_bus.publish(MarkGetAllRequestEventData()))

    def get_available_sounds(self) -> List[str]:
        """Return the cached list of available sounds."""
        return self.available_sounds

    def get_default_training_samples(self) -> int:
        """Return the default number of training samples."""
        return 5

    def get_sound_command_mapping(self, sound: str) -> Optional[str]:
        """Return the command phrase mapped to a sound, or None.

        Args:
            sound: Sound label to look up.
        """
        return self._sound_mappings_cache.get(sound)

    def _update_sound_mappings_cache(self, mappings: Dict[str, str]):
        """Merge new mappings into the local sound mappings cache.

        Args:
            mappings: Dict of sound label to command phrase.
        """
        self._sound_mappings_cache.update(mappings)

    def get_available_exact_match_commands(self) -> List[str]:
        """Return all available exact-match command phrases."""
        try:
            return sorted(list(set(AutomationCommandRegistry.get_command_phrases())))
        except Exception as e:
            self.logger.error(f"Error getting exact match commands: {e}")
            return []

    def get_available_mark_names(self) -> List[str]:
        """Return cached mark names, requesting them from the service if the cache is empty."""
        if not self._marks_cache:
            self._request_marks_for_cache()
            return []
        return self._marks_cache.copy()

    def get_grid_trigger_words(self) -> List[str]:
        """Return the configured grid trigger phrases."""
        try:
            return [
                self.config.grid.show_grid_phrase,
                self.config.grid.hover_grid_phrase,
                self.config.grid.drag_grid_phrase,
            ]
        except Exception as e:
            self.logger.error(f"Error getting grid trigger words: {e}")
            return []

    def get_mapping_command_types(self) -> List[str]:
        """Return the available command type categories for sound mapping."""
        return ["Commands", "Marks", "Grid"]

    def cleanup(self) -> None:
        """Unsubscribe from all events and release resources."""
        try:
            self.event_bus.unsubscribe(SoundListUpdatedEvent, self._on_sound_list_updated)
            self.event_bus.unsubscribe(SoundDeletedEvent, self._on_sound_deleted)
            self.event_bus.unsubscribe(AllSoundsResetEvent, self._on_all_sounds_reset)
            self.event_bus.unsubscribe(SoundToCommandMappingUpdatedEvent, self._on_sound_mapping_updated)
            self.event_bus.unsubscribe(SoundMappingsResponseEvent, self._on_sound_mappings_response)
            self.event_bus.unsubscribe(SoundTrainingInitiatedEvent, self._on_training_initiated)
            self.event_bus.unsubscribe(SoundTrainingProgressEvent, self._on_training_progress)
            self.event_bus.unsubscribe(SoundTrainingCompleteEvent, self._on_training_complete)
            self.event_bus.unsubscribe(SoundTrainingFailedEvent, self._on_training_failed)
            self.event_bus.unsubscribe(SoundTrainingStatusEvent, self._on_training_status)
            self.event_bus.unsubscribe(MarksChangedEventData, self._on_marks_changed)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
