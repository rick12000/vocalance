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

    # Signals for sound operations
    sounds_loaded = Signal(list)  # List of sound names
    training_started = Signal(str, int)  # sound_name, total_samples
    training_progress = Signal(str, int, int)  # sound_name, current, total
    training_completed = Signal(str)  # sound_name
    training_error = Signal(str, str)  # sound_name, error_msg
    training_status = Signal(str, str)  # message, status_type
    sound_deleted = Signal(str)  # sound_name
    all_sounds_deleted = Signal()
    sound_mapping_updated = Signal(str, str)  # sound_label, command_phrase
    operation_error = Signal(str)

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        sound_service,
        storage_service,
        config: GlobalAppConfig,
        main_window,
    ):
        """Initialize sound controller.

        Args:
            event_bus: Event bus for pub/sub.
            event_loop: Asyncio event loop.
            sound_service: Sound service instance.
            storage_service: Storage service instance.
            config: Global app configuration.
            main_window: Main window reference.
        """
        super().__init__(
            event_bus=event_bus,
            event_loop=event_loop,
            logger=logging.getLogger("QtSoundController"),
        )

        self.sound_service = sound_service
        self.storage_service = storage_service
        self.config = config
        self.main_window = main_window

        # Caches matching legacy
        self.available_sounds = []  # Cache of available sounds for dropdown
        self._sound_mappings_cache = {}  # Cache for sound mappings
        self._marks_cache = []  # Cache for available marks

        # Subscribe to sound service events
        self._subscribe_to_events()

        self.logger.debug("QtSoundController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to sound-related events using exact legacy event types."""
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
            self.logger.debug("Subscribed to sound events (legacy types)")
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def on_view_ready(self):
        """Request initial data when view is ready."""
        self.refresh_sound_mappings()
        self._request_marks_for_cache()

    # --- Event Handlers ---

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
        """Handle sound-to-command mapping update events."""
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
        sound_name = getattr(event, "sound_name", "Unknown")
        total_samples = getattr(event, "total_samples", 0)
        self.training_started.emit(sound_name, total_samples)

    async def _on_training_progress(self, event):
        """Handle training progress event."""
        label = getattr(event, "label", "Unknown")
        current_sample = getattr(event, "current_sample", 0)
        total_samples = getattr(event, "total_samples", 0)
        self.training_progress.emit(label, current_sample, total_samples)

    async def _on_training_complete(self, event):
        """Handle training complete event."""
        sound_name = getattr(event, "sound_name", "Unknown")
        self.refresh_sound_list()
        self.training_completed.emit(sound_name)

    async def _on_training_failed(self, event):
        """Handle training failed event."""
        sound_name = getattr(event, "sound_name", "Unknown")
        reason = getattr(event, "reason", "Unknown error")
        self.training_error.emit(sound_name, reason)

    async def _on_training_status(self, event):
        """Handle training status event."""
        message = getattr(event, "message", "")
        status_type = getattr(event, "status_type", "info")
        self.training_status.emit(message, status_type)

    async def _on_marks_changed(self, event):
        """Handle marks changed event to update cache."""
        if hasattr(event, "marks") and event.marks:
            if isinstance(event.marks, dict):
                self._marks_cache = list(event.marks.keys())
            else:
                # Fallback for other formats
                marks_list = []
                for mark in event.marks:
                    if isinstance(mark, str) and mark.strip():
                        marks_list.append(mark.strip())
                    elif hasattr(mark, "name") and mark.name:
                        marks_list.append(mark.name)
                    elif hasattr(mark, "get") and mark.get("name"):
                        marks_list.append(mark.get("name"))
                self._marks_cache = marks_list
        else:
            self._marks_cache = []

    # --- Public Methods (Publish Events) ---

    def delete_individual_sound(self, sound_label: str) -> None:
        """Delete an individual sound."""
        event = DeleteSoundCommand(label=sound_label)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def delete_all_sounds(self) -> None:
        """Delete all sounds."""
        event = ResetAllSoundsCommand()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def train_sound(self, sound_name: str, num_samples: int) -> None:
        """Train a new sound."""
        event = SoundTrainingRequestEvent(sound_label=sound_name, num_samples=num_samples)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def start_training(self, sound_name: str, num_samples: int) -> None:
        """Start training a sound (alias for train_sound for legacy compatibility)."""
        self.train_sound(sound_name, num_samples)

    def map_sound_to_command(self, sound_label: str, command_phrase: str) -> None:
        """Map a sound to a command phrase."""
        mapping_command = MapSoundToCommandPhraseCommand(sound_label=sound_label, command_phrase=command_phrase)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(mapping_command), self.event_loop)

        # Refresh after a short delay
        self.event_loop.call_later(0.5, self.refresh_sound_mappings)

    def refresh_sound_list(self) -> None:
        """Refresh the sound list by requesting it from the event bus."""
        event = RequestSoundListEvent()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def refresh_sound_mappings(self) -> None:
        """Request sound mappings from the service."""
        event = RequestSoundMappingsEvent()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def _request_marks_for_cache(self) -> None:
        """Request marks from service to populate cache."""
        event = MarkGetAllRequestEventData()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    # --- Getters for View ---

    def get_available_sounds(self) -> List[str]:
        """Get the list of available sounds."""
        return self.available_sounds

    def get_default_training_samples(self) -> int:
        """Get default number of training samples."""
        return 5

    def get_sound_command_mapping(self, sound: str) -> Optional[str]:
        """Get command mapping for a sound if available."""
        return self._sound_mappings_cache.get(sound)

    def _update_sound_mappings_cache(self, mappings: Dict[str, str]):
        """Update the local cache of sound mappings."""
        self._sound_mappings_cache.update(mappings)

    def get_available_exact_match_commands(self) -> List[str]:
        """Get all available exact match commands."""
        try:
            automation_phrases = AutomationCommandRegistry.get_command_phrases()
            return sorted(list(set(automation_phrases)))
        except Exception as e:
            self.logger.error(f"Error getting exact match commands: {e}")
            return []

    def get_available_mark_names(self) -> List[str]:
        """Get list of available mark names for sound mapping."""
        if not self._marks_cache:
            # Request marks from service if cache is empty
            self._request_marks_for_cache()
            return []
        return self._marks_cache.copy()

    def get_grid_trigger_words(self) -> List[str]:
        """Get grid trigger words from config."""
        try:
            return [self.config.grid.show_grid_phrase, self.config.grid.hover_grid_phrase]
        except Exception as e:
            self.logger.error(f"Error getting grid trigger words: {e}")
            return []

    def get_mapping_command_types(self) -> List[str]:
        """Get available command types for mapping."""
        return ["Commands", "Marks", "Grid"]

    def cleanup(self) -> None:
        """Clean up controller resources."""
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
