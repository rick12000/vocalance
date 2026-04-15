import asyncio
import logging
from typing import Dict, List, Optional

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.automation_command_registry import AutomationCommandRegistry
from vocalance.app.event_bus import EventBus
from vocalance.app.events.mark_events import MarksChangedEventData, MarkUiRequestEvent
from vocalance.app.events.sound_events import (
    SoundListUpdatedEvent,
    SoundMappingsResponseEvent,
    SoundToCommandMappingUpdatedEvent,
    SoundTrainingCompleteEvent,
    SoundTrainingFailedEvent,
    SoundTrainingInitiatedEvent,
    SoundTrainingProgressEvent,
    SoundUiOperationEvent,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtSoundController(QtBaseController):
    sounds_loaded = Signal(list)
    training_started = Signal(str, int)
    training_progress = Signal(str, int, int)
    training_completed = Signal(str)
    training_error = Signal(str, str)
    sound_mapping_updated = Signal(str, str)
    operation_error = Signal(str)

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtSoundController"),
        )

        self.config = config

        self.available_sounds = []
        self._sound_mappings_cache = {}
        self._marks_cache = []

        self.event_bus.subscribe(SoundListUpdatedEvent, self._on_sound_list_updated)
        self.event_bus.subscribe(SoundMappingsResponseEvent, self._on_sound_mappings_response)
        self.event_bus.subscribe(SoundToCommandMappingUpdatedEvent, self._on_sound_mapping_updated)
        self.event_bus.subscribe(SoundTrainingInitiatedEvent, self._on_training_initiated)
        self.event_bus.subscribe(SoundTrainingProgressEvent, self._on_training_progress)
        self.event_bus.subscribe(SoundTrainingCompleteEvent, self._on_training_complete)
        self.event_bus.subscribe(SoundTrainingFailedEvent, self._on_training_failed)
        self.event_bus.subscribe(MarksChangedEventData, self._on_marks_changed)

    def on_view_ready(self) -> None:
        asyncio.create_task(self.event_bus.publish(SoundUiOperationEvent(op="refresh_snapshots")))
        asyncio.create_task(self.event_bus.publish(MarkUiRequestEvent(op="refresh_list")))

    def _on_marks_changed(self, snapshot: MarksChangedEventData) -> None:
        self._marks_cache = list(snapshot.marks.keys()) if snapshot.marks else []

    def _on_sound_list_updated(self, list_update: SoundListUpdatedEvent) -> None:
        self.available_sounds = list_update.sounds
        self.sounds_loaded.emit(self.available_sounds)

    def _on_sound_mappings_response(self, mappings_snapshot: SoundMappingsResponseEvent) -> None:
        self._update_sound_mappings_cache(mappings_snapshot.mappings)

    def _on_sound_mapping_updated(self, mapping_update: SoundToCommandMappingUpdatedEvent) -> None:
        if mapping_update.success:
            self._sound_mappings_cache[mapping_update.sound_label] = mapping_update.command_phrase
            self.sound_mapping_updated.emit(mapping_update.sound_label, mapping_update.command_phrase)

    def _on_training_initiated(self, initiated: SoundTrainingInitiatedEvent) -> None:
        self.training_started.emit(initiated.sound_name, initiated.total_samples)

    def _on_training_progress(self, progress: SoundTrainingProgressEvent) -> None:
        self.training_progress.emit(progress.label, progress.current_sample, progress.total_samples)

    def _on_training_complete(self, complete: SoundTrainingCompleteEvent) -> None:
        asyncio.create_task(self.event_bus.publish(SoundUiOperationEvent(op="refresh_snapshots")))
        self.training_completed.emit(complete.sound_name)

    def _on_training_failed(self, failed: SoundTrainingFailedEvent) -> None:
        self.training_error.emit(failed.sound_name, failed.reason)

    def delete_individual_sound(self, sound_label: str) -> None:
        asyncio.create_task(self.event_bus.publish(SoundUiOperationEvent(op="delete", sound_label=sound_label)))

    def delete_all_sounds(self) -> None:
        asyncio.create_task(self.event_bus.publish(SoundUiOperationEvent(op="reset_all")))

    def train_sound(self, sound_name: str, num_samples: int) -> None:
        asyncio.create_task(
            self.event_bus.publish(SoundUiOperationEvent(op="train", sound_name=sound_name, num_samples=num_samples))
        )

    def start_training(self, sound_name: str, num_samples: int) -> None:
        self.train_sound(sound_name, num_samples)

    def map_sound_to_command(self, sound_label: str, command_phrase: str) -> None:
        asyncio.create_task(
            self.event_bus.publish(SoundUiOperationEvent(op="map", sound_label=sound_label, command_phrase=command_phrase))
        )

    def refresh_sound_list(self) -> None:
        asyncio.create_task(self.event_bus.publish(SoundUiOperationEvent(op="refresh_snapshots")))

    def refresh_sound_mappings(self) -> None:
        asyncio.create_task(self.event_bus.publish(SoundUiOperationEvent(op="refresh_snapshots")))

    def get_available_sounds(self) -> List[str]:
        return self.available_sounds

    def get_default_training_samples(self) -> int:
        return 5

    def get_sound_command_mapping(self, sound: str) -> Optional[str]:
        return self._sound_mappings_cache.get(sound)

    def _update_sound_mappings_cache(self, mappings: Dict[str, str]) -> None:
        self._sound_mappings_cache.update(mappings)

    def get_available_exact_match_commands(self) -> List[str]:
        try:
            return sorted(list(set(AutomationCommandRegistry.get_command_phrases())))
        except Exception as e:
            self.logger.error(f"Error getting exact match commands: {e}")
            return []

    def get_available_mark_names(self) -> List[str]:
        return self._marks_cache.copy()

    def get_grid_trigger_words(self) -> List[str]:
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
        return ["Commands", "Marks", "Grid"]

    def cleanup(self) -> None:
        try:
            self.event_bus.unsubscribe(SoundListUpdatedEvent, self._on_sound_list_updated)
            self.event_bus.unsubscribe(SoundMappingsResponseEvent, self._on_sound_mappings_response)
            self.event_bus.unsubscribe(SoundToCommandMappingUpdatedEvent, self._on_sound_mapping_updated)
            self.event_bus.unsubscribe(SoundTrainingInitiatedEvent, self._on_training_initiated)
            self.event_bus.unsubscribe(SoundTrainingProgressEvent, self._on_training_progress)
            self.event_bus.unsubscribe(SoundTrainingCompleteEvent, self._on_training_complete)
            self.event_bus.unsubscribe(SoundTrainingFailedEvent, self._on_training_failed)
            self.event_bus.unsubscribe(MarksChangedEventData, self._on_marks_changed)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
