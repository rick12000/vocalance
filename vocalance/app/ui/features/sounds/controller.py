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

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger("QtSoundController"))
        self.config = config
        self.available_sounds: List[str] = []
        self.sound_mappings_cache: Dict[str, str] = {}
        self.marks_cache: List[str] = []
        self.subscribe(SoundListUpdatedEvent, self.on_sound_list_updated)
        self.subscribe(SoundMappingsResponseEvent, self.on_sound_mappings_response)
        self.subscribe(SoundToCommandMappingUpdatedEvent, self.on_sound_mapping_updated)
        self.subscribe(SoundTrainingInitiatedEvent, self.on_training_initiated)
        self.subscribe(SoundTrainingProgressEvent, self.on_training_progress)
        self.subscribe(SoundTrainingCompleteEvent, self.on_training_complete)
        self.subscribe(SoundTrainingFailedEvent, self.on_training_failed)
        self.subscribe(MarksChangedEventData, self.on_marks_changed)

    def sound_op(self, op: str, **kwargs: object) -> None:
        asyncio.create_task(self.event_bus.publish(SoundUiOperationEvent(op=op, **kwargs)))

    def on_view_ready(self) -> None:
        self.refresh_snapshots()
        asyncio.create_task(self.event_bus.publish(MarkUiRequestEvent(op="refresh_list")))

    def refresh_snapshots(self) -> None:
        self.sound_op(op="refresh_snapshots")

    def on_marks_changed(self, snapshot: MarksChangedEventData) -> None:
        self.marks_cache = list(snapshot.marks.keys()) if snapshot.marks else []

    def on_sound_list_updated(self, list_update: SoundListUpdatedEvent) -> None:
        self.available_sounds = list_update.sounds
        self.sounds_loaded.emit(self.available_sounds)

    def on_sound_mappings_response(self, mappings_snapshot: SoundMappingsResponseEvent) -> None:
        self.sound_mappings_cache.update(mappings_snapshot.mappings)

    def on_sound_mapping_updated(self, mapping_update: SoundToCommandMappingUpdatedEvent) -> None:
        if mapping_update.success:
            self.sound_mappings_cache[mapping_update.sound_label] = mapping_update.command_phrase
            self.sound_mapping_updated.emit(mapping_update.sound_label, mapping_update.command_phrase)

    def on_training_initiated(self, initiated: SoundTrainingInitiatedEvent) -> None:
        self.training_started.emit(initiated.sound_name, initiated.total_samples)

    def on_training_progress(self, progress: SoundTrainingProgressEvent) -> None:
        self.training_progress.emit(progress.label, progress.current_sample, progress.total_samples)

    def on_training_complete(self, complete: SoundTrainingCompleteEvent) -> None:
        self.refresh_snapshots()
        self.training_completed.emit(complete.sound_name)

    def on_training_failed(self, failed: SoundTrainingFailedEvent) -> None:
        self.training_error.emit(failed.sound_name, failed.reason)

    def delete_individual_sound(self, sound_label: str) -> None:
        self.sound_op(op="delete", sound_label=sound_label)

    def delete_all_sounds(self) -> None:
        self.sound_op(op="reset_all")

    def train_sound(self, sound_name: str, num_samples: int) -> None:
        self.sound_op(op="train", sound_name=sound_name, num_samples=num_samples)

    def map_sound_to_command(self, sound_label: str, command_phrase: str) -> None:
        self.sound_op(op="map", sound_label=sound_label, command_phrase=command_phrase)

    def get_available_sounds(self) -> List[str]:
        return self.available_sounds

    def get_default_training_samples(self) -> int:
        return 5

    def get_sound_command_mapping(self, sound: str) -> Optional[str]:
        return self.sound_mappings_cache.get(sound)

    def get_available_exact_match_commands(self) -> List[str]:
        return sorted(set(AutomationCommandRegistry.get_command_phrases()))

    def get_available_mark_names(self) -> List[str]:
        return self.marks_cache.copy()

    def get_grid_trigger_words(self) -> List[str]:
        return [
            self.config.grid.show_grid_phrase,
            self.config.grid.hover_grid_phrase,
            self.config.grid.drag_grid_phrase,
        ]

    def get_mapping_command_types(self) -> List[str]:
        return ["Commands", "Marks", "Grid"]
