import logging
import asyncio
from typing import List, Optional, Dict
from iris.app.ui.controls.base_control import BaseController
from iris.app.events.sound_events import (
    RequestSoundListEvent, DeleteSoundCommand, ResetAllSoundsCommand,
    MapSoundToCommandPhraseCommand, SoundTrainingRequestEvent,
    SoundToCommandMappingUpdatedEvent, RequestSoundMappingsEvent,
    SoundMappingsResponseEvent,
    SoundListUpdatedEvent, SoundDeletedEvent, AllSoundsResetEvent,
    SoundTrainingInitiatedEvent, SoundTrainingProgressEvent, 
    SoundTrainingCompleteEvent, SoundTrainingFailedEvent, SoundTrainingStatusEvent
)
from iris.app.events.mark_events import MarkGetAllRequestEventData, MarksChangedEventData
from iris.app.config.automation_command_registry import AutomationCommandRegistry
from iris.app.config.app_config import GlobalAppConfig


class SoundController(BaseController):
    """Business logic controller for sound functionality."""
    
    def __init__(self, event_bus, event_loop, logger):
        super().__init__(event_bus, event_loop, logger, "SoundController")
        
        self.available_sounds = []  # Cache of available sounds for dropdown
        self._sound_mappings_cache = {}  # Cache for sound mappings
        self._marks_cache = []  # Cache for available marks
        
        self.subscribe_to_events([
            (SoundListUpdatedEvent, self._on_sound_list_updated),
            (SoundDeletedEvent, self._on_sound_deleted),
            (AllSoundsResetEvent, self._on_all_sounds_reset),
            (SoundToCommandMappingUpdatedEvent, self._on_sound_mapping_updated),
            (SoundMappingsResponseEvent, self._on_sound_mappings_response),
            (SoundTrainingInitiatedEvent, self._on_training_initiated),
            (SoundTrainingProgressEvent, self._on_training_progress),
            (SoundTrainingCompleteEvent, self._on_training_complete),
            (SoundTrainingFailedEvent, self._on_training_failed),
            (SoundTrainingStatusEvent, self._on_training_status),
            (MarksChangedEventData, self._on_marks_changed),
        ])

    def on_view_ready(self):
        """Request initial data when view is ready."""
        self.refresh_sound_mappings()
        self._request_marks_for_cache()

    async def _on_sound_list_updated(self, event):
        """Handle sound list updated event."""
        self.available_sounds = getattr(event, 'sounds', [])
        if self.view_callback:
            self.schedule_ui_update(self.view_callback.on_sounds_updated, self.available_sounds)

    async def _on_sound_deleted(self, event):
        """Handle sound deleted event."""
        if getattr(event, 'success', False):
            self.refresh_sound_list()

    async def _on_all_sounds_reset(self, event):
        """Handle all sounds reset event."""
        if getattr(event, 'success', False):
            self.available_sounds = []
            self.refresh_sound_list()

    async def _on_sound_mapping_updated(self, event):
        """Handle sound-to-command mapping update events."""
        if event.success:
            self._sound_mappings_cache[event.sound_label] = event.command_phrase
            self.refresh_sound_list()

    async def _on_sound_mappings_response(self, event):
        """Handle sound mappings response event."""
        if hasattr(event, 'mappings'):
            self._update_sound_mappings_cache(event.mappings)
            if self.view_callback:
                self.refresh_sound_list()

    async def _on_training_initiated(self, event):
        """Handle training initiated event."""
        sound_name = getattr(event, 'sound_name', 'Unknown')
        total_samples = getattr(event, 'total_samples', 0)
        if self.view_callback:
            self.schedule_ui_update(self.view_callback.on_training_initiated, sound_name, total_samples)

    async def _on_training_progress(self, event):
        """Handle training progress event."""
        label = getattr(event, 'label', 'Unknown')
        current_sample = getattr(event, 'current_sample', 0)
        total_samples = getattr(event, 'total_samples', 0)
        is_last = getattr(event, 'is_last_sample', False)
        
        if self.view_callback:
            # Use the existing sample_recorded callback for compatibility
            self.schedule_ui_update(self.view_callback.on_sample_recorded, current_sample, total_samples, is_last)
            # Also call the progress callback
            self.schedule_ui_update(self.view_callback.on_training_progress, label, current_sample, total_samples)

    async def _on_training_complete(self, event):
        """Handle training complete event."""
        sound_name = getattr(event, 'sound_name', 'Unknown')
        self.refresh_sound_list()
        if self.view_callback:
            self.schedule_ui_update(self.view_callback.on_training_complete, sound_name)

    async def _on_training_failed(self, event):
        """Handle training failed event."""
        sound_name = getattr(event, 'sound_name', 'Unknown')
        reason = getattr(event, 'reason', 'Unknown error')
        if self.view_callback:
            self.schedule_ui_update(self.view_callback.on_training_failed, sound_name, reason)

    async def _on_training_status(self, event):
        """Handle training status event."""
        message = getattr(event, 'message', '')
        status_type = getattr(event, 'status_type', 'info')
        
        if self.view_callback:
            self.schedule_ui_update(self.view_callback.on_training_status, message, status_type)

    async def _on_marks_changed(self, event):
        """Handle marks changed event to update cache."""
        if hasattr(event, 'marks') and event.marks:
            if isinstance(event.marks, dict):
                self._marks_cache = list(event.marks.keys())
            else:
                # Fallback for other formats
                marks_list = []
                for mark in event.marks:
                    if isinstance(mark, str) and mark.strip():
                        marks_list.append(mark.strip())
                    elif hasattr(mark, 'name') and mark.name:
                        marks_list.append(mark.name)
                    elif hasattr(mark, 'get') and mark.get('name'):
                        marks_list.append(mark.get('name'))
                self._marks_cache = marks_list
        else:
            self._marks_cache = []

    def delete_individual_sound(self, sound_label: str) -> None:
        """Delete an individual sound."""
        event = DeleteSoundCommand(label=sound_label)
        self.publish_event(event)

    def delete_all_sounds(self) -> None:
        """Delete all sounds."""
        event = ResetAllSoundsCommand()
        self.publish_event(event)

    def train_sound(self, sound_name: str, num_samples: int) -> None:
        """Train a new sound."""
        event = SoundTrainingRequestEvent(sound_label=sound_name, num_samples=num_samples)
        self.publish_event(event)

    def map_sound_to_command(self, sound_label: str, command_phrase: str) -> None:
        """Map a sound to a command phrase."""
        mapping_command = MapSoundToCommandPhraseCommand(
            sound_label=sound_label,
            command_phrase=command_phrase
        )
        self.publish_event(mapping_command)
        
        # Schedule a delayed refresh to allow backend processing to complete
        self.event_loop.call_later(0.5, self.refresh_sound_mappings)

    def refresh_sound_list(self) -> None:
        """Refresh the sound list by requesting it from the event bus."""
        event = RequestSoundListEvent()
        self.publish_event(event)
    
    def refresh_sound_mappings(self) -> None:
        """Request sound mappings from the service."""
        event = RequestSoundMappingsEvent()
        self.publish_event(event)

    def _request_marks_for_cache(self) -> None:
        """Request marks from service to populate cache."""
        event = MarkGetAllRequestEventData()
        self.publish_event(event)

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
            config = GlobalAppConfig()
            return [config.grid.show_grid_phrase]
        except Exception as e:
            self.logger.error(f"Error getting grid trigger words: {e}")
            return []

    def get_mapping_command_types(self) -> List[str]:
        """Get available command types for mapping."""
        return ["Commands", "Marks", "Grid"] 