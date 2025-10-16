from typing import Dict, List, Literal

from pydantic import Field

from iris.app.events.base_event import BaseEvent, EventPriority


class SoundTrainingRequestEvent(BaseEvent):
    sound_label: str
    num_samples: int
    priority: EventPriority = EventPriority.NORMAL


class CancelSoundTrainingCommand(BaseEvent):
    label: str = Field(..., description="Label for the sound training to cancel.")
    priority: EventPriority = EventPriority.NORMAL


class SoundTrainingInitiatedEvent(BaseEvent):
    sound_name: str
    total_samples: int
    priority: EventPriority = EventPriority.NORMAL


class SoundTrainingStatusEvent(BaseEvent):
    message: str
    status_type: Literal["info", "warning", "error", "success"]
    priority: EventPriority = EventPriority.LOW


class SoundTrainingProgressEvent(BaseEvent):
    label: str
    current_sample: int
    total_samples: int
    is_last_sample: bool = False
    priority: EventPriority = EventPriority.LOW


class SoundTrainingCompleteEvent(BaseEvent):
    sound_name: str
    success: bool
    priority: EventPriority = EventPriority.NORMAL


class SoundTrainingFailedEvent(BaseEvent):
    sound_name: str
    reason: str
    priority: EventPriority = EventPriority.NORMAL


class DeleteSoundCommand(BaseEvent):
    label: str = Field(..., description="Label of the sound to delete.")
    priority: EventPriority = EventPriority.NORMAL


class ResetAllSoundsCommand(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL


class SoundDeletedEvent(BaseEvent):
    label: str
    success: bool
    priority: EventPriority = EventPriority.LOW


class AllSoundsResetEvent(BaseEvent):
    success: bool
    priority: EventPriority = EventPriority.LOW


class RequestSoundListEvent(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL


class SoundListUpdatedEvent(BaseEvent):
    sounds: List[str]
    priority: EventPriority = EventPriority.LOW


class MapSoundToCommandPhraseCommand(BaseEvent):
    sound_label: str = Field(..., description="Label of the sound to map.")
    command_phrase: str = Field(..., description="Command phrase to map to the sound.")
    priority: EventPriority = EventPriority.NORMAL


class SoundToCommandMappingUpdatedEvent(BaseEvent):
    sound_label: str
    command_phrase: str
    success: bool
    priority: EventPriority = EventPriority.LOW


class RequestSoundMappingsEvent(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL


class SoundMappingsResponseEvent(BaseEvent):
    mappings: Dict[str, str] = {}
    priority: EventPriority = EventPriority.LOW
