from typing import Dict, List, Literal

from pydantic import Field

from vocalance.app.events.base_event import BaseEvent, EventPriority


class SoundTrainingRequestEvent(BaseEvent):
    """Request to initiate sound training for a specific label."""

    sound_label: str
    num_samples: int
    priority: EventPriority = EventPriority.NORMAL


class CancelSoundTrainingCommand(BaseEvent):
    """Command to cancel ongoing sound training."""

    label: str = Field(..., description="Label for the sound training to cancel.")
    priority: EventPriority = EventPriority.NORMAL


class SoundTrainingInitiatedEvent(BaseEvent):
    """Event indicating sound training has been initiated."""

    sound_name: str
    total_samples: int
    priority: EventPriority = EventPriority.NORMAL


class SoundTrainingStatusEvent(BaseEvent):
    """Status message event during sound training process."""

    message: str
    status_type: Literal["info", "warning", "error", "success"]
    priority: EventPriority = EventPriority.LOW


class SoundTrainingProgressEvent(BaseEvent):
    """Progress update event during sound training."""

    label: str
    current_sample: int
    total_samples: int
    is_last_sample: bool = False
    priority: EventPriority = EventPriority.LOW


class SoundTrainingCompleteEvent(BaseEvent):
    """Event indicating sound training completion."""

    sound_name: str
    success: bool
    priority: EventPriority = EventPriority.NORMAL


class SoundTrainingFailedEvent(BaseEvent):
    """Event indicating sound training failed."""

    sound_name: str
    reason: str
    priority: EventPriority = EventPriority.NORMAL


class DeleteSoundCommand(BaseEvent):
    """Command to delete a sound mapping."""

    label: str = Field(..., description="Label of the sound to delete.")
    priority: EventPriority = EventPriority.NORMAL


class ResetAllSoundsCommand(BaseEvent):
    """Command to reset all sound mappings."""

    priority: EventPriority = EventPriority.NORMAL


class SoundDeletedEvent(BaseEvent):
    """Event indicating a sound has been deleted."""

    label: str
    success: bool
    priority: EventPriority = EventPriority.LOW


class AllSoundsResetEvent(BaseEvent):
    """Event indicating all sounds have been reset."""

    success: bool
    priority: EventPriority = EventPriority.LOW


class RequestSoundListEvent(BaseEvent):
    """Request to retrieve list of all sounds."""

    priority: EventPriority = EventPriority.NORMAL


class SoundListUpdatedEvent(BaseEvent):
    """Event providing updated list of available sounds."""

    sounds: List[str]
    priority: EventPriority = EventPriority.LOW


class MapSoundToCommandPhraseCommand(BaseEvent):
    """Command to map a sound to a command phrase."""

    sound_label: str = Field(..., description="Label of the sound to map.")
    command_phrase: str = Field(..., description="Command phrase to map to the sound.")
    priority: EventPriority = EventPriority.NORMAL


class SoundToCommandMappingUpdatedEvent(BaseEvent):
    """Event indicating a sound-to-command mapping has been updated."""

    sound_label: str
    command_phrase: str
    success: bool
    priority: EventPriority = EventPriority.LOW


class RequestSoundMappingsEvent(BaseEvent):
    """Request to retrieve current sound-to-command mappings."""

    priority: EventPriority = EventPriority.NORMAL


class SoundMappingsResponseEvent(BaseEvent):
    """Response event providing current sound-to-command mappings."""

    mappings: Dict[str, str] = {}
    priority: EventPriority = EventPriority.LOW
