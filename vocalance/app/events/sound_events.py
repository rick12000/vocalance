from typing import Dict, List, Literal

from vocalance.app.events.base_event import BaseEvent

SoundUiOp = Literal["delete", "reset_all", "train", "map", "refresh_snapshots"]


class SoundTrainingInitiatedEvent(BaseEvent):
    """Event indicating sound training has been initiated."""

    sound_name: str
    total_samples: int


class SoundTrainingProgressEvent(BaseEvent):
    """Progress update event during sound training."""

    label: str
    current_sample: int
    total_samples: int
    is_last_sample: bool = False


class SoundTrainingCompleteEvent(BaseEvent):
    """Event indicating sound training completion."""

    sound_name: str
    success: bool


class SoundTrainingFailedEvent(BaseEvent):
    """Event indicating sound training failed."""

    sound_name: str
    reason: str


class SoundListUpdatedEvent(BaseEvent):
    """Broadcast providing the current list of trained sounds."""

    sounds: List[str]


class SoundToCommandMappingUpdatedEvent(BaseEvent):
    """Broadcast when a sound-to-command mapping is updated."""

    sound_label: str
    command_phrase: str
    success: bool


class SoundMappingsResponseEvent(BaseEvent):
    """Broadcast providing the full current sound-to-command mappings."""

    mappings: Dict[str, str] = {}


class SoundUiOperationEvent(BaseEvent):
    """UI-originated sound operations; handled by ``SoundService``."""

    op: SoundUiOp
    sound_label: str = ""
    sound_name: str = ""
    num_samples: int = 0
    command_phrase: str = ""
