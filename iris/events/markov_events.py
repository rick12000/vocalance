"""Events related to Markov chain command prediction"""

from pydantic import Field
from iris.events.base_event import BaseEvent, EventPriority


class MarkovPredictionEvent(BaseEvent):
    """Published when Markov chain predicts a command with high confidence"""
    predicted_command: str = Field(description="The predicted command text")
    confidence: float = Field(description="Confidence probability (0.0-1.0)")
    audio_id: int = Field(description="ID of the audio bytes that triggered this prediction")
    priority: EventPriority = EventPriority.CRITICAL


class CommandExecutedEvent(BaseEvent):
    """Published when a command is successfully executed"""
    command_text: str = Field(description="The executed command text")
    timestamp: float = Field(description="Unix timestamp of execution")
    priority: EventPriority = EventPriority.LOW

