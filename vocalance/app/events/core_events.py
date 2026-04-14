from typing import Any, Dict, Optional

from pydantic import Field

from vocalance.app.events.base_event import BaseEvent


class CommandAudioSegmentReadyEvent(BaseEvent):
    """Audio segment ready for command mode processing."""

    audio_bytes: bytes
    sample_rate: int


class AudioDetectedEvent(BaseEvent):
    """Published immediately when audio above threshold is detected."""

    timestamp: float = Field(description="Timestamp when audio was detected")


class ProcessAudioChunkForSoundRecognitionEvent(BaseEvent):
    """Audio chunk ready for sound recognition processing."""

    audio_chunk: bytes
    sample_rate: int = 16000


class CustomSoundRecognizedEvent(BaseEvent):
    """Custom sound recognized by sound recognizer."""

    label: str
    confidence: float
    mapped_command: Optional[str] = None


class MouseClickEvent(BaseEvent):
    """Request to perform a mouse click."""

    x: int
    y: int
    source: Optional[str] = "unknown"


# Keep the old name as an alias so existing consumers don't break before migration
PerformMouseClickEventData = MouseClickEvent


class MarkovPredictionEvent(BaseEvent):
    """Published when Markov chain predicts a command."""

    predicted_command: str = Field(description="The predicted command text")
    confidence: float = Field(description="Confidence probability (0.0-1.0)")
    audio_id: int = Field(description="ID of the audio bytes that triggered this prediction")


class MarkovPredictionFeedbackEvent(BaseEvent):
    """Feedback about Markov prediction accuracy."""

    predicted_command: str = Field(description="The command that was predicted")
    actual_command: str = Field(description="The command that was actually recognized")
    was_correct: bool = Field(description="True if prediction matched actual command")
    source: str = Field(description="Source of actual command: 'stt' or 'sound'")


class SettingsChangedEvent(BaseEvent):
    """Event published when runtime settings are changed."""

    updated_settings: Dict[str, Any] = Field(description="Dictionary of setting paths to new values")
    all_settings: Dict[str, Any] = Field(description="Dictionary of all effective settings")


class CommandTextRecognizedEvent(BaseEvent):
    """Text recognized in command mode."""

    text: str
    confidence: float = 1.0
    engine: str = "unknown"
    processing_time_ms: float = 0.0
    mode: str = "command"


class DictationTextRecognizedEvent(BaseEvent):
    """Text recognized in dictation mode."""

    text: str
    confidence: float = 1.0
    engine: str = "unknown"
    processing_time_ms: float = 0.0
    mode: str = "command"


class AudioDeviceErrorEvent(BaseEvent):
    """Event published when the default input device is lost or capture fails."""

    error_message: str = Field(description="User-facing message for the warning dialog")
