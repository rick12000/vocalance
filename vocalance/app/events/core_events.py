from typing import Any, Dict, Optional

from pydantic import Field

from vocalance.app.events.base_event import BaseEvent, EventPriority


class CommandAudioSegmentReadyEvent(BaseEvent):
    """Audio segment ready for command mode processing.

    Attributes:
        audio_bytes: Audio data.
        sample_rate: Sample rate of audio.
    """

    audio_bytes: bytes
    sample_rate: int
    priority: EventPriority = EventPriority.HIGH


class AudioDetectedEvent(BaseEvent):
    """Published immediately when audio above threshold is detected.

    Attributes:
        timestamp: Timestamp when audio was detected.
    """

    timestamp: float = Field(description="Timestamp when audio was detected")
    priority: EventPriority = EventPriority.CRITICAL


class ProcessAudioChunkForSoundRecognitionEvent(BaseEvent):
    """Audio chunk ready for sound recognition processing.

    Rate-limited to 100ms intervals (10/sec) to balance responsiveness with CPU efficiency.
    LOW priority ensures sound recognition doesn't interfere with command/dictation processing.

    Attributes:
        audio_chunk: Audio data chunk (100ms at 50ms/chunk base rate).
        sample_rate: Sample rate of audio.
    """

    audio_chunk: bytes
    sample_rate: int = 16000
    priority: EventPriority = EventPriority.LOW


class TextRecognizedEvent(BaseEvent):
    """Text recognized from audio processing.

    Attributes:
        text: Recognized text.
        confidence: Confidence score (0.0-1.0).
        engine: STT engine used.
        processing_time_ms: Time to process in milliseconds.
        mode: Processing mode (command or dictation).
    """

    text: str
    confidence: float = 1.0
    engine: str = "unknown"
    processing_time_ms: float = 0.0
    mode: str = "command"
    priority: EventPriority = EventPriority.HIGH


class CustomSoundRecognizedEvent(BaseEvent):
    """Custom sound recognized by sound recognizer.

    Attributes:
        label: Sound label.
        confidence: Confidence score.
        mapped_command: Command mapped to this sound.
    """

    label: str
    confidence: float
    mapped_command: Optional[str] = None
    priority: EventPriority = EventPriority.HIGH


class PerformMouseClickEventData(BaseEvent):
    """Request to perform a mouse click.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
        source: Source of the click request.
    """

    x: int
    y: int
    source: Optional[str] = "unknown"
    priority: EventPriority = EventPriority.CRITICAL


class MarkovPredictionEvent(BaseEvent):
    """Published when Markov chain predicts a command.

    Attributes:
        predicted_command: The predicted command text.
        confidence: Confidence probability (0.0-1.0).
        audio_id: ID of audio that triggered prediction.
    """

    predicted_command: str = Field(description="The predicted command text")
    confidence: float = Field(description="Confidence probability (0.0-1.0)")
    audio_id: int = Field(description="ID of the audio bytes that triggered this prediction")
    priority: EventPriority = EventPriority.CRITICAL


class MarkovPredictionFeedbackEvent(BaseEvent):
    """Feedback about Markov prediction accuracy.

    Attributes:
        predicted_command: The command that was predicted.
        actual_command: The command that was actually recognized.
        was_correct: True if prediction matched actual command.
        source: Source of actual command (stt or sound).
    """

    predicted_command: str = Field(description="The command that was predicted")
    actual_command: str = Field(description="The command that was actually recognized")
    was_correct: bool = Field(description="True if prediction matched actual command")
    source: str = Field(description="Source of actual command: 'stt' or 'sound'")
    priority: EventPriority = EventPriority.LOW  # Background training feedback


class SettingsChangedEvent(BaseEvent):
    """Event published when runtime settings are changed.

    Attributes:
        updated_settings: Dictionary of setting paths to new values.
        all_settings: Dictionary of all effective settings.
    """

    updated_settings: Dict[str, Any] = Field(description="Dictionary of setting paths to new values")
    all_settings: Dict[str, Any] = Field(description="Dictionary of all effective settings")
    priority: EventPriority = EventPriority.HIGH


class CommandTextRecognizedEvent(TextRecognizedEvent):
    """Text recognized in command mode.

    Typically from a faster STT engine like Vosk.
    Used for application commands and detecting stop words during dictation.
    """


class DictationTextRecognizedEvent(TextRecognizedEvent):
    """Text recognized in dictation mode (e.g. Moonshine streaming or segment batch)."""


class AudioDeviceErrorEvent(BaseEvent):
    """Event published when the default input device is lost or capture fails.

    Attributes:
        error_message: Full user-facing message for the warning dialog (restart urged).
    """

    error_message: str = Field(description="User-facing message for the warning dialog")
    priority: EventPriority = EventPriority.HIGH
