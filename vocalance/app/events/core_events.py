from typing import Any, Dict, Literal, Optional

from pydantic import Field

from vocalance.app.events.base_event import BaseEvent, EventPriority


class RecordingTriggerEvent(BaseEvent):
    """Unified event for starting or stopping audio recording.

    Attributes:
        trigger: Recording action (start or stop).
    """

    trigger: Literal["start", "stop"] = Field(..., description="Recording action")
    priority: EventPriority = EventPriority.CRITICAL


class CommandAudioSegmentReadyEvent(BaseEvent):
    """Audio segment ready for command mode processing.

    Attributes:
        audio_bytes: Audio data.
        sample_rate: Sample rate of audio.
    """

    audio_bytes: bytes
    sample_rate: int
    priority: EventPriority = EventPriority.HIGH


class DictationAudioSegmentReadyEvent(BaseEvent):
    """Audio segment ready for dictation mode processing.

    Attributes:
        audio_bytes: Audio data.
        sample_rate: Sample rate of audio.
    """

    audio_bytes: bytes
    sample_rate: int
    priority: EventPriority = EventPriority.HIGH


class AudioChunkEvent(BaseEvent):
    """Continuous audio chunk stream from recorder (base unit: 50ms).

    Published continuously by AudioRecorder at fixed intervals. Downstream
    listeners (CommandAudioListener, DictationAudioListener) accumulate these
    chunks and apply their own VAD logic and silence timeouts.

    Attributes:
        audio_chunk: Raw audio data for this chunk (numpy int16 format as bytes).
        sample_rate: Sample rate of audio.
        timestamp: Timestamp when chunk was captured.
    """

    audio_chunk: bytes
    sample_rate: int
    timestamp: float = Field(description="Timestamp when chunk was captured")
    priority: EventPriority = EventPriority.CRITICAL  # High priority for real-time streaming


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


class ProcessCommandPhraseEvent(BaseEvent):
    """Command phrase ready for processing.

    Attributes:
        phrase: Command phrase text.
        source: Source of the command.
        context: Additional context data.
    """

    phrase: str
    source: Optional[str] = None
    context: Optional[Any] = None
    priority: EventPriority = EventPriority.HIGH


class CommandExecutedStatusEvent(BaseEvent):
    """Status update after command execution.

    Attributes:
        command: Command data.
        success: Whether execution succeeded.
        message: Status message.
        source: Source of the command.
    """

    command: Dict[str, Any]
    success: bool
    message: Optional[str] = None
    source: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL


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


class ClickLoggedEventData(BaseEvent):
    """Click logged for tracking.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
        timestamp: Timestamp of click.
    """

    x: int
    y: int
    timestamp: float
    priority: EventPriority = EventPriority.LOW


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


class SettingsResponseEvent(BaseEvent):
    """Event containing current effective settings.

    Attributes:
        settings: Dictionary of current settings.
    """

    settings: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL


class DynamicSettingsUpdatedEvent(BaseEvent):
    """Event published when runtime settings are changed.

    Attributes:
        updated_settings: Dictionary of setting paths to new values.
    """

    updated_settings: Dict[str, Any] = Field(description="Dictionary of setting paths to new values")
    priority: EventPriority = EventPriority.HIGH


class CommandTextRecognizedEvent(TextRecognizedEvent):
    """Text recognized in command mode.

    Typically from a faster STT engine like Vosk.
    Used for application commands and detecting stop words during dictation.
    """


class DictationTextRecognizedEvent(TextRecognizedEvent):
    """Text recognized in dictation mode.

    Typically from a more accurate STT engine like Whisper.
    """


class STTProcessingStartedEvent(BaseEvent):
    """Event indicating STT processing has started.

    Attributes:
        engine: STT engine being used.
        mode: Processing mode (command or dictation).
        audio_size_bytes: Size of audio being processed.
    """

    engine: str = Field(description="STT engine being used")
    mode: str = Field(description="Processing mode (command, dictation)")
    audio_size_bytes: int = Field(description="Size of audio being processed")
    priority: EventPriority = EventPriority.NORMAL


class STTProcessingCompletedEvent(BaseEvent):
    """Event indicating STT processing has completed.

    Attributes:
        engine: STT engine that was used.
        mode: Processing mode (command or dictation).
        processing_time_ms: Processing time in milliseconds.
        text_length: Length of recognized text.
    """

    engine: str = Field(description="STT engine that was used")
    mode: str = Field(description="Processing mode (command, dictation)")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    text_length: int = Field(description="Length of recognized text")
    priority: EventPriority = EventPriority.NORMAL


class GetMainWindowHandleRequest(BaseEvent):
    """Request the main window handle (e.g., HWND)."""

    priority: EventPriority = EventPriority.CRITICAL


class GetMainWindowHandleResponse(BaseEvent):
    """Response carrying the main window handle.

    Attributes:
        hwnd: Window handle or None if error.
        error_message: Error message if retrieval failed.
    """

    hwnd: Optional[int] = None
    error_message: Optional[str] = None
    priority: EventPriority = EventPriority.CRITICAL


class ApplicationShutdownRequestedEvent(BaseEvent):
    """Event published when application shutdown is requested.

    Can be triggered by user closing windows, system signals, or critical errors.

    Attributes:
        reason: Reason for shutdown request.
        source: Source of shutdown request (startup_window, main_window, signal, etc.).
    """

    reason: str = Field(description="Reason for shutdown request")
    source: str = Field(description="Source of shutdown request (e.g., 'startup_window', 'main_window', 'signal')")
    priority: EventPriority = EventPriority.CRITICAL
