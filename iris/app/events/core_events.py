from iris.app.events.base_event import BaseEvent, EventPriority
from pydantic import Field, BaseModel
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
import uuid
import numpy as np


class RecordingTriggerEvent(BaseEvent):
    """Unified event for starting or stopping audio recording"""
    trigger: Literal["start", "stop"] = Field(..., description="Recording action")
    priority: EventPriority = EventPriority.CRITICAL

class CommandAudioSegmentReadyEvent(BaseEvent):
    audio_bytes: bytes
    sample_rate: int
    priority: EventPriority = EventPriority.HIGH

class DictationAudioSegmentReadyEvent(BaseEvent):
    audio_bytes: bytes
    sample_rate: int
    priority: EventPriority = EventPriority.HIGH

class AudioDetectedEvent(BaseEvent):
    """Published immediately when audio above threshold is first detected"""
    timestamp: float = Field(description="Timestamp when audio was detected")
    priority: EventPriority = EventPriority.CRITICAL

class ProcessAudioChunkForSoundRecognitionEvent(BaseEvent):
    audio_chunk: bytes
    sample_rate: int = 16000
    priority: EventPriority = EventPriority.HIGH

class TextRecognizedEvent(BaseEvent):
    text: str
    confidence: float = 1.0
    engine: str = "unknown"
    processing_time_ms: float = 0.0
    mode: str = "command"
    priority: EventPriority = EventPriority.HIGH

class ProcessCommandPhraseEvent(BaseEvent):
    phrase: str
    source: Optional[str] = None
    context: Optional[Any] = None
    priority: EventPriority = EventPriority.HIGH

class CommandExecutedStatusEvent(BaseEvent):
    command: Dict[str, Any]
    success: bool
    message: Optional[str] = None
    source: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL


class CustomSoundRecognizedEvent(BaseEvent):
    label: str
    confidence: float
    mapped_command: Optional[str] = None
    priority: EventPriority = EventPriority.HIGH


class PerformMouseClickEventData(BaseEvent):
    x: int
    y: int
    source: Optional[str] = "unknown"
    priority: EventPriority = EventPriority.CRITICAL

class ClickLoggedEventData(BaseEvent):
    x: int
    y: int
    timestamp: float
    priority: EventPriority = EventPriority.LOW


class MarkovPredictionEvent(BaseEvent):
    """Published when Markov chain predicts a command with high confidence"""
    predicted_command: str = Field(description="The predicted command text")
    confidence: float = Field(description="Confidence probability (0.0-1.0)")
    audio_id: int = Field(description="ID of the audio bytes that triggered this prediction")
    priority: EventPriority = EventPriority.CRITICAL


class MarkovPredictionFeedbackEvent(BaseEvent):
    """Feedback from command parser to Markov predictor about prediction accuracy"""
    predicted_command: str = Field(description="The command that was predicted")
    actual_command: str = Field(description="The command that was actually recognized")
    was_correct: bool = Field(description="True if prediction matched actual command")
    source: str = Field(description="Source of actual command: 'stt' or 'sound'")
    priority: EventPriority = EventPriority.NORMAL


class SettingsResponseEvent(BaseEvent):
    """Event containing current effective settings for UI and services"""
    settings: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL


class DynamicSettingsUpdatedEvent(BaseEvent):
    """Event published when settings that can be updated at runtime are changed"""
    updated_settings: Dict[str, Any] = Field(description="Dictionary of setting paths to new values")
    priority: EventPriority = EventPriority.HIGH


class CommandTextRecognizedEvent(TextRecognizedEvent):
    """
    Published when the STT service recognizes text in command mode.
    This is typically from a faster, less accurate engine like Vosk.
    It's used for application commands and for detecting stop words during dictation.
    """
    pass


class DictationTextRecognizedEvent(TextRecognizedEvent):
    """
    Published when the STT service recognizes text in dictation mode.
    This is typically from a more accurate, slower engine like Whisper.
    """
    pass


class STTProcessingStartedEvent(BaseEvent):
    """Event indicating STT processing has started"""
    engine: str = Field(description="STT engine being used")
    mode: str = Field(description="Processing mode (command, dictation)")
    audio_size_bytes: int = Field(description="Size of audio being processed")
    priority: EventPriority = EventPriority.NORMAL


class STTProcessingCompletedEvent(BaseEvent):
    """Event indicating STT processing has completed"""
    engine: str = Field(description="STT engine that was used")
    mode: str = Field(description="Processing mode (command, dictation)")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    text_length: int = Field(description="Length of recognized text")
    priority: EventPriority = EventPriority.NORMAL


class GetMainWindowHandleRequest(BaseEvent):
    """Event to request the main window handle (e.g., HWND)."""
    priority: EventPriority = EventPriority.CRITICAL


class GetMainWindowHandleResponse(BaseEvent):
    """Event carrying the main window handle or an error if it couldn't be retrieved."""
    hwnd: Optional[int] = None
    error_message: Optional[str] = None
    priority: EventPriority = EventPriority.CRITICAL