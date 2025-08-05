from iris.events.base_event import BaseEvent, EventPriority
from pydantic import Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import numpy as np

# === AUDIO EVENTS ===

class StartRecordingCommand(BaseEvent):
    priority: EventPriority = EventPriority.CRITICAL

class StopRecordingCommand(BaseEvent):
    priority: EventPriority = EventPriority.CRITICAL

class RequestAudioSampleForTrainingCommand(BaseEvent):
    label: str
    sample_idx: int
    duration_sec: float = 3.0
    priority: EventPriority = EventPriority.NORMAL

class CommandAudioSegmentReadyEvent(BaseEvent):
    audio_bytes: bytes
    sample_rate: int
    priority: EventPriority = EventPriority.HIGH

class DictationAudioSegmentReadyEvent(BaseEvent):
    audio_bytes: bytes
    sample_rate: int
    priority: EventPriority = EventPriority.HIGH

class AudioSampleForTrainingReadyEvent(BaseEvent):
    audio_bytes: bytes
    sample_rate: int
    label: str
    sample_idx: int
    priority: EventPriority = EventPriority.NORMAL

class AudioRecordingStateEvent(BaseEvent):
    is_recording: bool
    priority: EventPriority = EventPriority.LOW

# VADSegmentDetectedEvent removed - unused in codebase

class ProcessAudioChunkForSoundRecognitionEvent(BaseEvent):
    audio_chunk: bytes
    sample_rate: int = 16000
    priority: EventPriority = EventPriority.HIGH

# === COMMAND EVENTS ===

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

class ParsedCommandReadyEvent(BaseEvent):
    parsed_command: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL

class CommandParseFailedEvent(BaseEvent):
    phrase: str
    error_message: str
    priority: EventPriority = EventPriority.NORMAL

class CommandExecutedStatusEvent(BaseEvent):
    command: Dict[str, Any]
    success: bool
    message: Optional[str] = None
    source: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL

# StopFurtherProcessingForSegmentEvent removed - imported but never used


# STT processing events moved to stt_events.py to avoid duplication

# === SOUND RECOGNITION EVENTS ===

class CustomSoundRecognizedEvent(BaseEvent):
    label: str
    confidence: float
    mapped_command: Optional[str] = None
    priority: EventPriority = EventPriority.HIGH

class SoundModelUpdatedEvent(BaseEvent):
    label: str
    model_path: str
    priority: EventPriority = EventPriority.LOW

# === CLICK EVENTS ===

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