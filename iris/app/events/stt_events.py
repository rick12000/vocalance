"""
Events related to Speech-to-Text processing, providing more specific event types
for command and dictation results.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from iris.app.events.core_events import TextRecognizedEvent
from iris.app.events.base_event import BaseEvent, EventPriority

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


class STTEngineSwitch(BaseEvent):
    """Event to request switching STT engine"""
    target_engine: str = Field(description="Target STT engine (vosk, whisper)")
    reason: str = Field(description="Reason for switching (dictation_start, dictation_stop, manual)")
    dictation_mode: str = Field(default="", description="Current dictation mode if applicable")
    priority: EventPriority = EventPriority.NORMAL

class STTEngineSwitched(BaseEvent):
    """Event indicating STT engine has been switched"""
    previous_engine: str = Field(description="Previous STT engine")
    new_engine: str = Field(description="New STT engine")
    reason: str = Field(description="Reason for switching")
    priority: EventPriority = EventPriority.LOW

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

class SmartTimeoutUpdateEvent(BaseEvent):
    """Event to update audio recorder timeout based on recognized text"""
    recognized_text: str = Field(description="Partial or complete recognized text")
    suggested_timeout: float = Field(description="Suggested timeout in seconds")
    is_ambiguous: bool = Field(description="Whether the text is ambiguous and might continue")
    priority: EventPriority = EventPriority.LOW
# STTRequestEvent and STTResponseEvent removed - unused dataclass events

