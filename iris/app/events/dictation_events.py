from iris.app.events.base_event import BaseEvent, EventPriority
from pydantic import Field
from typing import Optional, List

class DictationStartedEvent(BaseEvent):
    """Event fired when dictation mode is activated"""
    mode: str = Field(description="Type of dictation mode: 'continuous' or 'type'")
    priority: EventPriority = EventPriority.NORMAL

class DictationStoppedEvent(BaseEvent):
    """Event fired when dictation mode is deactivated"""
    mode: str = Field(description="Type of dictation mode that was stopped")
    total_text: str = Field(description="Complete text that was dictated")
    priority: EventPriority = EventPriority.NORMAL

class DictationTextEvent(BaseEvent):
    """Event fired when new text is recognized during dictation"""
    text: str = Field(description="The recognized text to be typed")
    mode: str = Field(description="Type of dictation mode: 'continuous' or 'type'")
    is_incremental: bool = Field(default=True, description="Whether this is incremental text or complete replacement")
    priority: EventPriority = EventPriority.HIGH

# DictationTimeoutEvent removed - unused in codebase

class DictationErrorEvent(BaseEvent):
    """Event fired when an error occurs during dictation"""
    error_message: str = Field(description="Description of the error")
    mode: str = Field(description="Type of dictation mode where error occurred")
    priority: EventPriority = EventPriority.NORMAL

class DictationStatusChangedEvent(BaseEvent):
    """Event fired when dictation status changes for UI updates"""
    is_active: bool = Field(description="Whether dictation is currently active")
    mode: str = Field(description="Current dictation mode: 'inactive', 'continuous', 'type', or 'smart'")
    show_ui: bool = Field(default=False, description="Whether to show the dictation UI indicator")
    stop_command: Optional[str] = Field(default=None, description="The command to stop this dictation mode")
    priority: EventPriority = EventPriority.LOW

class DictationModeDisableOthersEvent(BaseEvent):
    """Event fired to disable other speech/sound processing during dictation"""
    dictation_mode_active: bool = Field(description="Whether dictation mode is active, disabling other processing")
    dictation_mode: str = Field(description="Current dictation mode")
    priority: EventPriority = EventPriority.CRITICAL

# New streamlined dictation events
class StandardDictationEnabledEvent(BaseEvent):
    """Event fired when standard dictation is enabled (green trigger word detected)"""
    trigger_word: str = Field(description="The trigger word that was detected")
    priority: EventPriority = EventPriority.NORMAL

class StandardDictationDisabledEvent(BaseEvent):
    """Event fired when standard dictation is disabled (amber stop word detected)"""
    stop_word: str = Field(description="The stop word that was detected")
    priority: EventPriority = EventPriority.NORMAL

class SmartDictationEnabledEvent(BaseEvent):
    """Event fired when smart dictation is enabled (smart green trigger detected)"""
    trigger_word: str = Field(description="The trigger word that was detected")
    priority: EventPriority = EventPriority.NORMAL

class TypeDictationEnabledEvent(BaseEvent):
    """Event fired when type dictation is enabled (type trigger detected)"""
    trigger_word: str = Field(description="The trigger word that was detected")
    priority: EventPriority = EventPriority.NORMAL

class AudioModeChangeRequestEvent(BaseEvent):
    """Event to request audio mode change between command and dictation"""
    mode: str = Field(description="Target audio mode: 'command' or 'dictation'")
    reason: str = Field(description="Reason for the mode change")
    priority: EventPriority = EventPriority.CRITICAL

# Smart dictation events
class SmartDictationStartedEvent(BaseEvent):
    """Event fired when smart dictation mode is activated"""
    mode: str = Field(default="smart", description="Smart dictation mode")
    priority: EventPriority = EventPriority.NORMAL

class SmartDictationStoppedEvent(BaseEvent):
    """Event fired when smart dictation mode is deactivated"""
    mode: str = Field(default="smart", description="Smart dictation mode")
    raw_text: str = Field(description="Raw text that was dictated before LLM processing")
    priority: EventPriority = EventPriority.NORMAL

class LLMProcessingStartedEvent(BaseEvent):
    """Event fired when LLM processing begins"""
    raw_text: str = Field(description="Raw dictated text to be processed")
    agentic_prompt: str = Field(description="The agentic prompt being used")
    session_id: Optional[str] = Field(default=None, description="Session ID for coordination")
    priority: EventPriority = EventPriority.NORMAL

class LLMProcessingCompletedEvent(BaseEvent):
    """Event fired when LLM processing is completed"""
    processed_text: str = Field(description="LLM processed text")
    agentic_prompt: str = Field(description="The agentic prompt that was used")
    priority: EventPriority = EventPriority.NORMAL

class LLMProcessingFailedEvent(BaseEvent):
    """Event published when LLM processing fails"""
    error_message: str = Field(description="Error message describing what went wrong")
    original_text: str = Field(description="Original text that failed to process")
    priority: EventPriority = EventPriority.NORMAL

class LLMTokenGeneratedEvent(BaseEvent):
    """Event fired when a token is generated during LLM streaming"""
    token: str = Field(description="The generated token from LLM streaming")
    priority: EventPriority = EventPriority.HIGH

class SmartDictationTextDisplayEvent(BaseEvent):
    """Event fired when cleaned text should be displayed in smart dictation UI"""
    text: str = Field(description="Cleaned text to display in smart dictation UI")
    priority: EventPriority = EventPriority.HIGH

class LLMProcessingReadyEvent(BaseEvent):
    """Event fired when UI is ready to receive LLM tokens"""
    session_id: str = Field(description="Session ID to match processing requests")
    priority: EventPriority = EventPriority.HIGH

class AgenticPromptUpdatedEvent(BaseEvent):
    """Event fired when the current agentic prompt is updated"""
    prompt: str = Field(description="The new agentic prompt")
    prompt_id: str = Field(description="Unique identifier for the prompt")
    priority: EventPriority = EventPriority.LOW

class AgenticPromptListUpdatedEvent(BaseEvent):
    """Event fired when the list of agentic prompts is updated"""
    prompts: list = Field(description="List of available agentic prompts")
    priority: EventPriority = EventPriority.LOW

class AgenticPromptActionRequest(BaseEvent):
    """Event for requesting agentic prompt actions"""
    action: str = Field(description="The action to perform")
    name: Optional[str] = Field(default=None, description="Name for add_prompt action")
    text: Optional[str] = Field(default=None, description="Text for add_prompt action")
    prompt_id: Optional[str] = Field(default=None, description="Prompt ID for delete/set actions")
    priority: EventPriority = EventPriority.NORMAL