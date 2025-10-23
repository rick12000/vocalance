from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from iris.app.events.base_event import BaseEvent, EventPriority


class DictationStatusChangedEvent(BaseEvent):
    """Event fired when dictation status changes for UI updates"""

    is_active: bool = Field(description="Whether dictation is currently active")
    mode: Literal["inactive", "standard", "type", "smart"] = Field(description="Current dictation mode")
    show_ui: bool = Field(default=False, description="Whether to show the dictation UI indicator")
    stop_command: Optional[str] = Field(default=None, description="The command to stop this dictation mode")
    priority: EventPriority = EventPriority.LOW


class DictationModeDisableOthersEvent(BaseEvent):
    """Event fired to disable other speech/sound processing during dictation"""

    dictation_mode_active: bool = Field(description="Whether dictation mode is active, disabling other processing")
    dictation_mode: Literal["inactive", "standard", "type", "smart"]
    priority: EventPriority = EventPriority.CRITICAL


class AudioModeChangeRequestEvent(BaseEvent):
    """Event to request audio mode change between command and dictation"""

    mode: Literal["command", "dictation"] = Field(description="Target audio mode")
    reason: str = Field(description="Reason for the mode change")
    priority: EventPriority = EventPriority.CRITICAL


class SmartDictationStartedEvent(BaseEvent):
    """Event fired when smart dictation mode is activated"""

    mode: Literal["smart"] = "smart"
    priority: EventPriority = EventPriority.NORMAL


class SmartDictationStoppedEvent(BaseEvent):
    """Event fired when smart dictation mode is deactivated"""

    mode: Literal["smart"] = "smart"
    raw_text: str = Field(description="Raw text that was dictated before LLM processing")
    priority: EventPriority = EventPriority.NORMAL


class LLMProcessingStartedEvent(BaseEvent):
    """Event fired when LLM processing begins"""

    raw_text: str = Field(description="Raw dictated text to be processed")
    agentic_prompt: str = Field(description="The agentic prompt being used")
    session_id: Optional[str] = None
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


class SmartDictationRemoveCharactersEvent(BaseEvent):
    """Event fired when characters should be removed from smart dictation UI (for period removal)"""

    count: int = Field(description="Number of characters to remove from end of dictation text")
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

    prompts: List[Dict[str, Any]] = Field(description="List of available agentic prompts with their metadata")
    priority: EventPriority = EventPriority.LOW


class AgenticPromptActionRequest(BaseEvent):
    """Event for requesting agentic prompt actions"""

    action: Literal["add_prompt", "delete_prompt", "edit_prompt", "set_current_prompt", "get_prompts"] = Field(
        description="The action to perform"
    )
    name: Optional[str] = None
    text: Optional[str] = None
    prompt_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
