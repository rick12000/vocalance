from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from vocalance.app.events.base_event import BaseEvent

DictationModifierId = Literal["upper", "capitals", "camel", "snake", "spelling", "kebab", "diminish", "strip", "numeral"]


class DictationStatusChangedEvent(BaseEvent):
    """Event fired when dictation status changes for UI synchronization."""

    is_active: bool = Field(description="Whether dictation is currently active")
    mode: Literal["inactive", "standard", "type", "smart", "visual", "hidden", "amend"] = Field(
        description="Current dictation mode"
    )
    show_ui: bool = Field(default=False, description="Whether to show the dictation UI indicator")
    stop_command: Optional[str] = Field(default=None, description="The command to stop this dictation mode")


class DictationModeDisableOthersEvent(BaseEvent):
    """Event fired to disable other speech/sound processing during dictation."""

    dictation_mode_active: bool = Field(description="Whether dictation mode is active, disabling other processing")
    dictation_mode: Literal["inactive", "standard", "type", "smart", "visual", "hidden", "amend"]


class DictationSessionEvent(BaseEvent):
    """Event fired when a dictation session starts or stops."""

    mode: Literal["smart", "visual", "hidden", "amend"] = Field(description="Which dictation mode is starting or stopping")
    state: Literal["started", "stopped"] = Field(description="Whether the session is starting or stopping")
    raw_text: Optional[str] = Field(default=None, description="Raw text before LLM (for smart/amend stopped)")
    accumulated_text: Optional[str] = Field(default=None, description="Accumulated text to be pasted (for visual/hidden stopped)")


class LLMProcessingStartedEvent(BaseEvent):
    """Event fired when LLM processing begins."""

    raw_text: str = Field(description="Raw dictated text to be processed")
    agentic_prompt: str = Field(description="The agentic prompt being used")
    session_id: Optional[str] = Field(default=None, description="LLM session identifier for correlating ready/token events")


class LLMProcessingCompletedEvent(BaseEvent):
    """Event fired when LLM processing is completed."""

    processed_text: str = Field(description="LLM processed text")
    agentic_prompt: str = Field(description="The agentic prompt that was used")


class LLMProcessingFailedEvent(BaseEvent):
    """Event published when LLM processing fails."""

    error_message: str = Field(description="Error message describing what went wrong")
    original_text: str = Field(description="Original text that failed to process")


class LLMTokenGeneratedEvent(BaseEvent):
    """Event fired when a token is generated during LLM streaming."""

    token: str = Field(description="The generated token from LLM streaming")
    session_id: str = Field(description="Matches LLMProcessingStartedEvent.session_id for UI correlation")


class SmartDictationRemoveCharactersEvent(BaseEvent):
    """Event fired when characters should be removed from smart dictation UI."""

    count: int = Field(description="Number of characters to remove from end of dictation text")


class LLMProcessingReadyEvent(BaseEvent):
    """Event fired when UI is ready to receive LLM tokens."""

    session_id: str = Field(description="Session ID to match processing requests")


class AgenticPromptUpdatedEvent(BaseEvent):
    """Event fired when the current agentic prompt is updated."""

    prompt: str = Field(description="The new agentic prompt")
    prompt_id: str = Field(description="Unique identifier for the prompt")


class AgenticPromptListUpdatedEvent(BaseEvent):
    """Event fired when the list of agentic prompts is updated."""

    prompts: List[Dict[str, Any]] = Field(description="List of available agentic prompts with their metadata")


class PartialDictationTextEvent(BaseEvent):
    """Event fired for partial (unstable) streaming dictation text."""

    text: str = Field(description="Partial transcription text (unstable)")
    segment_id: str = Field(description="Unique segment identifier for tracking updates")


class FinalDictationTextEvent(BaseEvent):
    """Event fired for finalized streaming dictation text."""

    text: str = Field(description="Final transcription text (stable)")
    segment_id: str = Field(description="Unique segment identifier")


class DictationAliasListUpdatedEvent(BaseEvent):
    """Event fired when the dictation alias list is updated."""

    aliases: Dict[str, str] = Field(description="Current alias mappings (key -> substitution)")


AgenticPromptUiOp = Literal["add", "select", "delete", "edit", "publish_state"]


class AgenticPromptUiOperationEvent(BaseEvent):
    """UI-originated agentic prompt mutations; handled by ``AgenticPromptService``."""

    op: AgenticPromptUiOp
    name: str = ""
    prompt_text: str = ""
    prompt_id: str = ""
    text: str = ""


DictationAliasUiOp = Literal["add", "update", "delete", "refresh_list"]


class DictationAliasUiOperationEvent(BaseEvent):
    """UI-originated dictation alias mutations; handled by ``DictationAliasService``."""

    op: DictationAliasUiOp
    key: str = ""
    value: str = ""


class DictationStopWordDetectedEvent(BaseEvent):
    """Event fired when the dictation stop word is detected."""

    mode: Literal["standard", "type", "smart", "visual", "hidden", "amend"] = Field(description="Current dictation mode")


class DictationModifierPhraseEvent(BaseEvent):
    """Published when Vosk recognizes a configured modifier phrase while dictation is active."""

    modifier_id: DictationModifierId = Field(description="Which modifier phrase matched")
    raw_recognized_text: str = Field(default="", description="Raw Vosk text for logging")


class DictationModifierStateChangedEvent(BaseEvent):
    """Published when the active dictation modifier is toggled, switched, or cleared."""

    active: bool = Field(description="True when a modifier is now active")
    active_modifiers: set[DictationModifierId] = Field(default_factory=set, description="Active modifiers, if any")
    display_label: str = Field(default="", description="Human-readable label for the chip, e.g. 'Upper'")
