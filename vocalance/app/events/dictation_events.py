from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from vocalance.app.events.base_event import BaseEvent, EventPriority

DictationModifierId = Literal["upper", "capitals", "camel", "snake", "spelling", "kebab", "diminish", "strip"]


class DictationStatusChangedEvent(BaseEvent):
    """Event fired when dictation status changes for UI synchronization.

    Published by DictationCoordinator when dictation mode is activated, deactivated,
    or transitions between modes. Drives UI indicator visibility and stop command display.

    Attributes:
        is_active: True if any dictation mode is currently active.
        mode: Current dictation mode type.
        show_ui: True if dictation UI indicator should be visible.
        stop_command: Voice phrase to stop current dictation mode, or None if inactive.
    """

    is_active: bool = Field(description="Whether dictation is currently active")
    mode: Literal["inactive", "standard", "type", "smart", "visual", "hidden", "amend"] = Field(
        description="Current dictation mode"
    )
    show_ui: bool = Field(default=False, description="Whether to show the dictation UI indicator")
    stop_command: Optional[str] = Field(default=None, description="The command to stop this dictation mode")
    priority: EventPriority = EventPriority.LOW


class DictationModeDisableOthersEvent(BaseEvent):
    """Event fired to disable other speech/sound processing during dictation"""

    dictation_mode_active: bool = Field(description="Whether dictation mode is active, disabling other processing")
    dictation_mode: Literal["inactive", "standard", "type", "smart", "visual", "hidden", "amend"]
    priority: EventPriority = EventPriority.CRITICAL


class AudioModeChangeRequestEvent(BaseEvent):
    """Event to request audio mode change between command and dictation"""

    mode: Literal["command", "dictation"] = Field(description="Target audio mode")
    reason: str = Field(description="Reason for the mode change")
    priority: EventPriority = EventPriority.CRITICAL


class DictationSessionEvent(BaseEvent):
    """Event fired when a dictation session starts or stops."""

    mode: Literal["smart", "visual", "hidden", "amend"] = Field(description="Which dictation mode is starting or stopping")
    state: Literal["started", "stopped"] = Field(description="Whether the session is starting or stopping")
    raw_text: Optional[str] = Field(default=None, description="Raw text before LLM (for smart/amend stopped)")
    accumulated_text: Optional[str] = Field(default=None, description="Accumulated text to be pasted (for visual/hidden stopped)")
    priority: EventPriority = EventPriority.NORMAL


class LLMProcessingStartedEvent(BaseEvent):
    """Event fired when LLM processing begins."""

    raw_text: str = Field(description="Raw dictated text to be processed")
    agentic_prompt: str = Field(description="The agentic prompt being used")
    session_id: Optional[str] = Field(
        default=None,
        description="LLM session identifier for correlating ready/token events, if set",
    )
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


class PartialDictationTextEvent(BaseEvent):
    """Event fired for partial (unstable) streaming dictation text.

    Emitted during streaming dictation (smart/visual modes) when text prediction
    is still being refined. UI should display this as gray/tentative text that
    may change with subsequent predictions.

    Attributes:
        text: Partial transcription text that may still change.
        segment_id: Unique identifier for this text segment.
    """

    text: str = Field(description="Partial transcription text (unstable)")
    segment_id: str = Field(description="Unique segment identifier for tracking updates")
    priority: EventPriority = EventPriority.HIGH


class FinalDictationTextEvent(BaseEvent):
    """Event fired for finalized streaming dictation text.

    Emitted during streaming dictation (smart/visual modes) when text prediction
    has stabilized (4+ consecutive identical predictions at 200ms intervals, i.e.,
    800ms of stable output). UI should display this as white/permanent text that
    will no longer be edited.

    Attributes:
        text: Final transcription text that will not change.
        segment_id: Unique identifier for this finalized segment.
    """

    text: str = Field(description="Final transcription text (stable)")
    segment_id: str = Field(description="Unique segment identifier")
    priority: EventPriority = EventPriority.HIGH


class DictationAliasListUpdatedEvent(BaseEvent):
    """Event fired when the dictation alias list is updated.

    Published by DictationAliasService when aliases are added, updated, or deleted.
    UI should refresh the alias list display when receiving this event.
    """

    aliases: Dict[str, str] = Field(description="Current alias mappings (key -> substitution)")
    priority: EventPriority = EventPriority.LOW


class DictationStopWordDetectedEvent(BaseEvent):
    """Event fired when the dictation stop word is detected.

    Published by STT service when the stop trigger is recognized during dictation.
    UI should change the border color to orange to indicate the stop word was heard.
    """

    mode: Literal["standard", "type", "smart", "visual", "hidden", "amend"] = Field(description="Current dictation mode")
    priority: EventPriority = EventPriority.HIGH


class DictationModifierPhraseEvent(BaseEvent):
    """Published when Vosk recognizes a configured modifier phrase while dictation is active.

    Does not produce ``CommandTextRecognizedEvent``; the coordinator updates session state only.
    """

    modifier_id: DictationModifierId = Field(description="Which modifier phrase matched")
    raw_recognized_text: str = Field(default="", description="Raw Vosk text for logging")
    priority: EventPriority = EventPriority.HIGH


class DictationModifierStateChangedEvent(BaseEvent):
    """Published when the active dictation modifier is toggled, switched, or cleared."""

    active: bool = Field(description="True when a modifier is now active")
    active_modifiers: set[DictationModifierId] = Field(default_factory=set, description="Active modifiers, if any")
    display_label: str = Field(default="", description="Human-readable label for the chip, e.g. 'Upper'")
    priority: EventPriority = EventPriority.HIGH
