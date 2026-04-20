from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from vocalance.app.events.dictation_events import DictationModifierId


class DictationMode(StrEnum):
    INACTIVE = "inactive"
    STANDARD = "standard"
    SMART = "smart"
    TYPE = "type"
    VISUAL = "visual"
    HIDDEN = "hidden"
    AMEND = "amend"


class DictationState(StrEnum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING_LLM = "processing_llm"
    SHUTTING_DOWN = "shutting_down"


class DictationSession(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    mode: DictationMode
    start_time: float
    accumulated_text: str = ""
    last_text_time: float | None = None
    is_first_segment: bool = True
    active_modifiers: set[DictationModifierId] = Field(default_factory=set)
    explicit_modifiers: set[DictationModifierId] = Field(default_factory=set)


class LLMSession(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    raw_text: str
    agentic_prompt: str
    clipboard_text: str | None = None
