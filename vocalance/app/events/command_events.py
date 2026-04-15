from typing import Any, Dict, Optional

from pydantic import Field

from vocalance.app.config.command_types import (
    AutomationCommandType,
    DictationCommandType,
    GridCommandType,
    MarkCommandType,
    SystemControlCommandType,
)
from vocalance.app.events.base_event import BaseEvent


class BaseCommandEvent(BaseEvent):
    """Base event for all parsed command types with source tracking and context."""

    source: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class DictationCommandParsedEvent(BaseCommandEvent):
    """Event carrying a parsed dictation-mode command."""

    command: DictationCommandType = Field(..., description="The parsed dictation command")


class AutomationCommandParsedEvent(BaseCommandEvent):
    """Event carrying a parsed automation (hotkey/click/scroll) command."""

    command: AutomationCommandType = Field(..., description="The parsed automation command")


class MarkCommandParsedEvent(BaseCommandEvent):
    """Event carrying a parsed mark system command."""

    command: MarkCommandType = Field(..., description="The parsed mark command")


class GridCommandParsedEvent(BaseCommandEvent):
    """Event carrying a parsed grid overlay command."""

    command: GridCommandType = Field(..., description="The parsed grid command")


class SystemControlCommandParsedEvent(BaseCommandEvent):
    """Event carrying a parsed pause/resume (system control) command."""

    command: SystemControlCommandType = Field(..., description="The parsed system control command")
