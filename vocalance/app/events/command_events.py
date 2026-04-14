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
    command: DictationCommandType = Field(..., description="The parsed dictation command")


class AutomationCommandParsedEvent(BaseCommandEvent):
    command: AutomationCommandType = Field(..., description="The parsed automation command")


class MarkCommandParsedEvent(BaseCommandEvent):
    command: MarkCommandType = Field(..., description="The parsed mark command")


class GridCommandParsedEvent(BaseCommandEvent):
    command: GridCommandType = Field(..., description="The parsed grid command")


class SystemControlCommandParsedEvent(BaseCommandEvent):
    command: SystemControlCommandType = Field(..., description="The parsed system control command")
