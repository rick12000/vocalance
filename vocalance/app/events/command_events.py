from typing import Any, Dict, List, Optional

from pydantic import Field

from vocalance.app.config.command_types import (
    AutomationCommandType,
    DictationCommandType,
    GridCommandType,
    MarkCommandType,
    SoundCommandType,
)
from vocalance.app.events.base_event import BaseEvent, EventPriority


class BaseCommandEvent(BaseEvent):
    """Base class for all command events"""

    source: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.NORMAL


class DictationCommandParsedEvent(BaseCommandEvent):
    """Event published when a dictation command is parsed"""

    command: DictationCommandType = Field(..., description="The parsed dictation command")


class AutomationCommandParsedEvent(BaseCommandEvent):
    """Event published when an automation command is parsed"""

    command: AutomationCommandType = Field(..., description="The parsed automation command")


class MarkCommandParsedEvent(BaseCommandEvent):
    """Event published when a mark command is parsed"""

    command: MarkCommandType = Field(..., description="The parsed mark command")


class GridCommandParsedEvent(BaseCommandEvent):
    """Event published when a grid command is parsed"""

    command: GridCommandType = Field(..., description="The parsed grid command")


class SoundCommandParsedEvent(BaseCommandEvent):
    """Event published when a sound command is parsed"""

    command: SoundCommandType = Field(..., description="The parsed sound command")


class CommandNoMatchEvent(BaseCommandEvent):
    """Event published when no command matches the input text"""

    attempted_parsers: List[str] = []


class CommandParseErrorEvent(BaseCommandEvent):
    """Event published when an error occurs during command parsing"""

    error_message: str = Field(..., description="Description of the parsing error")
    attempted_parser: Optional[str] = Field(default=None, description="The parser that encountered the error")
