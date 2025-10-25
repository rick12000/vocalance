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
    """Base event for all parsed command types.

    Attributes:
        source: Source of the command (stt, sound, markov, etc.).
        context: Optional context data about the command.
    """

    source: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.NORMAL


class DictationCommandParsedEvent(BaseCommandEvent):
    """Event published when a dictation command is parsed.

    Attributes:
        command: The parsed dictation command.
    """

    command: DictationCommandType = Field(..., description="The parsed dictation command")


class AutomationCommandParsedEvent(BaseCommandEvent):
    """Event published when an automation command is parsed.

    Attributes:
        command: The parsed automation command.
    """

    command: AutomationCommandType = Field(..., description="The parsed automation command")


class MarkCommandParsedEvent(BaseCommandEvent):
    """Event published when a mark command is parsed.

    Attributes:
        command: The parsed mark command.
    """

    command: MarkCommandType = Field(..., description="The parsed mark command")


class GridCommandParsedEvent(BaseCommandEvent):
    """Event published when a grid command is parsed.

    Attributes:
        command: The parsed grid command.
    """

    command: GridCommandType = Field(..., description="The parsed grid command")


class SoundCommandParsedEvent(BaseCommandEvent):
    """Event published when a sound command is parsed.

    Attributes:
        command: The parsed sound command.
    """

    command: SoundCommandType = Field(..., description="The parsed sound command")


class CommandNoMatchEvent(BaseCommandEvent):
    """Event published when no command matches the input text.

    Attributes:
        attempted_parsers: List of parsers that were attempted.
    """

    attempted_parsers: List[str] = []


class CommandParseErrorEvent(BaseCommandEvent):
    """Event published when an error occurs during command parsing.

    Attributes:
        error_message: Description of the parsing error.
        attempted_parser: The parser that encountered the error.
    """

    error_message: str = Field(..., description="Description of the parsing error")
    attempted_parser: Optional[str] = Field(default=None, description="The parser that encountered the error")
