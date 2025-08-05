from pydantic import Field
from typing import Optional, Dict, Any
from iris.config.command_types import (
    DictationCommandType, AutomationCommandType, MarkCommandType, 
    GridCommandType, SoundCommandType, AnyCommand
)
from iris.events.base_event import BaseEvent, EventPriority

# ============================================================================
# BASE COMMAND EVENT CLASSES
# ============================================================================

class BaseCommandEvent(BaseEvent):
    """Base class for all command events"""
    source: Optional[str] = Field(default=None, description="Source of the command (e.g., 'speech', 'sound:label')")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context information")
    priority: EventPriority = EventPriority.NORMAL

# ============================================================================
# SPECIFIC COMMAND EVENTS
# ============================================================================

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

# ============================================================================
# ERROR AND NO MATCH EVENTS
# ============================================================================

class CommandNoMatchEvent(BaseCommandEvent):
    """Event published when no command matches the input text"""
    attempted_parsers: list[str] = Field(default_factory=list, description="List of parsers that were attempted")

class CommandParseErrorEvent(BaseCommandEvent):
    """Event published when an error occurs during command parsing"""
    error_message: str = Field(..., description="Description of the parsing error")
    attempted_parser: Optional[str] = Field(default=None, description="The parser that encountered the error")

