"""
Command Types Configuration

Defines all command types and data structures used throughout the application.
Streamlined for simplicity and proper OOP inheritance.
"""

from typing import Optional, Union
from pydantic import BaseModel, Field
from abc import ABC

# ============================================================================
# RESULT CLASSES
# ============================================================================

class ParseResult(BaseModel):
    """Base class for parsing results"""
    pass

class NoMatchResult(ParseResult):
    """Indicates no command was matched"""
    pass

class ErrorResult(ParseResult):
    """Indicates an error occurred during parsing"""
    error_message: str

# ============================================================================
# BASE COMMAND CLASSES
# ============================================================================

class BaseCommand(BaseModel, ABC):
    """Abstract base class for all commands"""
    
    class Config:
        arbitrary_types_allowed = True

# ============================================================================
# DICTATION COMMANDS
# ============================================================================

class DictationCommand(BaseCommand):
    """Base class for dictation-related commands"""
    pass

class DictationStartCommand(DictationCommand):
    """Command to start dictation mode"""
    trigger_type: str = Field(..., description="Type of dictation trigger used")

class DictationStopCommand(DictationCommand):
    """Command to stop dictation mode"""
    pass

class DictationTypeCommand(DictationCommand):
    """Command to enter type mode"""
    pass

class DictationSmartStartCommand(DictationCommand):
    """Command to start smart dictation mode"""
    pass

# ============================================================================
# AUTOMATION COMMANDS (PyAutoGUI)
# ============================================================================

class AutomationCommand(BaseCommand):
    """Base class for automation commands (PyAutoGUI)"""
    command_key: str = Field(..., description="The command phrase that triggers this action")
    action_type: str = Field(..., description="Type of action: 'hotkey', 'key', 'click', 'scroll'")
    action_value: str = Field(..., description="The action value")
    is_custom: bool = Field(default=False, description="Whether this is a custom user-defined command")
    short_description: str = Field(default="", description="Short description for UI")
    long_description: str = Field(default="", description="Detailed description")

    @property
    def display_description(self) -> str:
        """Get the appropriate description for display"""
        return self.short_description if self.short_description else self._generate_short_description()
    
    def _generate_short_description(self) -> str:
        """Generate a short description based on action type"""
        action_type_map = {
            "hotkey": "Hotkey",
            "key": "Key",
            "click": "Click", 
            "scroll": "Scroll",
            "custom": "Custom",
            "type": "Type"
        }
        return action_type_map.get(self.action_type, "Action")

class ExactMatchCommand(AutomationCommand):
    """Command for exact phrase matches that execute once"""
    pass

class ParameterizedCommand(AutomationCommand):
    """Command for parameterized actions with repeat count"""
    count: int = Field(default=1, description="Number of times to repeat the command")

# ============================================================================
# MARK COMMANDS
# ============================================================================

class MarkCommand(BaseCommand):
    """Base class for mark-related commands"""
    pass

class MarkCreateCommand(MarkCommand):
    """Command to create a new mark"""
    label: str = Field(..., description="The label for the new mark")
    x: float = Field(..., description="The x coordinate for the new mark")
    y: float = Field(..., description="The y coordinate for the new mark")

class MarkExecuteCommand(MarkCommand):
    """Command to execute a click at a mark location"""
    label: str = Field(..., description="The label of the mark to execute")

class MarkDeleteCommand(MarkCommand):
    """Command to delete a specific mark"""
    label: str = Field(..., description="The label of the mark to delete")

class MarkVisualizeCommand(MarkCommand):
    """Command to show/visualize all marks"""
    pass

class MarkResetCommand(MarkCommand):
    """Command to reset/delete all marks"""
    pass

class MarkVisualizeCancelCommand(MarkCommand):
    """Command to cancel mark visualization"""
    pass

# ============================================================================
# GRID COMMANDS
# ============================================================================

class GridCommand(BaseCommand):
    """Base class for grid-related commands"""
    pass

class GridShowCommand(GridCommand):
    """Command to show the grid overlay"""
    num_rects: Optional[int] = Field(default=None, description="Number of rectangles to show")

class GridSelectCommand(GridCommand):
    """Command to select a grid cell by number"""
    selected_number: int = Field(..., description="The number selected from the grid")

class GridCancelCommand(GridCommand):
    """Command to cancel/hide the grid"""
    pass

# ============================================================================
# SOUND MANAGEMENT COMMANDS
# ============================================================================

class SoundCommand(BaseCommand):
    """Base class for sound management commands"""
    pass

class SoundTrainCommand(SoundCommand):
    """Command to train a new sound"""
    sound_label: str = Field(..., description="The label for the sound to be trained")

class SoundDeleteCommand(SoundCommand):
    """Command to delete a trained sound"""
    sound_label: str = Field(..., description="The label of the sound to delete")

class SoundResetAllCommand(SoundCommand):
    """Command to reset all trained sounds"""
    pass

class SoundListAllCommand(SoundCommand):
    """Command to list all trained sounds"""
    pass

class SoundMapCommand(SoundCommand):
    """Command to map a sound to a command phrase"""
    sound_label: str = Field(..., description="The label of the sound to map")
    command_phrase: str = Field(..., description="The command phrase to map to the sound")

# ============================================================================
# TYPE ALIASES AND UNIONS
# ============================================================================

# Union types for different command categories
DictationCommandType = Union[
    DictationStartCommand, DictationStopCommand, DictationTypeCommand, DictationSmartStartCommand
]

AutomationCommandType = Union[
    ExactMatchCommand, ParameterizedCommand
]

MarkCommandType = Union[
    MarkCreateCommand, MarkExecuteCommand, MarkDeleteCommand, 
    MarkVisualizeCommand, MarkResetCommand, MarkVisualizeCancelCommand
]

GridCommandType = Union[
    GridShowCommand, GridSelectCommand, GridCancelCommand
]

SoundCommandType = Union[
    SoundTrainCommand, SoundDeleteCommand, SoundResetAllCommand, 
    SoundListAllCommand, SoundMapCommand
]

# Union of all command types
AnyCommand = Union[
    DictationCommandType, AutomationCommandType, MarkCommandType, 
    GridCommandType, SoundCommandType
]

# Union of all parsing results
ParseResultType = Union[AnyCommand, NoMatchResult, ErrorResult] 