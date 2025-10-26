"""Command types and data structures used throughout the application."""

from abc import ABC
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class ParseResult(BaseModel):
    """Base class for parsing results."""


class NoMatchResult(ParseResult):
    """Indicates no command was matched."""


class ErrorResult(ParseResult):
    """Indicates an error occurred during parsing.

    Attributes:
        error_message: Description of the error.
    """

    error_message: str


class BaseCommand(BaseModel, ABC):
    """Abstract base class for all commands.

    Configuration allows arbitrary types for flexibility with command structures.
    """

    class Config:
        arbitrary_types_allowed = True


class DictationCommand(BaseCommand):
    """Base class for dictation-related commands."""


class DictationStartCommand(DictationCommand):
    """Command to start standard dictation mode."""


class DictationStopCommand(DictationCommand):
    """Command to stop dictation mode."""


class DictationTypeCommand(DictationCommand):
    """Command to enter type mode."""


class DictationSmartStartCommand(DictationCommand):
    """Command to start smart dictation mode."""


ActionType = Literal["hotkey", "key", "key_sequence", "click", "scroll"]


class AutomationCommand(BaseCommand):
    """Base class for automation commands (PyAutoGUI).

    Attributes:
        command_key: The command phrase that triggers this action.
        action_type: Type of action (hotkey, key, key_sequence, click, scroll).
        action_value: The action value.
        is_custom: Whether this is a custom user-defined command.
        short_description: Short description for UI.
        long_description: Detailed description.
        functional_group: Functional grouping (Basic, Window Navigation, etc.).
    """

    command_key: str = Field(..., description="The command phrase that triggers this action")
    action_type: ActionType = Field(..., description="Type of action: 'hotkey', 'key', 'key_sequence', 'click', 'scroll'")
    action_value: str = Field(..., description="The action value")
    is_custom: bool = Field(default=False, description="Whether this is a custom user-defined command")
    short_description: str = Field(default="", description="Short description for UI")
    long_description: str = Field(default="", description="Detailed description")
    functional_group: str = Field(
        default="Other",
        description="Functional grouping: Basic, Window Navigation, Editing, Cursor IDE, VSCode IDE, Other, Custom",
    )

    @property
    def display_description(self) -> str:
        """Get the appropriate description for display.

        Returns:
            Short description if available, otherwise generated description.
        """
        return self.short_description if self.short_description else self._generate_short_description()

    def _generate_short_description(self) -> str:
        """Generate a short description based on action type.

        Returns:
            Description string for the action type.
        """
        action_type_map = {
            "hotkey": "Hotkey",
            "key": "Key",
            "key_sequence": "Key Sequence",
            "click": "Click",
            "scroll": "Scroll",
            "custom": "Custom",
            "type": "Type",
        }
        return action_type_map.get(self.action_type, "Action")


class ExactMatchCommand(AutomationCommand):
    """Command for exact phrase matches that execute once."""


class ParameterizedCommand(AutomationCommand):
    """Command for parameterized actions with repeat count.

    Attributes:
        count: Number of times to repeat the command.
    """

    count: int = Field(default=1, description="Number of times to repeat the command")


class MarkCommand(BaseCommand):
    """Base class for mark-related commands."""


class MarkCreateCommand(MarkCommand):
    """Command to create a new mark.

    Attributes:
        label: The label for the new mark.
        x: The x coordinate for the new mark.
        y: The y coordinate for the new mark.
    """

    label: str = Field(..., description="The label for the new mark")
    x: float = Field(..., description="The x coordinate for the new mark")
    y: float = Field(..., description="The y coordinate for the new mark")


class MarkExecuteCommand(MarkCommand):
    """Command to execute a click at a mark location.

    Attributes:
        label: The label of the mark to execute.
    """

    label: str = Field(..., description="The label of the mark to execute")


class MarkDeleteCommand(MarkCommand):
    """Command to delete a specific mark.

    Attributes:
        label: The label of the mark to delete.
    """

    label: str = Field(..., description="The label of the mark to delete")


class MarkVisualizeCommand(MarkCommand):
    """Command to show/visualize all marks."""


class MarkResetCommand(MarkCommand):
    """Command to reset/delete all marks."""


class MarkVisualizeCancelCommand(MarkCommand):
    """Command to cancel mark visualization."""


class GridCommand(BaseCommand):
    """Base class for grid-related commands."""


class GridShowCommand(GridCommand):
    """Command to show the grid overlay.

    Attributes:
        num_rects: Number of rectangles to show.
    """

    num_rects: Optional[int] = Field(default=None, description="Number of rectangles to show")


class GridSelectCommand(GridCommand):
    """Command to select a grid cell by number.

    Attributes:
        selected_number: The number selected from the grid.
    """

    selected_number: int = Field(..., description="The number selected from the grid")


class SoundCommand(BaseCommand):
    """Base class for sound management commands."""


class SoundTrainCommand(SoundCommand):
    """Command to train a new sound.

    Attributes:
        sound_label: The label for the sound to be trained.
    """

    sound_label: str = Field(..., description="The label for the sound to be trained")


class SoundDeleteCommand(SoundCommand):
    """Command to delete a trained sound.

    Attributes:
        sound_label: The label of the sound to delete.
    """

    sound_label: str = Field(..., description="The label of the sound to delete")


class SoundResetAllCommand(SoundCommand):
    """Command to reset all trained sounds."""


class SoundListAllCommand(SoundCommand):
    """Command to list all trained sounds."""


class SoundMapCommand(SoundCommand):
    """Command to map a sound to a command phrase.

    Attributes:
        sound_label: The label of the sound to map.
        command_phrase: The command phrase to map to the sound.
    """

    sound_label: str = Field(..., description="The label of the sound to map")
    command_phrase: str = Field(..., description="The command phrase to map to the sound")


DictationCommandType = Union[DictationStartCommand, DictationStopCommand, DictationTypeCommand, DictationSmartStartCommand]

AutomationCommandType = Union[ExactMatchCommand, ParameterizedCommand]

MarkCommandType = Union[
    MarkCreateCommand, MarkExecuteCommand, MarkDeleteCommand, MarkVisualizeCommand, MarkResetCommand, MarkVisualizeCancelCommand
]

GridCommandType = Union[GridShowCommand, GridSelectCommand]

SoundCommandType = Union[SoundTrainCommand, SoundDeleteCommand, SoundResetAllCommand, SoundListAllCommand, SoundMapCommand]

AnyCommand = Union[DictationCommandType, AutomationCommandType, MarkCommandType, GridCommandType, SoundCommandType]

ParseResultType = Union[AnyCommand, NoMatchResult, ErrorResult]
