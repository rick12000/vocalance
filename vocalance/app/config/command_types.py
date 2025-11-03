from abc import ABC
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class ParseResult(BaseModel):
    """Base class for command parsing results.

    Marker class used as a discriminated union base for successful command parses,
    no-match results, and error results.
    """


class NoMatchResult(ParseResult):
    """Indicates no command was matched during parsing.

    Used by the command parser to signal that the input text did not match any
    known command pattern or trigger phrase.
    """


class ErrorResult(ParseResult):
    """Indicates an error occurred during command parsing.

    Captures parsing errors distinct from no-matches, such as malformed input,
    invalid parameters, or internal parsing failures.

    Attributes:
        error_message: Human-readable description of the error.
    """

    error_message: str


class BaseCommand(BaseModel, ABC):
    """Abstract base class for all command types in the system.

    Serves as the root of the command hierarchy, inherited by dictation, automation,
    mark, grid, and sound command families. Pydantic configuration allows arbitrary
    types for flexibility in command structure extensions.
    """

    class Config:
        arbitrary_types_allowed = True


class DictationCommand(BaseCommand):
    """Base class for dictation-related commands.

    Parent class for commands controlling dictation mode lifecycle and behavior.
    """


class DictationStartCommand(DictationCommand):
    """Command to start standard dictation mode.

    Activates dictation with the configured start trigger phrase, capturing speech
    for text generation without formatting.
    """


class DictationStopCommand(DictationCommand):
    """Command to stop dictation mode and return to command mode.

    Triggered by the configured stop phrase to end dictation capture.
    """


class DictationTypeCommand(DictationCommand):
    """Command to enter type mode for immediate text input.

    Captures a short phrase and types it directly without additional processing.
    """


class DictationSmartStartCommand(DictationCommand):
    """Command to start smart dictation mode with LLM formatting.

    Activates dictation with automatic punctuation and capitalization applied
    through the LLM service before output.
    """


class DictationVisualStartCommand(DictationCommand):
    """Command to start visual dictation mode with UI display but no LLM.

    Activates dictation with accumulated text displayed in a popup UI,
    similar to smart mode but without LLM processing. Text is pasted
    directly when stopped.
    """


ActionType = Literal["hotkey", "key", "key_sequence", "click", "scroll"]


class AutomationCommand(BaseCommand):
    """Base class for automation commands executed via PyAutoGUI.

    Represents voice-triggered keyboard/mouse automation actions including hotkeys,
    individual keys, key sequences, clicks, and scrolling. Supports both default
    and custom user-defined commands with categorization by functional group.

    Attributes:
        command_key: Voice trigger phrase that activates this automation action.
        action_type: Type of action - 'hotkey', 'key', 'key_sequence', 'click', or 'scroll'.
        action_value: Action-specific value (e.g., 'ctrl+s' for hotkey, 'enter' for key).
        is_custom: True if user-defined custom command, False for default commands.
        short_description: Brief description displayed in UI command listings.
        long_description: Detailed explanation of command purpose and behavior.
        functional_group: Category for organization (Basic, Window Navigation, Editing, etc.).
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
        """Get the appropriate description for UI display.

        Prefers the explicit short_description if set, otherwise generates a
        fallback description based on the action_type.

        Returns:
            Short description string suitable for UI display.
        """
        return self.short_description if self.short_description else self._generate_short_description()

    def _generate_short_description(self) -> str:
        """Generate a fallback short description from action type.

        Maps action_type to a human-readable category label when no explicit
        short_description is provided.

        Returns:
            Generated description string based on action_type, or "Action" as fallback.
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
    """Command for exact phrase matches that execute a single action.

    Represents commands that match the input phrase exactly and execute their
    action once without repetition or parameterization.
    """


class ParameterizedCommand(AutomationCommand):
    """Command for parameterized actions supporting repeat counts.

    Extends AutomationCommand with a count parameter enabling repeated execution
    of the same action, commonly used with scrolling or repeated key presses.

    Attributes:
        count: Number of times to repeat the command execution (default 1).
    """

    count: int = Field(default=1, description="Number of times to repeat the command")


class MarkCommand(BaseCommand):
    """Base class for mark-related commands.

    Parent class for commands managing screen position bookmarks (marks) that
    enable clicking saved locations via voice commands.
    """


class MarkCreateCommand(MarkCommand):
    """Command to create a new mark at current cursor position.

    Captures the current mouse coordinates and associates them with a voice-activated
    label for future click execution.

    Attributes:
        label: Voice-activated label for the new mark.
        x: Screen x-coordinate of the mark position.
        y: Screen y-coordinate of the mark position.
    """

    label: str = Field(..., description="The label for the new mark")
    x: float = Field(..., description="The x coordinate for the new mark")
    y: float = Field(..., description="The y coordinate for the new mark")


class MarkExecuteCommand(MarkCommand):
    """Command to execute a click at a previously saved mark location.

    Performs a mouse click at the screen coordinates associated with the
    specified mark label.

    Attributes:
        label: Voice label of the mark to click.
    """

    label: str = Field(..., description="The label of the mark to execute")


class MarkDeleteCommand(MarkCommand):
    """Command to delete a specific mark by label.

    Removes the mark from storage, preventing future execution of that label.

    Attributes:
        label: Voice label of the mark to delete.
    """

    label: str = Field(..., description="The label of the mark to delete")


class MarkVisualizeCommand(MarkCommand):
    """Command to show overlay visualization of all saved marks.

    Displays all mark positions and labels on screen for a configured duration.
    """


class MarkResetCommand(MarkCommand):
    """Command to reset and delete all marks from storage.

    Clears all saved marks, requiring re-creation from scratch.
    """


class MarkVisualizeCancelCommand(MarkCommand):
    """Command to immediately cancel mark visualization overlay.

    Hides the mark overlay before the auto-hide timeout expires.
    """


class GridCommand(BaseCommand):
    """Base class for grid-related commands.

    Parent class for commands controlling the click grid overlay system used for
    precise mouse positioning via numbered grid cells.
    """


class GridShowCommand(GridCommand):
    """Command to display the numbered grid overlay on screen.

    Shows a configurable grid of numbered cells covering the screen, with optional
    specification of cell count.

    Attributes:
        num_rects: Optional number of grid cells to display (uses default if None).
    """

    num_rects: Optional[int] = Field(default=None, description="Number of rectangles to show")


class GridSelectCommand(GridCommand):
    """Command to select and click a specific grid cell by its number.

    Identifies the grid cell by spoken number and performs a click at its center,
    then hides the grid overlay.

    Attributes:
        selected_number: Numeric identifier of the grid cell to click.
    """

    selected_number: int = Field(..., description="The number selected from the grid")


class SoundCommand(BaseCommand):
    """Base class for sound recognition management commands.

    Parent class for commands controlling training, deletion, listing, and mapping
    of custom sound recognition triggers.
    """


class SoundTrainCommand(SoundCommand):
    """Command to train a new custom sound recognition trigger.

    Initiates audio sample collection for training the sound recognizer to detect
    a specific sound associated with the given label.

    Attributes:
        sound_label: Voice label for the sound to be trained.
    """

    sound_label: str = Field(..., description="The label for the sound to be trained")


class SoundDeleteCommand(SoundCommand):
    """Command to delete a trained sound and its samples.

    Removes the trained sound model and associated samples from storage.

    Attributes:
        sound_label: Voice label of the sound to delete.
    """

    sound_label: str = Field(..., description="The label of the sound to delete")


class SoundResetAllCommand(SoundCommand):
    """Command to reset and delete all trained sounds.

    Clears all trained sound models and samples, requiring re-training.
    """


class SoundListAllCommand(SoundCommand):
    """Command to list all currently trained sounds.

    Displays or returns information about all trained sound labels in the system.
    """


class SoundMapCommand(SoundCommand):
    """Command to map a trained sound to a command phrase.

    Associates a trained sound trigger with a specific command phrase, so that
    detecting the sound executes the mapped command automatically.

    Attributes:
        sound_label: Voice label of the trained sound to map.
        command_phrase: Command phrase to execute when sound is detected.
    """

    sound_label: str = Field(..., description="The label of the sound to map")
    command_phrase: str = Field(..., description="The command phrase to map to the sound")


DictationCommandType = Union[
    DictationStartCommand, DictationStopCommand, DictationTypeCommand, DictationSmartStartCommand, DictationVisualStartCommand
]

AutomationCommandType = Union[ExactMatchCommand, ParameterizedCommand]

MarkCommandType = Union[
    MarkCreateCommand, MarkExecuteCommand, MarkDeleteCommand, MarkVisualizeCommand, MarkResetCommand, MarkVisualizeCancelCommand
]

GridCommandType = Union[GridShowCommand, GridSelectCommand]

SoundCommandType = Union[SoundTrainCommand, SoundDeleteCommand, SoundResetAllCommand, SoundListAllCommand, SoundMapCommand]

AnyCommand = Union[DictationCommandType, AutomationCommandType, MarkCommandType, GridCommandType, SoundCommandType]

ParseResultType = Union[AnyCommand, NoMatchResult, ErrorResult]
