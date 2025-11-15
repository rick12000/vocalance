from typing import Dict, List, Set

from vocalance.app.config.command_types import AutomationCommand, ExactMatchCommand


class AutomationCommandRegistry:
    """Registry for default automation commands with lookup and query utilities.

    Provides a comprehensive set of default voice-activated automation commands organized
    by functional groups (editing, navigation, IDE operations, etc.). Supports exact match
    lookups, command key retrieval, and validation of protected terms through helper methods.
    """

    DEFAULT_COMMANDS: List[AutomationCommand] = [
        ExactMatchCommand(
            command_key="web open",
            action_type="hotkey",
            action_value="ctrl+t",
            short_description="New Tab",
            long_description="Open a new browser tab (also works in VSCode or other tabbed applications)",
            functional_group="Window Navigation",
        ),
        ExactMatchCommand(
            command_key="web close",
            action_type="hotkey",
            action_value="ctrl+w",
            short_description="Close Tab",
            long_description="Close the current browser tab (also works in VSCode or other tabbed applications)",
            functional_group="Window Navigation",
        ),
        ExactMatchCommand(
            command_key="close",
            action_type="hotkey",
            action_value="alt+f4",
            short_description="Close Window",
            long_description="Close the currently active window",
            functional_group="Window Navigation",
        ),
        ExactMatchCommand(
            command_key="minimize",
            action_type="key_sequence",
            action_value="alt+space, n",
            short_description="Minimize Window",
            long_description="Minimize the currently active window to the taskbar",
            functional_group="Window Navigation",
        ),
        ExactMatchCommand(
            command_key="web reopen",
            action_type="hotkey",
            action_value="ctrl+shift+t",
            short_description="Reopen Tab",
            long_description="Reopen the last closed browser tab",
            functional_group="Window Navigation",
        ),
        ExactMatchCommand(
            command_key="web right",
            action_type="hotkey",
            action_value="ctrl+tab",
            short_description="Next Tab",
            long_description="Open the browser tab to the right of the current browser tab (also works in VSCode or other tabbed applications)",
            functional_group="Window Navigation",
        ),
        ExactMatchCommand(
            command_key="web left",
            action_type="hotkey",
            action_value="ctrl+shift+tab",
            short_description="Previous Tab",
            long_description="Open the browser tab to the left of the current browser tab (also works in VSCode or other tabbed applications)",
            functional_group="Window Navigation",
        ),
        ExactMatchCommand(
            command_key="save",
            action_type="hotkey",
            action_value="ctrl+s",
            short_description="Save",
            long_description="Save the current file",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="save all",
            action_type="hotkey",
            action_value="ctrl+alt+s",
            short_description="Save All",
            long_description="Save all open files",
            functional_group="General IDE",
        ),
        ExactMatchCommand(
            command_key="close all",
            action_type="hotkey",
            action_value="ctrl+m+w",
            short_description="Close All",
            long_description="Close all open files",
            functional_group="General IDE",
        ),
        ExactMatchCommand(
            command_key="copy",
            action_type="hotkey",
            action_value="ctrl+c",
            short_description="Copy",
            long_description="Copy selected text or content to clipboard",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="paste",
            action_type="hotkey",
            action_value="ctrl+v",
            short_description="Paste",
            long_description="Paste content from clipboard",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="select all",
            action_type="hotkey",
            action_value="ctrl+a",
            short_description="Select All",
            long_description="Select all content in the current context",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="wipe",
            action_type="hotkey",
            action_value="ctrl+z",
            short_description="Undo",
            long_description="Undo the last action",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="redo",
            action_type="hotkey",
            action_value="ctrl+y",
            short_description="Redo",
            long_description="Redo the last undone action",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="zoom",
            action_type="hotkey",
            action_value="ctrl++",
            short_description="Zoom In",
            long_description="Increase zoom level",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="zoom out",
            action_type="hotkey",
            action_value="ctrl+-",
            short_description="Zoom Out",
            long_description="Decrease zoom level",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="click",
            action_type="click",
            action_value="click",
            short_description="Click",
            long_description="Perform a left mouse click at current mouse cursor position",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="right click",
            action_type="click",
            action_value="right_click",
            short_description="Right Click",
            long_description="Perform a right mouse click at current mouse cursor position",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="double click",
            action_type="click",
            action_value="double_click",
            short_description="Double Click",
            long_description="Perform a double left mouse click at current mouse cursor position",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="triple click",
            action_type="click",
            action_value="triple_click",
            short_description="Triple Click",
            long_description="Perform a triple left mouse click at current mouse cursor position",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="up",
            action_type="key",
            action_value="up",
            short_description="Up Arrow",
            long_description="Press the up arrow key",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="down",
            action_type="key",
            action_value="down",
            short_description="Down Arrow",
            long_description="Press the down arrow key",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="left",
            action_type="key",
            action_value="left",
            short_description="Left Arrow",
            long_description="Press the left arrow key",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="right",
            action_type="key",
            action_value="right",
            short_description="Right Arrow",
            long_description="Press the right arrow key",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="wind",
            action_type="key",
            action_value="pageup",
            short_description="Page Up",
            long_description="Press the page up key (use to scroll up)",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="ground",
            action_type="key",
            action_value="pagedown",
            short_description="Page Down",
            long_description="Press the page down key (use to scroll down)",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="sky",
            action_type="scroll",
            action_value="up",
            short_description="Scroll Up",
            long_description="Scroll upward using mouse scroll wheel",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="earth",
            action_type="scroll",
            action_value="down",
            short_description="Scroll Down",
            long_description="Scroll downward using mouse scroll wheel",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="enter",
            action_type="key",
            action_value="enter",
            short_description="Enter",
            long_description="Press the enter key",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="delete",
            action_type="key",
            action_value="delete",
            short_description="Delete",
            long_description="Press the delete key",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="escape",
            action_type="key",
            action_value="escape",
            short_description="Escape",
            long_description="Press the escape key",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="tab",
            action_type="key",
            action_value="tab",
            short_description="Tab",
            long_description="Press the tab key",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="space",
            action_type="key",
            action_value="space",
            short_description="Space",
            long_description="Press the space bar",
            functional_group="Editing",
        ),
        ExactMatchCommand(
            command_key="square next",
            action_type="hotkey",
            action_value="alt+f5",
            short_description="Next",
            long_description="Move to next AI changed section in Cursor",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="square previous",
            action_type="hotkey",
            action_value="shift+alt+f5",
            short_description="Previous",
            long_description="Move to previous AI changed section in Cursor",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="blue context",
            action_type="hotkey",
            action_value="ctrl+/",
            short_description="Context Menu",
            long_description="Open context drop down in Copilot Chat (only trigger if you're in the context of a chat)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue talk",
            action_type="hotkey",
            action_value="alt+n",
            short_description="Talk",
            long_description="Activate microphone in Copilot Chat (only trigger if you're in the context of a chat)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue models",
            action_type="hotkey",
            action_value="ctrl+alt+.",
            short_description="Models",
            long_description="Open AI model selection menu in Copilot Chat (only trigger if you're in the context of a chat)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue ask",
            action_type="hotkey",
            action_value="ctrl+i",
            short_description="Ask AI",
            long_description="Toggle in line AI Chat in VSCode",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue new",
            action_type="hotkey",
            action_value="ctrl+n",
            short_description="AI New Chat",
            long_description="Open a new AI chat in VSCode (only trigger if you're in the context of a chat)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue keep line",
            action_type="hotkey",
            action_value="ctrl+y",
            short_description="Keep Line",
            long_description="Keep edit at given or next highlighted code section in VSCode (only trigger if you are on a file with AI changes)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue drop line",
            action_type="hotkey",
            action_value="ctrl+n",
            short_description="Reject Line",
            long_description="Reject edit at given or next highlighted code section in VSCode (only trigger if you are on a file with AI changes)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue keep file",
            action_type="hotkey",
            action_value="ctrl+shift+y",
            short_description="Keep File",
            long_description="Keep edits in the current file in VSCode (only trigger if you are on a file with AI changes)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue reject file",
            action_type="hotkey",
            action_value="ctrl+shift+n",
            short_description="Reject File",
            long_description="Reject edits in the current file in VSCode (only trigger if you are on a file with AI changes)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue next",
            action_type="hotkey",
            action_value="alt+f5",
            short_description="Next",
            long_description="Move to next AI changed section in VSCode (only trigger if you are on a file with AI changes)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="blue previous",
            action_type="hotkey",
            action_value="shift+alt+f5",
            short_description="Previous",
            long_description="Move to previous AI changed section in VSCode (only trigger if you are on a file with AI changes)",
            functional_group="VSCode IDE",
        ),
        ExactMatchCommand(
            command_key="square mode",
            action_type="hotkey",
            action_value="ctrl+alt+.",
            short_description="AI Modes",
            long_description="Cycle through Cursor Chat AI modes [Agent, Plan, Ask] (only trigger if you're in the context of a chat)",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="square models",
            action_type="hotkey",
            action_value="ctrl+/",
            short_description="AI Models",
            long_description="Open AI model selection menu in Cursor Chat (only trigger if you're in the context of a chat)",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="square keep line",
            action_type="hotkey",
            action_value="ctrl+shift+y",
            short_description="Keep Line",
            long_description="Keep edit at given or next highlighted code section in Cursor (only trigger if you are on a file with AI changes)",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="square drop line",
            action_type="hotkey",
            action_value="ctrl+n",
            short_description="Reject Line",
            long_description="Reject edit at given or next highlighted code section in Cursor (only trigger if you are on a file with AI changes)",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="square keep file",
            action_type="hotkey",
            action_value="ctrl+enter",
            short_description="Keep File",
            long_description="Keep edits in the current file in Cursor (only trigger if you are on a file with AI changes)",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="square reject file",
            action_type="hotkey",
            action_value="ctrl+shift+backspace",
            short_description="Reject File",
            long_description="Reject edits in the current file in Cursor (only trigger if you are on a file with AI changes)",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="square context",
            action_type="hotkey",
            action_value="ctrl+alt+p",
            short_description="AI Context",
            long_description="Open context drop down in Cursor Chat (only trigger if you're in the context of a chat)",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="square new",
            action_type="hotkey",
            action_value="ctrl+t",
            short_description="AI New Chat",
            long_description="Open a new AI chat in Cursor",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="search",
            action_type="hotkey",
            action_value="ctrl+f",
            short_description="Text Search",
            long_description="Search for text in the current file",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="back space",
            action_type="hotkey",
            action_value="backspace",
            short_description="Backspace",
            long_description="Press the backspace key",
            functional_group="Basic",
        ),
        ExactMatchCommand(
            command_key="references",
            action_type="hotkey",
            action_value="shift+alt+f12",
            short_description="References",
            long_description="Shows code references for the selected variable (trigger if your cursor is on a variable)",
            functional_group="General IDE",
        ),
        ExactMatchCommand(
            command_key="rename",
            action_type="hotkey",
            action_value="f2",
            short_description="Rename",
            long_description="Rename the selected variable (trigger if your cursor is on a variable)",
            functional_group="General IDE",
        ),
        ExactMatchCommand(
            command_key="definition",
            action_type="hotkey",
            action_value="f12",
            short_description="Definition",
            long_description="Shows where the selected variable is defined (trigger if your cursor is on a variable)",
            functional_group="General IDE",
        ),
        ExactMatchCommand(
            command_key="code search",
            action_type="hotkey",
            action_value="ctrl+shift+f",
            short_description="Code Search",
            long_description="Opens file search in IDE (if you trigger this when your cursor is highlighting some text, it will automatically search for that text across files)",
            functional_group="General IDE",
        ),
        ExactMatchCommand(
            command_key="back",
            action_type="hotkey",
            action_value="alt+left",
            short_description="Code Back",
            long_description="If triggered on a browser, goes back to the previous page. If triggered in a coding IDE, goes back to the previous cursor position.",
            functional_group="Window Navigation",
        ),
        ExactMatchCommand(
            command_key="jet",
            action_type="hotkey",
            action_value="alt+right",
            short_description="Code Forward",
            long_description="If triggered on a browser, goes forward to the next page. If triggered in a coding IDE, goes forward to the next cursor position.",
            functional_group="Window Navigation",
        ),
        ExactMatchCommand(
            command_key="square ask",
            action_type="hotkey",
            action_value="ctrl+k",
            short_description="Ask AI",
            long_description="Toggle in line AI Chat in Cursor",
            functional_group="Cursor IDE",
        ),
        ExactMatchCommand(
            command_key="old",
            action_type="hotkey",
            action_value="ctrl+pageup",
            short_description="Next open tab",
            long_description="If triggered on a browser, goes to tab to the left of the current tab. If triggered in a coding IDE, goes to the file tab to the left of the current file.",
            functional_group="General IDE",
        ),
        ExactMatchCommand(
            command_key="new",
            action_type="hotkey",
            action_value="ctrl+pagedown",
            short_description="Previous open tab",
            long_description="If triggered on a browser, goes to tab to the right of the current tab. If triggered in a coding IDE, goes to the file tab to the right of the current file.",
            functional_group="General IDE",
        ),
        ExactMatchCommand(
            command_key="explore",
            action_type="hotkey",
            action_value="ctrl+shift+e",
            short_description="Files",
            long_description="Open the files panel in coding IDEs",
            functional_group="General IDE",
        ),
    ]

    @classmethod
    def get_default_commands(cls) -> List[AutomationCommand]:
        """Get all default automation commands as a copy.

        Returns:
            Shallow copy of the complete default automation commands list.
        """
        return cls.DEFAULT_COMMANDS.copy()

    @classmethod
    def get_commands_dict(cls) -> Dict[str, AutomationCommand]:
        """Get automation commands as a dictionary keyed by command phrase.

        Constructs a dictionary mapping each command's voice trigger phrase to its
        AutomationCommand object for efficient lookup operations.

        Returns:
            Dictionary mapping command key strings to AutomationCommand instances.
        """
        return {cmd.command_key: cmd for cmd in cls.DEFAULT_COMMANDS}

    @classmethod
    def get_command_phrases(cls) -> List[str]:
        """Get list of all automation command voice trigger phrases.

        Returns:
            List of command key strings used as voice triggers.
        """
        return [cmd.command_key for cmd in cls.DEFAULT_COMMANDS]

    @classmethod
    def get_protected_phrases(cls) -> Set[str]:
        """Get phrases that are protected from being overridden by custom commands.

        Returns:
            Set of command key strings reserved for default automation commands.
        """
        return {cmd.command_key for cmd in cls.DEFAULT_COMMANDS}
