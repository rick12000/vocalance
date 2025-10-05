"""
Automation Command Registry

Centralized registry for automation commands using proper command objects.
Only handles automation commands - other command types are managed by their respective services.
"""

from typing import Dict, List, Set
from iris.app.config.command_types import ExactMatchCommand, AutomationCommand


class AutomationCommandRegistry:
    """
    Registry for automation commands using proper command objects.
    Provides default automation commands and helper methods for lookup.
    """
    
    # Default automation commands as proper command objects
    DEFAULT_COMMANDS = [
        # Tab/Window Management
        ExactMatchCommand(
            command_key="new tango",
            action_type="hotkey",
            action_value="ctrl+t",
            short_description="New Tab",
            long_description="Open a new tab in the current application"
        ),
        ExactMatchCommand(
            command_key="close tango",
            action_type="hotkey",
            action_value="ctrl+w",
            short_description="Close Tab",
            long_description="Close the current tab"
        ),
        ExactMatchCommand(
            command_key="reopen tango",
            action_type="hotkey",
            action_value="ctrl+shift+t",
            short_description="Reopen Tab",
            long_description="Reopen the last closed tab"
        ),
        ExactMatchCommand(
            command_key="tango right",
            action_type="hotkey",
            action_value="ctrl+tab",
            short_description="Next Tab",
            long_description="Switch to the next tab"
        ),
        ExactMatchCommand(
            command_key="tango left",
            action_type="hotkey",
            action_value="ctrl+shift+tab",
            short_description="Previous Tab",
            long_description="Switch to the previous tab"
        ),
        
        # File Operations
        ExactMatchCommand(
            command_key="save",
            action_type="hotkey",
            action_value="ctrl+s",
            short_description="Save",
            long_description="Save the current file"
        ),
        ExactMatchCommand(
            command_key="save all code",
            action_type="hotkey",
            action_value="ctrl+alt+s",
            short_description="Save All",
            long_description="Save all open files"
        ),
        
        # Edit Operations
        ExactMatchCommand(
            command_key="copy",
            action_type="hotkey",
            action_value="ctrl+c",
            short_description="Copy",
            long_description="Copy selected text or content to clipboard"
        ),
        ExactMatchCommand(
            command_key="paste",
            action_type="hotkey",
            action_value="ctrl+v",
            short_description="Paste",
            long_description="Paste content from clipboard"
        ),
        ExactMatchCommand(
            command_key="select all",
            action_type="hotkey",
            action_value="ctrl+a",
            short_description="Select All",
            long_description="Select all content in the current context"
        ),
        ExactMatchCommand(
            command_key="undo",
            action_type="hotkey",
            action_value="ctrl+z",
            short_description="Undo",
            long_description="Undo the last action"
        ),
        ExactMatchCommand(
            command_key="redo",
            action_type="hotkey",
            action_value="ctrl+y",
            short_description="Redo",
            long_description="Redo the last undone action"
        ),
        
        # View Operations
        ExactMatchCommand(
            command_key="zoom",
            action_type="hotkey",
            action_value="ctrl++",
            short_description="Zoom In",
            long_description="Increase zoom level"
        ),
        ExactMatchCommand(
            command_key="zoom out",
            action_type="hotkey",
            action_value="ctrl+-",
            short_description="Zoom Out",
            long_description="Decrease zoom level"
        ),
        
        # Click Operations
        ExactMatchCommand(
            command_key="click",
            action_type="click",
            action_value="click",
            short_description="Click",
            long_description="Perform a left mouse click at current cursor position"
        ),
        ExactMatchCommand(
            command_key="left click",
            action_type="click",
            action_value="left_click",
            short_description="Left Click",
            long_description="Perform a left mouse click"
        ),
        ExactMatchCommand(
            command_key="right click",
            action_type="click",
            action_value="right_click",
            short_description="Right Click",
            long_description="Perform a right mouse click to open context menu"
        ),
        ExactMatchCommand(
            command_key="double click",
            action_type="click",
            action_value="double_click",
            short_description="Double Click",
            long_description="Perform a double left mouse click"
        ),
        ExactMatchCommand(
            command_key="triple click",
            action_type="click",
            action_value="triple_click",
            short_description="Triple Click",
            long_description="Perform a triple left mouse click to select line"
        ),
        
        # Navigation Keys
        ExactMatchCommand(
            command_key="up",
            action_type="key",
            action_value="up",
            short_description="Up Arrow",
            long_description="Press the up arrow key"
        ),
        ExactMatchCommand(
            command_key="down",
            action_type="key",
            action_value="down",
            short_description="Down Arrow",
            long_description="Press the down arrow key"
        ),
        ExactMatchCommand(
            command_key="left",
            action_type="key",
            action_value="left",
            short_description="Left Arrow",
            long_description="Press the left arrow key"
        ),
        ExactMatchCommand(
            command_key="right",
            action_type="key",
            action_value="right",
            short_description="Right Arrow",
            long_description="Press the right arrow key"
        ),
        ExactMatchCommand(
            command_key="wind",
            action_type="key",
            action_value="pageup",
            short_description="Page Up",
            long_description="Press the page up key to scroll up"
        ),
        ExactMatchCommand(
            command_key="ground",
            action_type="key",
            action_value="pagedown",
            short_description="Page Down",
            long_description="Press the page down key to scroll down"
        ),
        
        # Special Keys
        ExactMatchCommand(
            command_key="enter",
            action_type="key",
            action_value="enter",
            short_description="Enter",
            long_description="Press the enter key to confirm or create new line"
        ),
        ExactMatchCommand(
            command_key="delete",
            action_type="key",
            action_value="delete",
            short_description="Delete",
            long_description="Press the delete key to remove character after cursor"
        ),
        ExactMatchCommand(
            command_key="escape",
            action_type="key",
            action_value="escape",
            short_description="Escape",
            long_description="Press the escape key to cancel or exit"
        ),
        ExactMatchCommand(
            command_key="tab",
            action_type="key",
            action_value="tab",
            short_description="Tab",
            long_description="Press the tab key for indentation or navigation"
        ),
        ExactMatchCommand(
            command_key="space",
            action_type="key",
            action_value="space",
            short_description="Space",
            long_description="Press the space bar"
        ),
        
        # Application Commands
        ExactMatchCommand(
            command_key="next",
            action_type="hotkey",
            action_value="alt+f5",
            short_description="Next",
            long_description="Navigate to next item or execute next command"
        ),
        ExactMatchCommand(
            command_key="previous",
            action_type="hotkey",
            action_value="shift+alt+f5",
            short_description="Previous",
            long_description="Navigate to previous item or execute previous command"
        ),
        ExactMatchCommand(
            command_key="context",
            action_type="hotkey",
            action_value="ctrl+/",
            short_description="Context Menu",
            long_description="Open context-sensitive help or menu"
        ),
        ExactMatchCommand(
            command_key="talk",
            action_type="hotkey",
            action_value="alt+n",
            short_description="Talk",
            long_description="Activate talk or communication feature"
        ),
        ExactMatchCommand(
            command_key="models",
            action_type="hotkey",
            action_value="ctrl+alt+.",
            short_description="Models",
            long_description="Access AI models or model selection"
        ),
        ExactMatchCommand(
            command_key="references",
            action_type="hotkey",
            action_value="shift+alt+f12",
            short_description="References",
            long_description="Show references or related information"
        ),
        
        # Scroll Operations
        ExactMatchCommand(
            command_key="sky",
            action_type="scroll",
            action_value="scroll_up",
            short_description="Scroll Up",
            long_description="Scroll the page or content upward"
        ),
        ExactMatchCommand(
            command_key="earth",
            action_type="scroll",
            action_value="scroll_down",
            short_description="Scroll Down",
            long_description="Scroll the page or content downward"
        ),

        # Code:
        ExactMatchCommand(
            command_key="rename",
            action_type="hotkey",
            action_value="f2",
            short_description="Rename",
            long_description="Rename the current cursor position variable in VSCode"
        ),

        ExactMatchCommand(
            command_key="definition",
            action_type="hotkey",
            action_value="f12",
            short_description="Definition",
            long_description="Go to the definition of the current cursor position variable in VSCode"
        ),

        ExactMatchCommand(
            command_key="code search",
            action_type="hotkey",
            action_value="ctrl+shift+f",
            short_description="Code Search",
            long_description="Open cross file search in VSCode"
        ),

        ExactMatchCommand(
            command_key="back",
            action_type="hotkey",
            action_value="alt+left",
            short_description="Code Back",
            long_description="Go back to the previous cursor position in VSCode"
        ),

        ExactMatchCommand(
            command_key="forward",
            action_type="hotkey",
            action_value="alt+right",
            short_description="Code Forward",
            long_description="Go forward to the next cursor position in VSCode"
        ),

        ExactMatchCommand(
            command_key="ask",
            action_type="hotkey",
            action_value="ctrl+k",
            short_description="Ask AI",
            long_description="Toggle in line AI in Cursor"
        ),   

        ExactMatchCommand(
            command_key="bat",
            action_type="hotkey",
            action_value="ctrl+pageup",
            short_description="Next open tab",
            long_description="Go to the next open tab in Cursor"
        ),     

        ExactMatchCommand(
            command_key="fox",
            action_type="hotkey",
            action_value="ctrl+pagedown",
            short_description="Previous open tab",
            long_description="Go to the previous open tab in Cursor"
        ),     

        ExactMatchCommand(
            command_key="explore",
            action_type="hotkey",
            action_value="ctrl+shift+e",
            short_description="Files",
            long_description="Open the files panel in Cursor"
        ),   
    ]
    
    @classmethod
    def get_default_commands(cls) -> List[AutomationCommand]:
        """Get all default automation commands"""
        return cls.DEFAULT_COMMANDS.copy()
    
    @classmethod
    def get_commands_dict(cls) -> Dict[str, AutomationCommand]:
        """Get automation commands as a dictionary keyed by command phrase"""
        return {cmd.command_key: cmd for cmd in cls.DEFAULT_COMMANDS}
    
    @classmethod
    def get_command_phrases(cls) -> List[str]:
        """Get list of all automation command phrases"""
        return [cmd.command_key for cmd in cls.DEFAULT_COMMANDS]
    
    @classmethod
    def get_protected_phrases(cls) -> Set[str]:
        """Get phrases that are protected from being used as custom commands"""
        return {cmd.command_key for cmd in cls.DEFAULT_COMMANDS} 