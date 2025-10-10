"""
Smart Timeout Manager

Determines command ambiguity based on full-word matching.
Only used for Markov prediction, not for partial recognition.
"""
import logging
from typing import Dict, Optional, Set

from iris.app.config.app_config import GlobalAppConfig

logger = logging.getLogger(__name__)

class SmartTimeoutManager:
    """Manages command ambiguity detection for full recognized commands"""
    
    def __init__(self, app_config: GlobalAppConfig, command_action_map: Optional[Dict[str, tuple]] = None):
        """Initialize with command ambiguity analysis"""
        self.app_config = app_config
        self._command_action_map = command_action_map or {}
        
        # All recognized complete commands
        self._all_commands: Set[str] = set()
        
        # Commands that are prefixes of other commands (e.g., "right" is prefix of "right click")
        self._ambiguous_commands: Set[str] = set()
        
        # Analyze commands
        self._analyze_commands()
        
        logger.info(f"SmartTimeoutManager initialized: {len(self._all_commands)} commands, "
                   f"{len(self._ambiguous_commands)} ambiguous commands")

    def update_command_action_map(self, command_action_map: Dict[str, tuple]) -> None:
        """Update the command action map and re-analyze commands"""
        self._command_action_map = command_action_map
        self._analyze_commands()
        logger.debug(f"Updated command action map with {len(command_action_map)} commands")

    def _analyze_commands(self) -> None:
        """Analyze commands for ambiguity detection based on full words"""
        self._all_commands.clear()
        self._ambiguous_commands.clear()
        
        # Extract all command phrases
        commands = set()
        for cmd_key in self._command_action_map.keys():
            clean_cmd = cmd_key.lower().strip()
            if clean_cmd:
                commands.add(clean_cmd)
                self._all_commands.add(clean_cmd)
        
        # Build ambiguity analysis for full words only
        self._build_ambiguity_analysis(commands)

    def _build_ambiguity_analysis(self, commands: Set[str]) -> None:
        """
        Build ambiguous command set based on full-word prefixes
        
        Example:
        - "right" is ambiguous if "right click" exists
        - "click" is NOT ambiguous if no other command starts with "click "
        """
        commands_list = list(commands)
        
        for command in commands_list:
            # Check if this command is a prefix of any other command
            # e.g., "right" is prefix of "right click"
            is_prefix_of_others = any(
                other_cmd.startswith(command + " ") 
                for other_cmd in commands_list 
                if other_cmd != command
            )
            
            if is_prefix_of_others:
                self._ambiguous_commands.add(command)
        
        logger.info(f"Built ambiguity analysis: {len(self._ambiguous_commands)} ambiguous commands")

    def is_ambiguous(self, text: str) -> bool:
        """
        Check if a recognized command text is ambiguous
        
        Args:
            text: Fully recognized command text
            
        Returns:
            True if command is ambiguous (is a prefix of other commands)
            False if unambiguous
            
        Examples:
            - is_ambiguous("right") -> True (if "right click" exists)
            - is_ambiguous("right click") -> False (complete command)
            - is_ambiguous("click") -> False (no "click ..." commands)
        """
        if not text:
            return True
        
        text_lower = text.lower().strip()
        
        # If text is not even a known command, it's ambiguous (unknown)
        if text_lower not in self._all_commands:
            return True
        
        # If text is a known ambiguous command (prefix of others), return True
        return text_lower in self._ambiguous_commands
