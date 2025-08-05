"""
Smart Timeout Manager

Dynamically adjusts STT timeouts based on command ambiguity analysis.
Provides optimized timeouts with systematic command grouping.
"""
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from iris.config.app_config import GlobalAppConfig

logger = logging.getLogger(__name__)

@dataclass
class AmbiguityGroup:
    """Represents a group of commands that share prefixes"""
    prefix: str
    commands: List[str]
    max_word_count: int
    timeout_ms: int = 500

class SmartTimeoutManager:
    """Manages dynamic timeouts based on systematic command ambiguity analysis"""
    
    # Optimized timeout constants
    ZERO_TIMEOUT = 0.02      # Instant execution
    INSTANT_TIMEOUT = 0.05   # Near-instant for recognized commands
    QUICK_TIMEOUT = 0.15     # Fast for single words
    DEFAULT_TIMEOUT = 0.4    # Standard timeout
    AMBIGUOUS_TIMEOUT = 0.6  # Longer for ambiguous commands
    
    def __init__(self, app_config: GlobalAppConfig, command_action_map: Optional[Dict[str, tuple]] = None):
        """Initialize with systematic command analysis"""
        self.app_config = app_config
        self._command_action_map = command_action_map or {}
        
        # Command categorization for timeout decisions
        self._instant_commands: Set[str] = set()
        self._quick_commands: Set[str] = set()
        self._ambiguity_groups: Dict[str, AmbiguityGroup] = {}
        
        # Analyze commands for timeout assignment
        self._analyze_commands()
        
        logger.info(f"SmartTimeoutManager initialized: {len(self._instant_commands)} instant, "
                   f"{len(self._quick_commands)} quick, {len(self._ambiguity_groups)} ambiguity groups")

    def update_command_action_map(self, command_action_map: Dict[str, tuple]) -> None:
        """Update the command action map and re-analyze commands"""
        self._command_action_map = command_action_map
        self._analyze_commands()
        logger.debug(f"Updated command action map with {len(command_action_map)} commands")

    def _analyze_commands(self) -> None:
        """Systematically analyze commands for timeout optimization"""
        self._instant_commands.clear()
        self._quick_commands.clear()
        self._ambiguity_groups.clear()
        
        # Get all available commands systematically
        all_commands = self._get_all_available_commands()
        
        # Categorize commands
        self._categorize_commands(all_commands)
        
        # Create ambiguity groups
        self._create_ambiguity_groups(all_commands)

    def _get_all_available_commands(self) -> List[str]:
        """Systematically collect all available commands"""
        commands = []
        
        # Add commands from action map (includes custom commands)
        commands.extend(self._command_action_map.keys())
        
        # Add automation commands
        try:
            from iris.config.automation_command_registry import AutomationCommandRegistry
            automation_phrases = AutomationCommandRegistry.get_command_phrases()
            commands.extend(phrase.lower().strip() for phrase in automation_phrases)
        except ImportError:
            logger.warning("Could not import AutomationCommandRegistry")
        
        # Add protected terms systematically
        try:
            # Note: This would require async access to protected terms service
            # For now, add basic system commands
            system_commands = [
                "start dictation", "stop dictation", "stop", "amber",
                "grid", "golf", "cancel", "yes", "no", "mark", "go to"
            ]
            commands.extend(system_commands)
        except Exception as e:
            logger.warning(f"Could not get protected terms: {e}")
        
        # Add grid numbers
        commands.extend(str(i) for i in range(1, 101))
        
        # Add letter combinations for marks
        for letter in "abcdefghijklmnopqrstuvwxyz":
            commands.extend([f"mark {letter}", f"go to {letter}"])
        
        return list(set(cmd.lower().strip() for cmd in commands if cmd.strip()))

    def _categorize_commands(self, commands: List[str]) -> None:
        """Categorize commands for optimized timeout handling"""
        for command in commands:
            words = command.split()
            
            # Instant execution: single short words and common actions
            if len(words) == 1:
                if len(command) <= 4 or command in ["click", "enter", "escape", "space", "tab"]:
                    self._instant_commands.add(command)
                elif command.isdigit() and int(command) <= 20:
                    self._instant_commands.add(command)
                else:
                    self._quick_commands.add(command)
            
            # Quick commands: common multi-key combinations
            elif command in ["ctrl c", "ctrl v", "ctrl z", "ctrl s", "alt tab"]:
                self._quick_commands.add(command)

    def _create_ambiguity_groups(self, commands: List[str]) -> None:
        """Create ambiguity groups based on shared prefixes"""
        prefix_groups = {}
        
        # Group commands by shared prefixes
        for command in commands:
            words = command.split()
            if not words:
                continue
            
            # Check all possible prefixes
            for prefix_len in range(1, len(words)):
                prefix = " ".join(words[:prefix_len])
                
                # Find commands that share this prefix
                matching_commands = [
                    cmd for cmd in commands 
                    if cmd.startswith(prefix + " ") and cmd != command
                ]
                
                if matching_commands:
                    if prefix not in prefix_groups:
                        prefix_groups[prefix] = []
                    prefix_groups[prefix].extend([command] + matching_commands)
        
        # Create ambiguity groups from prefix groups
        for prefix, group_commands in prefix_groups.items():
            unique_commands = list(set(group_commands))
            if len(unique_commands) > 1:
                max_words = max(len(cmd.split()) for cmd in unique_commands)
                
                self._ambiguity_groups[prefix] = AmbiguityGroup(
                    prefix=prefix,
                    commands=unique_commands,
                    max_word_count=max_words,
                    timeout_ms=int(self.AMBIGUOUS_TIMEOUT * 1000)
                )
                
                logger.debug(f"Created ambiguity group '{prefix}' with {len(unique_commands)} commands")

    def get_timeout_for_text(self, recognized_text: str) -> float:
        """Get optimized timeout based on recognized text"""
        if not recognized_text:
            return self.ZERO_TIMEOUT
            
        text_lower = recognized_text.lower().strip()
        
        # Instant execution for complete, unambiguous commands
        if text_lower in self._instant_commands:
            return self.ZERO_TIMEOUT
        
        # Quick timeout for single-word commands
        if text_lower in self._quick_commands:
            return self.INSTANT_TIMEOUT
        
        # Check for numeric commands
        if text_lower.isdigit():
            return self.ZERO_TIMEOUT if int(text_lower) <= 20 else self.QUICK_TIMEOUT
        
        # Check ambiguous prefixes
        for prefix, group in self._ambiguity_groups.items():
            if text_lower.startswith(prefix):
                # If we have the complete command, execute quickly
                if text_lower in [cmd.lower() for cmd in group.commands]:
                    return self.INSTANT_TIMEOUT
                # Still ambiguous, need more time
                return self.AMBIGUOUS_TIMEOUT
        
        return self.DEFAULT_TIMEOUT

    def is_ambiguous_prefix(self, text: str) -> bool:
        """Check if text is an ambiguous prefix"""
        if not text:
            return False
            
        text_lower = text.lower().strip()
        return any(text_lower.startswith(prefix) for prefix in self._ambiguity_groups.keys())

    def get_possible_completions(self, text: str) -> List[str]:
        """Get possible command completions for the given text"""
        if not text:
            return []
            
        text_lower = text.lower().strip()
        completions = []
        
        for prefix, group in self._ambiguity_groups.items():
            if text_lower.startswith(prefix):
                completions.extend(group.commands)
        
        return completions

    def get_status(self) -> Dict:
        """Get timeout manager status for debugging"""
        return {
            'instant_commands': len(self._instant_commands),
            'quick_commands': len(self._quick_commands),
            'ambiguity_groups': len(self._ambiguity_groups),
            'total_analyzed_commands': len(self._instant_commands) + len(self._quick_commands),
            'command_action_map_size': len(self._command_action_map)
        }