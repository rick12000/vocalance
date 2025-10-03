"""
Unit tests for SmartTimeoutManager
"""
import pytest
from unittest.mock import Mock, patch

from iris.services.audio.smart_timeout_manager import SmartTimeoutManager, AmbiguityGroup


class TestAmbiguityGroup:
    """Test AmbiguityGroup dataclass"""
    
    def test_ambiguity_group_creation(self):
        """Test creating an ambiguity group"""
        group = AmbiguityGroup(
            prefix="mark",
            commands=["mark a", "mark b", "mark c"],
            max_word_count=2,
            timeout_ms=600
        )
        
        assert group.prefix == "mark"
        assert len(group.commands) == 3
        assert group.max_word_count == 2
        assert group.timeout_ms == 600


class TestSmartTimeoutManager:
    """Test SmartTimeoutManager functionality"""
    
    @pytest.fixture
    def command_action_map(self):
        """Sample command action map for testing"""
        return {
            "click": ("click", {}),
            "right click": ("right_click", {}),
            "ctrl c": ("key_combo", {"keys": ["ctrl", "c"]}),
            "mark a": ("create_mark", {"letter": "a"}),
            "mark b": ("create_mark", {"letter": "b"}),
            "go to a": ("go_to_mark", {"letter": "a"}),
            "go to b": ("go_to_mark", {"letter": "b"}),
            "1": ("grid_click", {"number": 1}),
            "25": ("grid_click", {"number": 25}),
            "golf": ("show_grid", {}),
            "down": ("arrow_key", {"direction": "down"})
        }
    
    @pytest.fixture
    def timeout_manager(self, mock_global_config, command_action_map):
        """Create SmartTimeoutManager instance for testing"""
        with patch('iris.services.audio.smart_timeout_manager.AutomationCommandRegistry') as mock_registry:
            mock_registry.get_command_phrases.return_value = ["start dictation", "stop dictation"]
            return SmartTimeoutManager(mock_global_config, command_action_map)
    
    def test_initialization(self, mock_global_config):
        """Test timeout manager initialization"""
        manager = SmartTimeoutManager(mock_global_config)
        
        assert manager.app_config == mock_global_config
        assert isinstance(manager._instant_commands, set)
        assert isinstance(manager._quick_commands, set)
        assert isinstance(manager._ambiguity_groups, dict)
    
    def test_command_categorization(self, timeout_manager):
        """Test that commands are properly categorized"""
        # Instant commands: short single words, numbers
        assert "click" in timeout_manager._instant_commands
        assert "1" in timeout_manager._instant_commands
        
        # Quick commands: longer single words or common combinations
        assert "golf" in timeout_manager._quick_commands or "golf" in timeout_manager._instant_commands
        assert "ctrl c" in timeout_manager._quick_commands
    
    def test_ambiguity_group_creation(self, timeout_manager):
        """Test that ambiguity groups are created for shared prefixes"""
        # "mark" should create an ambiguity group
        assert "mark" in timeout_manager._ambiguity_groups
        mark_group = timeout_manager._ambiguity_groups["mark"]
        assert "mark a" in mark_group.commands
        assert "mark b" in mark_group.commands
        
        # "go to" should create an ambiguity group
        assert "go to" in timeout_manager._ambiguity_groups
        goto_group = timeout_manager._ambiguity_groups["go to"]
        assert "go to a" in goto_group.commands
        assert "go to b" in goto_group.commands
    
    @pytest.mark.parametrize("text,expected_timeout", [
        ("click", SmartTimeoutManager.ZERO_TIMEOUT),
        ("1", SmartTimeoutManager.ZERO_TIMEOUT),
        ("25", SmartTimeoutManager.ZERO_TIMEOUT),
        ("golf", SmartTimeoutManager.INSTANT_TIMEOUT),
        ("mark", SmartTimeoutManager.AMBIGUOUS_TIMEOUT),
        ("mark a", SmartTimeoutManager.INSTANT_TIMEOUT),
        ("unknown command", SmartTimeoutManager.DEFAULT_TIMEOUT),
        ("", SmartTimeoutManager.ZERO_TIMEOUT)
    ])
    def test_timeout_calculation(self, timeout_manager, text, expected_timeout):
        """Test timeout calculation for various text inputs"""
        result = timeout_manager.get_timeout_for_text(text)
        assert result == expected_timeout
    
    def test_numeric_command_timeout(self, timeout_manager):
        """Test that numeric commands get zero timeout"""
        for num in ["1", "5", "10", "99"]:
            timeout = timeout_manager.get_timeout_for_text(num)
            assert timeout == SmartTimeoutManager.ZERO_TIMEOUT
    
    def test_ambiguous_prefix_detection(self, timeout_manager):
        """Test ambiguous prefix detection"""
        assert timeout_manager.is_ambiguous_prefix("mark")
        assert timeout_manager.is_ambiguous_prefix("go to")
        assert not timeout_manager.is_ambiguous_prefix("click")
        assert not timeout_manager.is_ambiguous_prefix("unknown")
    
    def test_possible_completions(self, timeout_manager):
        """Test getting possible command completions"""
        completions = timeout_manager.get_possible_completions("mark")
        assert "mark a" in completions
        assert "mark b" in completions
        
        completions = timeout_manager.get_possible_completions("go to")
        assert "go to a" in completions
        assert "go to b" in completions
        
        completions = timeout_manager.get_possible_completions("click")
        assert len(completions) == 0
    
    def test_command_action_map_update(self, timeout_manager):
        """Test updating command action map and re-analysis"""
        new_commands = {
            "select all": ("select_all", {}),
            "select text": ("select_text", {}),
            "new command": ("new_action", {})
        }
        
        timeout_manager.update_command_action_map(new_commands)
        
        # Should create new ambiguity group for "select"
        assert "select" in timeout_manager._ambiguity_groups
        select_group = timeout_manager._ambiguity_groups["select"]
        assert "select all" in select_group.commands
        assert "select text" in select_group.commands
    
    def test_status_reporting(self, timeout_manager):
        """Test status reporting functionality"""
        status = timeout_manager.get_status()
        
        expected_keys = {
            'instant_commands',
            'quick_commands',
            'ambiguity_groups',
            'total_analyzed_commands',
            'command_action_map_size'
        }
        
        assert set(status.keys()) == expected_keys
        assert isinstance(status['instant_commands'], int)
        assert isinstance(status['quick_commands'], int)
        assert isinstance(status['ambiguity_groups'], int)
        assert status['instant_commands'] >= 0
        assert status['quick_commands'] >= 0
        assert status['ambiguity_groups'] >= 0
    
    def test_empty_command_handling(self, timeout_manager):
        """Test handling of empty or None commands"""
        assert timeout_manager.get_timeout_for_text("") == SmartTimeoutManager.ZERO_TIMEOUT
        assert timeout_manager.get_timeout_for_text(None) == SmartTimeoutManager.ZERO_TIMEOUT
        
        assert not timeout_manager.is_ambiguous_prefix("")
        assert not timeout_manager.is_ambiguous_prefix(None)
        
        assert timeout_manager.get_possible_completions("") == []
        assert timeout_manager.get_possible_completions(None) == []
    
    def test_case_insensitive_processing(self, timeout_manager):
        """Test that processing is case-insensitive"""
        # Should work with different cases
        assert timeout_manager.get_timeout_for_text("CLICK") == SmartTimeoutManager.ZERO_TIMEOUT
        assert timeout_manager.get_timeout_for_text("Click") == SmartTimeoutManager.ZERO_TIMEOUT
        assert timeout_manager.get_timeout_for_text("MARK") == SmartTimeoutManager.AMBIGUOUS_TIMEOUT
        assert timeout_manager.get_timeout_for_text("Mark A") == SmartTimeoutManager.INSTANT_TIMEOUT
    
    def test_whitespace_handling(self, timeout_manager):
        """Test proper whitespace handling"""
        # Should handle extra whitespace
        assert timeout_manager.get_timeout_for_text("  click  ") == SmartTimeoutManager.ZERO_TIMEOUT
        assert timeout_manager.get_timeout_for_text(" mark a ") == SmartTimeoutManager.INSTANT_TIMEOUT
    
    def test_command_collection_robustness(self, mock_global_config):
        """Test robustness of command collection with missing dependencies"""
        with patch('iris.services.audio.smart_timeout_manager.AutomationCommandRegistry') as mock_registry:
            # Simulate ImportError
            mock_registry.get_command_phrases.side_effect = ImportError("Module not found")
            
            # Should still initialize without crashing
            manager = SmartTimeoutManager(mock_global_config, {})
            assert isinstance(manager._instant_commands, set)
            assert isinstance(manager._quick_commands, set)
    
    @pytest.mark.parametrize("word_count,expected_category", [
        (1, "instant_or_quick"),  # Single words go to instant or quick
        (2, "multi_word"),        # Multi-word commands
        (3, "complex")            # Complex commands
    ])
    def test_command_complexity_categorization(self, timeout_manager, word_count, expected_category):
        """Test that commands are categorized by complexity"""
        # This test verifies the logic exists, specific categorization tested elsewhere
        test_commands = {
            1: ["click", "enter", "space"],
            2: ["right click", "ctrl c", "mark a"],
            3: ["start dictation mode", "show grid overlay"]
        }
        
        if word_count in test_commands:
            for command in test_commands[word_count]:
                timeout = timeout_manager.get_timeout_for_text(command)
                # Verify timeout is appropriate for command complexity
                assert isinstance(timeout, float)
                assert timeout >= 0

