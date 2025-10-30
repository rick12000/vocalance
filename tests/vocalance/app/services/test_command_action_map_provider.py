from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.services.command_action_map_provider import CommandActionMapProvider
from vocalance.app.services.storage.storage_models import CommandsData


@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    storage = Mock()
    storage.read = AsyncMock()
    storage.write = AsyncMock()
    return storage


@pytest.fixture
def action_map_provider(mock_storage):
    """Create CommandActionMapProvider instance."""
    return CommandActionMapProvider(storage=mock_storage)


@pytest.mark.asyncio
async def test_get_action_map_empty(action_map_provider, mock_storage):
    """Test getting action map with no custom commands."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    action_map = await action_map_provider.get_action_map()

    # Should contain default commands only
    assert len(action_map) > 0
    assert "copy" in action_map
    assert "paste" in action_map


@pytest.mark.asyncio
async def test_get_action_map_with_custom_commands(action_map_provider, mock_storage):
    """Test action map includes custom commands."""
    custom_cmd = AutomationCommand(
        command_key="my custom command",
        action_type="hotkey",
        action_value="ctrl+shift+c",
        is_custom=True,
        short_description="Custom",
        long_description="Custom command",
    )
    mock_storage.read.return_value = CommandsData(custom_commands={"my custom command": custom_cmd}, phrase_overrides={})

    action_map = await action_map_provider.get_action_map()

    assert "my custom command" in action_map
    assert action_map["my custom command"].action_value == "ctrl+shift+c"
    assert action_map["my custom command"].is_custom is True


@pytest.mark.asyncio
async def test_custom_command_overrides_default(action_map_provider, mock_storage):
    """Test custom commands override default commands with same phrase."""
    custom_copy = AutomationCommand(
        command_key="copy",
        action_type="hotkey",
        action_value="ctrl+shift+c",
        is_custom=True,
        short_description="Custom Copy",
        long_description="Custom copy command",
    )
    mock_storage.read.return_value = CommandsData(custom_commands={"copy": custom_copy}, phrase_overrides={})

    action_map = await action_map_provider.get_action_map()

    # Custom command should override default
    assert action_map["copy"].action_value == "ctrl+shift+c"
    assert action_map["copy"].is_custom is True


@pytest.mark.asyncio
async def test_phrase_overrides_apply_to_defaults(action_map_provider, mock_storage):
    """Test phrase overrides change default command phrases."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={"copy": "copy that"})

    action_map = await action_map_provider.get_action_map()

    # Original phrase should not exist
    assert "copy" not in action_map
    # Override phrase should exist with original action
    assert "copy that" in action_map
    # Action should still be the default copy action
    assert action_map["copy that"].action_value == "ctrl+c"


@pytest.mark.asyncio
async def test_phrase_overrides_normalization(action_map_provider, mock_storage):
    """Test phrase overrides are normalized (lowercase, stripped)."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={"copy": "  Copy That  "})

    action_map = await action_map_provider.get_action_map()

    # Should be normalized to lowercase and stripped
    assert "copy that" in action_map


@pytest.mark.asyncio
async def test_custom_and_overrides_together(action_map_provider, mock_storage):
    """Test custom commands and phrase overrides work together."""
    custom_cmd = AutomationCommand(
        command_key="custom action",
        action_type="hotkey",
        action_value="ctrl+alt+x",
        is_custom=True,
        short_description="Custom",
        long_description="Custom command",
    )
    mock_storage.read.return_value = CommandsData(
        custom_commands={"custom action": custom_cmd}, phrase_overrides={"paste": "paste it"}
    )

    action_map = await action_map_provider.get_action_map()

    # Custom command should exist
    assert "custom action" in action_map
    assert action_map["custom action"].is_custom is True

    # Override should exist
    assert "paste it" in action_map
    assert "paste" not in action_map

    # Other defaults should still exist
    assert "copy" in action_map


@pytest.mark.asyncio
async def test_action_map_contains_all_default_commands(action_map_provider, mock_storage):
    """Test action map includes all default commands."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    action_map = await action_map_provider.get_action_map()

    # Check for various default commands (using commands that actually exist)
    expected_commands = ["copy", "paste", "back", "select all"]
    for cmd in expected_commands:
        assert cmd in action_map, f"Default command '{cmd}' should be in action map"


@pytest.mark.asyncio
async def test_action_map_values_are_automation_commands(action_map_provider, mock_storage):
    """Test all action map values are AutomationCommand instances."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    action_map = await action_map_provider.get_action_map()

    for phrase, cmd in action_map.items():
        assert isinstance(cmd, AutomationCommand)
        assert hasattr(cmd, "command_key")
        assert hasattr(cmd, "action_type")
        assert hasattr(cmd, "action_value")


@pytest.mark.asyncio
async def test_multiple_custom_commands(action_map_provider, mock_storage):
    """Test action map handles multiple custom commands."""
    custom_commands = {}
    for i in range(5):
        cmd = AutomationCommand(
            command_key=f"custom {i}",
            action_type="hotkey",
            action_value=f"ctrl+{i}",
            is_custom=True,
            short_description=f"Custom {i}",
            long_description=f"Custom command {i}",
        )
        custom_commands[f"custom {i}"] = cmd

    mock_storage.read.return_value = CommandsData(custom_commands=custom_commands, phrase_overrides={})

    action_map = await action_map_provider.get_action_map()

    # All custom commands should be present
    for i in range(5):
        assert f"custom {i}" in action_map
        assert action_map[f"custom {i}"].is_custom is True
