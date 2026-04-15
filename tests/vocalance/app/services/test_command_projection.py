"""Tests for command projection and loading the parser action map from storage."""

from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.services.commands.utilities.command_projection import build_command_projection, load_action_map
from vocalance.app.services.storage.storage_models import CommandsData


def test_build_command_projection_custom_and_override():
    data = CommandsData(
        custom_commands={
            "my cmd": AutomationCommand(
                command_key="my cmd",
                action_type="key",
                action_value="a",
                is_custom=True,
                functional_group="Other",
            )
        },
        phrase_overrides={"paste": "put that"},
    )
    action_map, ui = build_command_projection(data)

    assert "my cmd" in action_map
    assert action_map["my cmd"].is_custom is True

    custom_ui = next(x for x in ui if x.command_key == "my cmd")
    assert custom_ui.functional_group == "Custom"

    assert "put that" in action_map
    paste_cmd = action_map["put that"]
    assert paste_cmd.action_value == "ctrl+v"


def test_build_command_projection_action_map_matches_ui_count_for_simple_case():
    data = CommandsData()
    action_map, ui = build_command_projection(data)
    assert len(ui) >= len(action_map)
    for row in ui:
        key = row.command_key.lower().strip()
        assert key in action_map


@pytest.fixture
def mock_storage():
    storage = Mock()
    storage.read = AsyncMock()
    storage.write = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_load_action_map_empty(mock_storage):
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    action_map = await load_action_map(mock_storage)

    assert len(action_map) > 0
    assert "copy" in action_map
    assert "paste" in action_map


@pytest.mark.asyncio
async def test_load_action_map_with_custom_commands(mock_storage):
    custom_cmd = AutomationCommand(
        command_key="my custom command",
        action_type="hotkey",
        action_value="ctrl+shift+c",
        is_custom=True,
        short_description="Custom",
        long_description="Custom command",
    )
    mock_storage.read.return_value = CommandsData(custom_commands={"my custom command": custom_cmd}, phrase_overrides={})

    action_map = await load_action_map(mock_storage)

    assert "my custom command" in action_map
    assert action_map["my custom command"].action_value == "ctrl+shift+c"
    assert action_map["my custom command"].is_custom is True


@pytest.mark.asyncio
async def test_load_action_map_custom_overrides_default(mock_storage):
    custom_copy = AutomationCommand(
        command_key="copy",
        action_type="hotkey",
        action_value="ctrl+shift+c",
        is_custom=True,
        short_description="Custom Copy",
        long_description="Custom copy command",
    )
    mock_storage.read.return_value = CommandsData(custom_commands={"copy": custom_copy}, phrase_overrides={})

    action_map = await load_action_map(mock_storage)

    assert action_map["copy"].action_value == "ctrl+shift+c"
    assert action_map["copy"].is_custom is True


@pytest.mark.asyncio
async def test_load_action_map_phrase_overrides_apply(mock_storage):
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={"copy": "copy that"})

    action_map = await load_action_map(mock_storage)

    assert "copy" not in action_map
    assert "copy that" in action_map
    assert action_map["copy that"].action_value == "ctrl+c"


@pytest.mark.asyncio
async def test_load_action_map_phrase_override_normalization(mock_storage):
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={"copy": "  Copy That  "})

    action_map = await load_action_map(mock_storage)

    assert "copy that" in action_map


@pytest.mark.asyncio
async def test_load_action_map_custom_and_overrides(mock_storage):
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

    action_map = await load_action_map(mock_storage)

    assert "custom action" in action_map
    assert action_map["custom action"].is_custom is True
    assert "paste it" in action_map
    assert "paste" not in action_map
    assert "copy" in action_map


@pytest.mark.asyncio
async def test_load_action_map_contains_expected_defaults(mock_storage):
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    action_map = await load_action_map(mock_storage)

    expected_commands = ["copy", "paste", "back", "select all"]
    for cmd in expected_commands:
        assert cmd in action_map, f"Default command '{cmd}' should be in action map"


@pytest.mark.asyncio
async def test_load_action_map_values_are_automation_commands(mock_storage):
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    action_map = await load_action_map(mock_storage)

    for phrase, cmd in action_map.items():
        assert isinstance(cmd, AutomationCommand)
        assert hasattr(cmd, "command_key")
        assert hasattr(cmd, "action_type")
        assert hasattr(cmd, "action_value")


@pytest.mark.asyncio
async def test_load_action_map_multiple_custom_commands(mock_storage):
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

    action_map = await load_action_map(mock_storage)

    for i in range(5):
        assert f"custom {i}" in action_map
        assert action_map[f"custom {i}"].is_custom is True
