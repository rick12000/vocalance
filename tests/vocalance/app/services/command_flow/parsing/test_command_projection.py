from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.services.command_flow.parsing.command_projection import (
    build_command_projection,
    command_with_phrase_override,
    load_action_map,
)
from vocalance.app.services.storage.storage_models import CommandsData


def test_command_with_phrase_override_applies_and_skips():
    template = AutomationCommand(command_key="copy", action_type="hotkey", action_value="ctrl+c")

    overridden = command_with_phrase_override(CommandsData(phrase_overrides={"copy": "copy that"}), template)
    assert overridden.command_key == "copy that"
    assert overridden.action_value == "ctrl+c"

    assert command_with_phrase_override(CommandsData(), template) is template


def test_build_command_projection_includes_custom_and_registry_defaults():
    custom = AutomationCommand(command_key="my cmd", action_type="key", action_value="a", is_custom=True, functional_group="Other")
    action_map, ui = build_command_projection(CommandsData(custom_commands={"my cmd": custom}))

    assert "my cmd" in action_map
    assert "copy" in action_map

    custom_ui = next(row for row in ui if row.command_key == "my cmd")
    assert custom_ui.functional_group == "Custom"


def test_build_command_projection_applies_phrase_override():
    action_map, _ = build_command_projection(CommandsData(phrase_overrides={"paste": "put that"}))

    assert "put that" in action_map
    assert "paste" not in action_map
    assert action_map["put that"].action_value == "ctrl+v"


def test_build_command_projection_ui_list_covers_action_map():
    action_map, ui = build_command_projection(CommandsData())

    assert len(ui) >= len(action_map)
    assert all(row.command_key.lower().strip() in action_map for row in ui)


@pytest.mark.asyncio
async def test_load_action_map_reads_storage_and_projects():
    custom = AutomationCommand(command_key="my custom command", action_type="hotkey", action_value="ctrl+shift+c", is_custom=True)
    storage = Mock()
    storage.read = AsyncMock(
        return_value=CommandsData(custom_commands={"my custom command": custom}, phrase_overrides={"copy": "copy that"})
    )

    action_map = await load_action_map(storage)

    storage.read.assert_awaited_once_with(model_type=CommandsData)
    assert "my custom command" in action_map
    assert "copy that" in action_map
    assert "copy" not in action_map
