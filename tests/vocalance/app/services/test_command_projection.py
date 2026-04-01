"""Tests for shared command projection (action map + UI list)."""

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.services.commands.projection import build_command_projection
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
