from unittest.mock import patch

import pytest

from vocalance.app.config.command_types import (
    DictationAmendStartCommand,
    DictationHiddenStartCommand,
    DictationSmartStartCommand,
    DictationStartCommand,
    DictationStopCommand,
    DictationTypeCommand,
    DictationVisualStartCommand,
    ErrorResult,
    ExactMatchCommand,
    GridSelectCommand,
    GridShowCommand,
    MarkCreateCommand,
    MarkDeleteCommand,
    MarkExecuteCommand,
    MarkResetCommand,
    MarkVisualizeCancelCommand,
    MarkVisualizeCommand,
    NoMatchResult,
    ParameterizedCommand,
    PauseCommand,
    ResumeCommand,
)
from vocalance.app.services.command_flow.parsing.text_command_parse import (
    grid_show_from_phrase,
    parse_automation_commands,
    parse_dictation,
    parse_full_command_text,
    parse_grid_commands,
    parse_mark_commands,
    parse_mark_execute_fallback,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("green", DictationStartCommand),
        ("amber", DictationStopCommand),
        ("type", DictationTypeCommand),
        ("smart green", DictationSmartStartCommand),
        ("visual green", DictationVisualStartCommand),
        ("hidden green", DictationHiddenStartCommand),
        ("amend", DictationAmendStartCommand),
    ],
)
def test_parse_dictation_triggers(parser_triggers, text, expected):
    assert isinstance(parse_dictation(text, parser_triggers), expected)


def test_parse_mark_create_captures_cursor_position(parser_triggers):
    with patch(
        "vocalance.app.services.command_flow.parsing.text_command_parse.pyautogui.position",
        return_value=(100, 200),
    ):
        result = parse_mark_commands("mark home", parser_triggers)
    assert isinstance(result, MarkCreateCommand)
    assert result.label == "home"
    assert (result.x, result.y) == (100.0, 200.0)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("delete mark home", MarkDeleteCommand),
        ("delete mark left side", ErrorResult),
    ],
)
def test_parse_mark_delete(parser_triggers, text, expected):
    assert isinstance(parse_mark_commands(text, parser_triggers), expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("show marks", MarkVisualizeCommand),
        ("visualize marks", MarkVisualizeCommand),
        ("reset marks", MarkResetCommand),
        ("clear all marks", MarkResetCommand),
        ("cancel marks", MarkVisualizeCancelCommand),
        ("hide marks", MarkVisualizeCancelCommand),
    ],
)
def test_parse_mark_keyword_commands(parser_triggers, text, expected):
    assert isinstance(parse_mark_commands(text, parser_triggers), expected)


def test_grid_show_from_phrase_variants():
    assert grid_show_from_phrase("go", "go", "click").num_rects is None
    assert grid_show_from_phrase("go 9", "go", "click").num_rects == 9
    assert grid_show_from_phrase("go", "go", "drag").click_mode == "drag"
    assert isinstance(grid_show_from_phrase("go banana", "go", "click"), ErrorResult)
    assert grid_show_from_phrase("hover", "go", "click") is None


@pytest.mark.parametrize("text,mode", [("go", "click"), ("hover", "hover"), ("move", "drag")])
def test_parse_grid_commands_show_modes(parser_triggers, parser_action_map, text, mode):
    result = parse_grid_commands(text, parser_triggers, parser_action_map)
    assert isinstance(result, GridShowCommand)
    assert result.click_mode == mode


def test_parse_grid_commands_selects_spoken_number(parser_triggers, parser_action_map):
    result = parse_grid_commands("five", parser_triggers, parser_action_map)
    assert isinstance(result, GridSelectCommand)
    assert result.selected_number == 5


def test_parse_automation_exact_match(parser_action_map):
    result = parse_automation_commands("copy", parser_action_map)
    assert isinstance(result, ExactMatchCommand)
    assert result.command_key == "copy"
    assert result.action_value == "ctrl+c"


def test_parse_automation_parameterized_with_count(parser_action_map):
    result = parse_automation_commands("scroll down three", parser_action_map)
    assert isinstance(result, ParameterizedCommand)
    assert result.command_key == "scroll down"
    assert result.count == 3


@pytest.mark.parametrize("text,expected", [("home", MarkExecuteCommand), ("two words", NoMatchResult)])
def test_parse_mark_execute_fallback(text, expected):
    assert isinstance(parse_mark_execute_fallback(text), expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("pause", PauseCommand),
        ("resume", ResumeCommand),
        ("green", DictationStartCommand),
        ("delete mark home", MarkDeleteCommand),
        ("go", GridShowCommand),
        ("copy", ExactMatchCommand),
        ("scroll down three", ParameterizedCommand),
        ("home", MarkExecuteCommand),
        ("", NoMatchResult),
    ],
)
def test_parse_full_command_text_pipeline(parser_triggers, parser_action_map, text, expected):
    assert isinstance(parse_full_command_text(text, parser_triggers, parser_action_map), expected)
