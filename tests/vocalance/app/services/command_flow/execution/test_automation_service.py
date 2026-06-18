import time
from unittest.mock import Mock, patch

import pytest
from conftest import skip_if_headless

skip_if_headless()

from vocalance.app.config.command_types import ExactMatchCommand, ParameterizedCommand
from vocalance.app.events.command_events import AutomationCommandParsedEvent
from vocalance.app.events.command_management_events import CommandMappingsUpdatedEvent


@pytest.mark.parametrize(
    "action_value, method, button",
    [
        ("click", "click", "left"),
        ("left_click", "click", "left"),
        ("right_click", "click", "right"),
        ("double_click", "doubleClick", None),
        ("triple_click", "tripleClick", None),
    ],
)
@pytest.mark.asyncio
async def test_create_action_function_click_dispatch(automation_service, action_value, method, button):
    with patch(f"pyautogui.{method}") as mock_method:
        automation_service.create_action_function("click", action_value)()
    if button is None:
        mock_method.assert_called_once_with()
    else:
        mock_method.assert_called_once_with(button=button)


@pytest.mark.parametrize("action_value", ["ctrl+c", "ctrl c"])
@pytest.mark.asyncio
async def test_create_action_function_hotkey_splits_keys(automation_service, action_value):
    with patch("pyautogui.hotkey") as mock_hotkey:
        automation_service.create_action_function("hotkey", action_value)()
    mock_hotkey.assert_called_once_with("ctrl", "c")


@pytest.mark.asyncio
async def test_create_action_function_key_presses_value(automation_service):
    with patch("pyautogui.press") as mock_press:
        automation_service.create_action_function("key", "enter")()
    mock_press.assert_called_once_with("enter")


@pytest.mark.parametrize("action_type, action_value", [("scroll", "sideways"), ("click", "quadruple_click")])
@pytest.mark.asyncio
async def test_create_action_function_returns_none_for_unsupported(automation_service, action_type, action_value):
    assert automation_service.create_action_function(action_type, action_value) is None


@pytest.mark.parametrize("count", [1, 10, 1000])
@pytest.mark.asyncio
async def test_run_action_repeats_exactly_count_times(automation_service, count):
    action = Mock()
    automation_service.run_action(action, count)
    assert action.call_count == count


@pytest.mark.asyncio
async def test_execute_key_sequence_handles_combos_and_single_keys(automation_service):
    with patch("pyautogui.hotkey") as mock_hotkey, patch("pyautogui.press") as mock_press, patch("time.sleep"):
        automation_service.execute_key_sequence(["ctrl+c", "enter", "alt+tab"])
    assert mock_hotkey.call_count == 2
    mock_press.assert_called_once_with("enter")


@pytest.mark.parametrize("direction, sign", [("up", 1), ("down", -1)])
@pytest.mark.asyncio
async def test_execute_animated_scroll_distributes_total_clicks(automation_service, direction, sign):
    cfg = automation_service.config.automation_service
    with patch("pyautogui.scroll") as mock_scroll, patch("time.sleep"):
        automation_service.execute_animated_scroll(direction)
    assert mock_scroll.call_count == cfg.scroll_animation_steps
    assert sum(call.args[0] for call in mock_scroll.call_args_list) == sign * cfg.scroll_total_clicks


@pytest.mark.asyncio
async def test_handle_executes_action_and_records_cooldown(automation_service):
    command = ExactMatchCommand(command_key="copy", action_type="hotkey", action_value="ctrl+c")
    with patch("pyautogui.hotkey") as mock_hotkey:
        await automation_service.handle_automation_command_parsed(AutomationCommandParsedEvent(command=command, source="speech"))
    mock_hotkey.assert_called_once_with("ctrl", "c")
    assert "copy" in automation_service.cooldown_timers


@pytest.mark.parametrize("count, expected_calls", [(0, 0), (3, 3), (999, 100)])
@pytest.mark.asyncio
async def test_handle_respects_repeat_count_bounds(automation_service, count, expected_calls):
    command = ParameterizedCommand(command_key="copy", action_type="hotkey", action_value="ctrl+c", count=count)
    with patch("pyautogui.hotkey") as mock_hotkey:
        await automation_service.handle_automation_command_parsed(AutomationCommandParsedEvent(command=command, source="speech"))
    assert mock_hotkey.call_count == expected_calls


@pytest.mark.asyncio
async def test_handle_blocks_second_command_within_cooldown(automation_service):
    command = ExactMatchCommand(command_key="copy", action_type="hotkey", action_value="ctrl+c")
    event = AutomationCommandParsedEvent(command=command, source="speech")
    with patch("pyautogui.hotkey") as mock_hotkey:
        await automation_service.handle_automation_command_parsed(event)
        await automation_service.handle_automation_command_parsed(event)
    assert mock_hotkey.call_count == 1


@pytest.mark.asyncio
async def test_command_mappings_update_clears_cooldowns(automation_service):
    automation_service.cooldown_timers["copy"] = time.time()
    await automation_service.handle_command_mappings_updated(CommandMappingsUpdatedEvent(success=True, updated_mappings=[]))
    assert automation_service.cooldown_timers == {}
