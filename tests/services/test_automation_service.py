"""
Tests for AutomationService - command execution and cooldown management.
"""
import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import Mock, patch, call

from iris.services.automation_service import AutomationService
from iris.events.command_events import AutomationCommandParsedEvent
from iris.events.core_events import CommandExecutedStatusEvent
from iris.config.command_types import ExactMatchCommand, ParameterizedCommand


@pytest_asyncio.fixture
async def automation_service(event_bus, app_config):
    """Create automation service for testing."""
    service = AutomationService(event_bus, app_config)
    service.setup_subscriptions()
    
    await event_bus.start_worker()
    yield service
    await event_bus.stop_worker()


@pytest.mark.asyncio
@patch('pyautogui.hotkey')
async def test_exact_match_hotkey_execution(mock_hotkey, automation_service):
    """Test execution of exact match hotkey command."""
    service = automation_service
    event_bus = service._event_bus
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = ExactMatchCommand(
        command_key="copy",
        action_type="hotkey",
        action_value="ctrl+c",
        is_custom=False,
        short_description="Copy",
        long_description="Copy text"
    )
    
    event = AutomationCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    mock_hotkey.assert_called_once_with('ctrl', 'c')
    assert len(captured_events) == 1
    assert captured_events[0].success is True


@pytest.mark.asyncio
@patch('pyautogui.press')
async def test_key_press_execution(mock_press, automation_service):
    """Test execution of key press command."""
    service = automation_service
    event_bus = service._event_bus
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = ExactMatchCommand(
        command_key="enter",
        action_type="key",
        action_value="enter",
        is_custom=False,
        short_description="Press Enter",
        long_description="Press Enter key"
    )
    
    event = AutomationCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    mock_press.assert_called_once_with('enter')
    assert len(captured_events) == 1
    assert captured_events[0].success is True


@pytest.mark.asyncio
@patch('pyautogui.click')
async def test_click_execution(mock_click, automation_service):
    """Test execution of click command."""
    service = automation_service
    event_bus = service._event_bus
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = ExactMatchCommand(
        command_key="click",
        action_type="click",
        action_value="left_click",
        is_custom=False,
        short_description="Left Click",
        long_description="Left mouse button click"
    )
    
    event = AutomationCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    mock_click.assert_called_once_with(button='left')
    assert len(captured_events) == 1
    assert captured_events[0].success is True


@pytest.mark.asyncio
@patch('pyautogui.scroll')
async def test_scroll_execution(mock_scroll, automation_service, app_config):
    """Test execution of scroll command."""
    service = automation_service
    event_bus = service._event_bus
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = ExactMatchCommand(
        command_key="scroll up",
        action_type="scroll",
        action_value="scroll_up",
        is_custom=False,
        short_description="Scroll Up",
        long_description="Scroll page upward"
    )
    
    event = AutomationCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    mock_scroll.assert_called_once_with(app_config.scroll_amount_vertical)
    assert len(captured_events) == 1
    assert captured_events[0].success is True


@pytest.mark.asyncio
@patch('pyautogui.hotkey')
async def test_parameterized_command_execution(mock_hotkey, automation_service):
    """Test execution of parameterized command with count."""
    service = automation_service
    event_bus = service._event_bus
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = ParameterizedCommand(
        command_key="copy",
        action_type="hotkey",
        action_value="ctrl+c",
        count=3,
        is_custom=False,
        short_description="Copy",
        long_description="Copy text"
    )
    
    event = AutomationCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    assert mock_hotkey.call_count == 3
    assert len(captured_events) == 1
    assert captured_events[0].success is True


@pytest.mark.asyncio
@patch('pyautogui.hotkey')
async def test_cooldown_enforcement(mock_hotkey, automation_service):
    """Test that cooldown prevents rapid command re-execution."""
    service = automation_service
    event_bus = service._event_bus
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = ExactMatchCommand(
        command_key="copy",
        action_type="hotkey",
        action_value="ctrl+c",
        is_custom=False,
        short_description="Copy",
        long_description="Copy text"
    )
    
    event = AutomationCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    assert mock_hotkey.call_count == 1
    assert len(captured_events) == 2
    assert captured_events[0].success is True
    assert captured_events[1].success is False


@pytest.mark.asyncio
@patch('pyautogui.hotkey')
async def test_cooldown_expiration(mock_hotkey, automation_service, app_config):
    """Test that commands execute after cooldown expires."""
    service = automation_service
    event_bus = service._event_bus
    
    app_config.automation_cooldown_seconds = 0.1
    service._app_config = app_config
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = ExactMatchCommand(
        command_key="copy",
        action_type="hotkey",
        action_value="ctrl+c",
        is_custom=False,
        short_description="Copy",
        long_description="Copy text"
    )
    
    event = AutomationCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    await asyncio.sleep(0.15)
    
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    assert mock_hotkey.call_count == 2
    assert len(captured_events) == 2
    assert captured_events[0].success is True
    assert captured_events[1].success is True


@pytest.mark.asyncio
async def test_invalid_repeat_count_rejected(automation_service):
    """Test that invalid repeat counts are rejected."""
    service = automation_service
    event_bus = service._event_bus
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = ParameterizedCommand(
        command_key="copy",
        action_type="hotkey",
        action_value="ctrl+c",
        count=0,
        is_custom=False,
        short_description="Copy",
        long_description="Copy text"
    )
    
    event = AutomationCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    assert len(captured_events) == 1
    assert captured_events[0].success is False


@pytest.mark.asyncio
@patch('pyautogui.hotkey')
async def test_command_mappings_update_clears_cooldowns(mock_hotkey, automation_service):
    """Test that command mapping updates clear cooldown timers."""
    service = automation_service
    event_bus = service._event_bus
    
    from iris.events.command_management_events import CommandMappingsUpdatedEvent
    
    command = ExactMatchCommand(
        command_key="copy",
        action_type="hotkey",
        action_value="ctrl+c",
        is_custom=False,
        short_description="Copy",
        long_description="Copy text"
    )
    
    event = AutomationCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    await event_bus.publish(CommandMappingsUpdatedEvent(updated_mappings={}))
    await asyncio.sleep(0.1)
    
    await event_bus.publish(event)
    await asyncio.sleep(0.2)
    
    assert mock_hotkey.call_count == 2

