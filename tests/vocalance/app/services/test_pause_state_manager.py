import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from vocalance.app.config.command_types import PauseCommand, ResumeCommand
from vocalance.app.events.command_events import AutomationCommandParsedEvent, SystemControlCommandParsedEvent
from vocalance.app.events.core_events import CommandTextRecognizedEvent
from vocalance.app.services.commands.parser import CentralizedCommandParser
from vocalance.app.services.pause_state_manager import PauseStateManager
from vocalance.app.services.storage.storage_models import CommandsData


@pytest.fixture
def command_parser_storage():
    storage = Mock()
    storage.read = AsyncMock(return_value=CommandsData(custom_commands={}, phrase_overrides={}))
    return storage


@pytest_asyncio.fixture
async def pause_state_manager(event_bus):
    manager = PauseStateManager(event_bus=event_bus)
    yield manager


@pytest_asyncio.fixture
async def command_parser_with_pause(event_bus, app_config, command_parser_storage, pause_state_manager):
    parser = CentralizedCommandParser(
        event_bus=event_bus,
        app_config=app_config,
        storage=command_parser_storage,
        pause_state_manager=pause_state_manager,
    )
    await parser.initialize()
    yield parser


@pytest.mark.asyncio
async def test_pause_command_parsing(command_parser_with_pause):
    """Test that pause command is correctly parsed."""
    parser = command_parser_with_pause
    event_bus = parser.event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(SystemControlCommandParsedEvent, capture_event)

    # Send pause command
    event = CommandTextRecognizedEvent(text="pause", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, PauseCommand)


@pytest.mark.asyncio
async def test_resume_command_parsing(command_parser_with_pause):
    """Test that resume command is correctly parsed."""
    parser = command_parser_with_pause
    event_bus = parser.event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(SystemControlCommandParsedEvent, capture_event)

    # Send resume command
    event = CommandTextRecognizedEvent(text="resume", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, ResumeCommand)


@pytest.mark.asyncio
async def test_pause_blocks_automation_commands(command_parser_with_pause):
    """Test that automation commands are blocked when paused."""
    parser = command_parser_with_pause
    event_bus = parser.event_bus
    pause_manager = parser.pause_state_manager

    automation_events = []

    async def capture_automation(event):
        automation_events.append(event)

    event_bus.subscribe(AutomationCommandParsedEvent, capture_automation)

    # Send regular command before pause - should work
    await event_bus.publish(CommandTextRecognizedEvent(text="click", engine="vosk"))
    await asyncio.sleep(0.1)
    assert len(automation_events) == 1

    # Pause the application
    await event_bus.publish(CommandTextRecognizedEvent(text="pause", engine="vosk"))
    await asyncio.sleep(0.1)

    # Verify paused state
    assert pause_manager.is_paused() is True

    # Send regular command after pause - should be blocked
    await event_bus.publish(CommandTextRecognizedEvent(text="click", engine="vosk"))
    await asyncio.sleep(0.1)
    assert len(automation_events) == 1  # Still only 1 event, second was blocked


@pytest.mark.asyncio
async def test_resume_unblocks_commands(command_parser_with_pause):
    """Test that resume command unblocks automation commands."""
    parser = command_parser_with_pause
    event_bus = parser.event_bus
    pause_manager = parser.pause_state_manager

    automation_events = []

    async def capture_automation(event):
        automation_events.append(event)

    event_bus.subscribe(AutomationCommandParsedEvent, capture_automation)

    # Pause the application
    await event_bus.publish(CommandTextRecognizedEvent(text="pause", engine="vosk"))
    await asyncio.sleep(0.1)
    assert pause_manager.is_paused() is True

    # Resume the application
    await event_bus.publish(CommandTextRecognizedEvent(text="resume", engine="vosk"))
    await asyncio.sleep(0.1)
    assert pause_manager.is_paused() is False

    # Send regular command after resume - should work
    await event_bus.publish(CommandTextRecognizedEvent(text="click", engine="vosk"))
    await asyncio.sleep(0.1)
    assert len(automation_events) == 1


@pytest.mark.asyncio
async def test_resume_command_works_when_paused(command_parser_with_pause):
    """Test that resume command can be executed even when paused."""
    parser = command_parser_with_pause
    event_bus = parser.event_bus
    pause_manager = parser.pause_state_manager

    system_control_events = []

    async def capture_system_control(event):
        system_control_events.append(event)

    event_bus.subscribe(SystemControlCommandParsedEvent, capture_system_control)

    # Pause the application
    await event_bus.publish(CommandTextRecognizedEvent(text="pause", engine="vosk"))
    await asyncio.sleep(0.1)
    assert pause_manager.is_paused() is True
    assert len(system_control_events) == 1  # Pause event

    # Resume should work even when paused
    await event_bus.publish(CommandTextRecognizedEvent(text="resume", engine="vosk"))
    await asyncio.sleep(0.1)
    assert pause_manager.is_paused() is False
    assert len(system_control_events) == 2  # Pause + Resume events


@pytest.mark.asyncio
async def test_pause_state_manager_initial_state(pause_state_manager):
    """Test that pause state manager starts in unpaused state."""
    assert pause_state_manager.is_paused() is False


@pytest.mark.asyncio
async def test_pause_state_toggle(pause_state_manager):
    """Test pause state can be toggled."""
    event_bus = pause_state_manager._event_bus

    # Initially not paused
    assert pause_state_manager.is_paused() is False

    # Pause
    await event_bus.publish(SystemControlCommandParsedEvent(command=PauseCommand(), source="test"))
    await asyncio.sleep(0.1)
    assert pause_state_manager.is_paused() is True

    # Resume
    await event_bus.publish(SystemControlCommandParsedEvent(command=ResumeCommand(), source="test"))
    await asyncio.sleep(0.1)
    assert pause_state_manager.is_paused() is False
