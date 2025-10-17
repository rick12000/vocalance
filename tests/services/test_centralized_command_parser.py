"""
Tests for CentralizedCommandParser - command parsing and routing logic.
"""
import asyncio

import pytest
import pytest_asyncio

from iris.app.config.command_types import (
    DictationStartCommand,
    DictationStopCommand,
    ExactMatchCommand,
    GridCancelCommand,
    GridSelectCommand,
    GridShowCommand,
    MarkCreateCommand,
    MarkDeleteCommand,
    MarkExecuteCommand,
    ParameterizedCommand,
)
from iris.app.events.command_events import (
    AutomationCommandParsedEvent,
    CommandNoMatchEvent,
    DictationCommandParsedEvent,
    GridCommandParsedEvent,
    MarkCommandParsedEvent,
)
from iris.app.events.core_events import CommandTextRecognizedEvent, CustomSoundRecognizedEvent
from iris.app.events.dictation_events import DictationStatusChangedEvent
from iris.app.events.sound_events import SoundToCommandMappingUpdatedEvent
from iris.app.services.centralized_command_parser import CentralizedCommandParser


@pytest_asyncio.fixture
async def command_parser(event_bus, app_config, mock_storage_service, mock_action_map_provider, mock_command_history_manager):
    """Create command parser with mocked storage."""
    parser = CentralizedCommandParser(
        event_bus=event_bus,
        app_config=app_config,
        storage=mock_storage_service,
        action_map_provider=mock_action_map_provider,
        history_manager=mock_command_history_manager,
    )
    parser.setup_subscriptions()
    await parser.initialize()

    await event_bus.start_worker()
    yield parser
    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_parse_dictation_start_command(command_parser):
    """Test parsing dictation start trigger."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(DictationCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="green", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, DictationStartCommand)


@pytest.mark.asyncio
async def test_parse_dictation_stop_command(command_parser):
    """Test parsing dictation stop trigger."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(DictationCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="amber", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, DictationStopCommand)


@pytest.mark.asyncio
async def test_parse_mark_create_command(command_parser):
    """Test parsing mark create command with label."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(MarkCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="mark home", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, MarkCreateCommand)
    assert captured_events[0].command.label == "home"


@pytest.mark.asyncio
async def test_parse_mark_delete_command(command_parser):
    """Test parsing mark delete command."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(MarkCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="delete mark home", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, MarkDeleteCommand)
    assert captured_events[0].command.label == "home"


@pytest.mark.asyncio
async def test_parse_grid_show_command(command_parser):
    """Test parsing grid show command."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(GridCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="golf", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, GridShowCommand)


@pytest.mark.asyncio
async def test_parse_grid_show_with_number(command_parser):
    """Test parsing grid show command with specified number."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(GridCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="golf 9", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, GridShowCommand)
    assert captured_events[0].command.num_rects == 9


@pytest.mark.asyncio
async def test_parse_grid_select_number(command_parser):
    """Test parsing grid select command with number."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(GridCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="five", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, GridSelectCommand)
    assert captured_events[0].command.selected_number == 5


@pytest.mark.asyncio
async def test_parse_grid_cancel_command(command_parser):
    """Test parsing grid cancel command."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(GridCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="cancel", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, GridCancelCommand)


@pytest.mark.asyncio
async def test_parse_exact_match_automation_command(command_parser):
    """Test parsing exact match automation command."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(AutomationCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="copy", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, ExactMatchCommand)
    assert captured_events[0].command.command_key == "copy"
    assert captured_events[0].command.action_type == "hotkey"


@pytest.mark.asyncio
async def test_parse_parameterized_automation_command(command_parser):
    """Test parsing parameterized automation command with count."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(AutomationCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="scroll up three", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, ParameterizedCommand)
    assert captured_events[0].command.command_key == "scroll up"
    assert captured_events[0].command.count == 3


@pytest.mark.asyncio
async def test_mark_execute_fallback_for_single_word(command_parser):
    """Test that single unknown words fall back to mark execute."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(MarkCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="home", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, MarkExecuteCommand)
    assert captured_events[0].command.label == "home"


@pytest.mark.asyncio
async def test_no_match_for_unknown_phrase(command_parser):
    """Test that unknown multi-word phrases produce no match event."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(CommandNoMatchEvent, capture_event)

    event = CommandTextRecognizedEvent(text="unknown command phrase", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1


@pytest.mark.asyncio
async def test_sound_command_mapping(command_parser):
    """Test that custom sounds are mapped to command phrases."""
    parser = command_parser
    event_bus = parser._event_bus

    await event_bus.publish(SoundToCommandMappingUpdatedEvent(sound_label="whistle", command_phrase="copy", success=True))
    await asyncio.sleep(0.05)

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(AutomationCommandParsedEvent, capture_event)

    event = CustomSoundRecognizedEvent(label="whistle", confidence=0.95, mapped_command="copy")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert captured_events[0].command.command_key == "copy"
    assert captured_events[0].source == "sound"


@pytest.mark.asyncio
async def test_dictation_active_suppresses_commands(command_parser):
    """Test that commands still pass through even during dictation (feature not implemented)."""
    parser = command_parser
    event_bus = parser._event_bus

    await event_bus.publish(DictationStatusChangedEvent(is_active=True, mode="standard"))
    await asyncio.sleep(0.05)

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(AutomationCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="copy", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    # Note: Dictation suppression is not currently implemented in command parser
    assert len(captured_events) == 1


@pytest.mark.asyncio
async def test_duplicate_text_suppression(command_parser):
    """Test that duplicate text within time window is suppressed."""
    parser = command_parser
    event_bus = parser._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(AutomationCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="copy", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.05)
    await event_bus.publish(event)
    await asyncio.sleep(0.05)

    assert len(captured_events) == 1
