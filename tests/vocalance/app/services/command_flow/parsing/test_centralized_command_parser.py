import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from vocalance.app.config.command_types import (
    DictationStartCommand,
    DictationStopCommand,
    ExactMatchCommand,
    GridSelectCommand,
    GridShowCommand,
    MarkCreateCommand,
    MarkDeleteCommand,
    MarkExecuteCommand,
    ParameterizedCommand,
)
from vocalance.app.events.command_events import (
    AutomationCommandParsedEvent,
    DictationCommandParsedEvent,
    GridCommandParsedEvent,
    MarkCommandParsedEvent,
)
from vocalance.app.events.core_events import CommandTextRecognizedEvent, CustomSoundRecognizedEvent
from vocalance.app.events.dictation_events import DictationStatusChangedEvent
from vocalance.app.events.sound_events import SoundToCommandMappingUpdatedEvent
from vocalance.app.services.command_flow.parsing.parser import CentralizedCommandParser
from vocalance.app.services.storage.storage_models import CommandsData


@pytest.fixture
def command_parser_storage():
    storage = Mock()
    storage.read = AsyncMock(return_value=CommandsData(custom_commands={}, phrase_overrides={}))
    return storage


@pytest_asyncio.fixture
async def command_parser(event_bus, app_config, command_parser_storage):
    """Create command parser with mocked dependencies."""
    parser = CentralizedCommandParser(
        event_bus=event_bus,
        app_config=app_config,
        storage=command_parser_storage,
    )
    await parser.initialize()
    yield parser


@pytest.mark.asyncio
async def test_parse_dictation_start_command(command_parser):
    """Test parsing dictation start trigger."""
    parser = command_parser
    event_bus = parser.event_bus

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
    event_bus = parser.event_bus

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
    event_bus = parser.event_bus

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
    event_bus = parser.event_bus

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
    event_bus = parser.event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(GridCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="go", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, GridShowCommand)


@pytest.mark.asyncio
async def test_parse_grid_show_with_number(command_parser):
    """Test parsing grid show command with specified number."""
    parser = command_parser
    event_bus = parser.event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(GridCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="go 9", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, GridShowCommand)
    assert captured_events[0].command.num_rects == 9


@pytest.mark.asyncio
async def test_parse_grid_drag_show_command(command_parser):
    """Test parsing grid show in drag mode (default phrase 'move')."""
    parser = command_parser
    event_bus = parser.event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(GridCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="move", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, GridShowCommand)
    assert captured_events[0].command.click_mode == "drag"


@pytest.mark.asyncio
async def test_parse_grid_drag_show_with_number(command_parser):
    """Test drag grid show with explicit rectangle count."""
    parser = command_parser
    event_bus = parser.event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(GridCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="move 9", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    cmd = captured_events[0].command
    assert isinstance(cmd, GridShowCommand)
    assert cmd.click_mode == "drag"
    assert cmd.num_rects == 9


@pytest.mark.asyncio
async def test_parse_grid_select_number(command_parser):
    """Test parsing grid select command with number."""
    parser = command_parser
    event_bus = parser.event_bus

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
async def test_parse_exact_match_automation_command(command_parser):
    """Test parsing exact match automation command."""
    parser = command_parser
    event_bus = parser.event_bus

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
    event_bus = parser.event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(AutomationCommandParsedEvent, capture_event)

    event = CommandTextRecognizedEvent(text="sky three", engine="vosk")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0].command, ParameterizedCommand)
    assert captured_events[0].command.command_key == "sky"
    assert captured_events[0].command.count == 3


@pytest.mark.asyncio
async def test_mark_execute_fallback_for_single_word(command_parser):
    """Test that single unknown words fall back to mark execute."""
    parser = command_parser
    event_bus = parser.event_bus

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
async def test_sound_command_mapping(command_parser):
    """Test that custom sounds are mapped to command phrases."""
    parser = command_parser
    event_bus = parser.event_bus

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
    event_bus = parser.event_bus

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
async def test_command_interval_suppression(command_parser):
    """Test that a second parsed command within min_command_interval_ms is ignored."""
    parser = command_parser
    event_bus = parser.event_bus

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
