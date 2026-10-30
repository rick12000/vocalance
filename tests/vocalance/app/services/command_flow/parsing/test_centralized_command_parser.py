from unittest.mock import Mock

import pytest
from conftest import skip_if_headless

skip_if_headless()

from vocalance.app.config.command_types import (
    ExactMatchCommand,
    GridShowCommand,
    ParameterizedCommand,
    ResumeCommand,
    SoundTrainCommand,
)
from vocalance.app.events.command_events import AutomationCommandParsedEvent, SystemControlCommandParsedEvent
from vocalance.app.events.core_events import CustomSoundRecognizedEvent
from vocalance.app.services.command_flow.parsing.parser import CentralizedCommandParser


@pytest.mark.asyncio
async def test_process_text_input_publishes_parsed_command(command_parser):
    await command_parser.process_text_input("copy", source="stt")

    command_parser.event_bus.publish.assert_awaited_once()
    event = command_parser.event_bus.publish.await_args.args[0]
    assert isinstance(event, AutomationCommandParsedEvent)
    assert isinstance(event.command, ExactMatchCommand)
    assert event.source == "stt"


@pytest.mark.asyncio
async def test_process_text_input_rate_limits_rapid_commands(command_parser):
    await command_parser.process_text_input("copy", source="stt")
    await command_parser.process_text_input("copy", source="stt")

    assert command_parser.event_bus.publish.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   "])
async def test_process_text_input_ignores_blank(command_parser, text):
    await command_parser.process_text_input(text, source="stt")

    command_parser.event_bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_pause_gate_blocks_non_resume_but_allows_resume(mock_event_bus, app_config, mock_storage_service):
    pause_manager = Mock()
    pause_manager.is_paused.return_value = True
    parser = CentralizedCommandParser(
        event_bus=mock_event_bus,
        app_config=app_config,
        storage=mock_storage_service,
        pause_state_manager=pause_manager,
    )

    await parser.process_text_input("copy", source="stt")
    mock_event_bus.publish.assert_not_awaited()

    await parser.process_text_input("resume", source="stt")
    mock_event_bus.publish.assert_awaited_once()
    event = mock_event_bus.publish.await_args.args[0]
    assert isinstance(event, SystemControlCommandParsedEvent)
    assert isinstance(event.command, ResumeCommand)


@pytest.mark.asyncio
async def test_publish_command_event_rejects_unregistered_command(command_parser):
    with pytest.raises(ValueError):
        await command_parser.publish_command_event(SoundTrainCommand(sound_label="x"), source="stt")


@pytest.mark.asyncio
async def test_handle_custom_sound_resolves_label_mapping(command_parser):
    command_parser.sound_to_command_mapping = {"whistle": "copy"}

    await command_parser.handle_custom_sound_recognized(
        CustomSoundRecognizedEvent(label="whistle", confidence=0.9, mapped_command=None)
    )

    command_parser.event_bus.publish.assert_awaited_once()
    event = command_parser.event_bus.publish.await_args.args[0]
    assert isinstance(event.command, ExactMatchCommand)
    assert event.source == "sound"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("copy", ExactMatchCommand),
        ("sky 3", ParameterizedCommand),
        ("go", GridShowCommand),
        ("go 9", GridShowCommand),
    ],
)
@pytest.mark.asyncio
async def test_repeat_replays_automation_and_go(command_parser, text, expected):
    command_parser.app_config.command_parser.min_command_interval_ms = 0

    await command_parser.process_text_input(text, source="stt")
    await command_parser.process_text_input("repeat", source="stt")

    replay = command_parser.event_bus.publish.await_args.args[0]
    assert isinstance(replay.command, expected)
    assert replay.source == "stt"
    if isinstance(replay.command, ParameterizedCommand):
        assert replay.command.count == 3
    if text == "go 9":
        assert replay.command.num_rects == 9


@pytest.mark.parametrize("text", ["green", "show marks", "five", "hover", "home"])
@pytest.mark.asyncio
async def test_repeat_ignores_non_automation_and_non_go_commands(command_parser, text):
    command_parser.app_config.command_parser.min_command_interval_ms = 0

    await command_parser.process_text_input("copy", source="stt")
    await command_parser.process_text_input(text, source="stt")
    await command_parser.process_text_input("repeat", source="stt")

    replay = command_parser.event_bus.publish.await_args.args[0]
    assert isinstance(replay.command, ExactMatchCommand)
    assert replay.command.command_key == "copy"


@pytest.mark.asyncio
async def test_repeat_with_no_history_publishes_nothing(command_parser):
    await command_parser.process_text_input("repeat", source="stt")
    command_parser.event_bus.publish.assert_not_awaited()
