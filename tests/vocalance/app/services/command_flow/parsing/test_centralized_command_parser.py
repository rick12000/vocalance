from unittest.mock import Mock

import pytest

from vocalance.app.config.command_types import ExactMatchCommand, ResumeCommand, SoundTrainCommand
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
