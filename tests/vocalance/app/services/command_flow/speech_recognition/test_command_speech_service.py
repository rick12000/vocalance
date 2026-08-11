import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.events.core_events import CommandAudioSegmentReadyEvent, CommandTextRecognizedEvent
from vocalance.app.events.dictation_events import (
    DictationModeDisableOthersEvent,
    DictationModifierPhraseEvent,
    DictationStopWordDetectedEvent,
)
from vocalance.app.services.command_flow.speech_recognition.command_speech_service import CommandSpeechService


@pytest.mark.asyncio
async def test_normal_mode_publishes_recognized_command_text(command_speech_service, command_audio_bytes, event_collector):
    command_speech_service.vosk_engine.recognize = AsyncMock(return_value="copy")
    recognized = event_collector(CommandTextRecognizedEvent)

    await command_speech_service.event_bus.publish(
        CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)
    )
    await asyncio.sleep(0.1)

    assert len(recognized) == 1
    assert recognized[0].text == "copy"
    assert recognized[0].engine == "vosk"
    assert recognized[0].mode == "command"


@pytest.mark.asyncio
async def test_normal_mode_suppresses_blank_recognition(command_speech_service, command_audio_bytes, event_collector):
    command_speech_service.vosk_engine.recognize = AsyncMock(return_value="   ")
    recognized = event_collector(CommandTextRecognizedEvent)

    await command_speech_service.event_bus.publish(
        CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)
    )
    await asyncio.sleep(0.1)

    assert len(recognized) == 0


@pytest.mark.asyncio
async def test_stop_trigger_during_dictation_emits_stop_and_command_text(
    command_speech_service, command_audio_bytes, event_collector
):
    command_speech_service.vosk_engine.recognize = AsyncMock(return_value="amber")
    stop_events = event_collector(DictationStopWordDetectedEvent)
    recognized = event_collector(CommandTextRecognizedEvent)

    await command_speech_service.event_bus.publish(
        DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard")
    )
    await asyncio.sleep(0.05)
    await command_speech_service.event_bus.publish(
        CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)
    )
    await asyncio.sleep(0.1)

    assert len(stop_events) == 1
    assert stop_events[0].mode == "standard"
    assert len(recognized) == 1
    assert recognized[0].text == "amber"


@pytest.mark.asyncio
async def test_modifier_phrase_during_dictation_emits_modifier_event(command_speech_service, command_audio_bytes, event_collector):
    command_speech_service.vosk_engine.recognize = AsyncMock(return_value="camel")
    modifier_events = event_collector(DictationModifierPhraseEvent)
    recognized = event_collector(CommandTextRecognizedEvent)

    await command_speech_service.event_bus.publish(
        DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard")
    )
    await asyncio.sleep(0.05)
    await command_speech_service.event_bus.publish(
        CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)
    )
    await asyncio.sleep(0.1)

    assert len(modifier_events) == 1
    assert modifier_events[0].modifier_id == "camel"
    assert len(recognized) == 0


@pytest.mark.asyncio
async def test_ordinary_command_suppressed_during_dictation(command_speech_service, command_audio_bytes, event_collector):
    command_speech_service.vosk_engine.recognize = AsyncMock(return_value="copy")
    recognized = event_collector(CommandTextRecognizedEvent)
    modifier_events = event_collector(DictationModifierPhraseEvent)

    await command_speech_service.event_bus.publish(
        DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard")
    )
    await asyncio.sleep(0.05)
    await command_speech_service.event_bus.publish(
        CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)
    )
    await asyncio.sleep(0.1)

    assert len(recognized) == 0
    assert len(modifier_events) == 0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("please toggle camel now", "camel"),
        ("use spelling mode", "spelling"),
        ("CAPITALS lock", "capitals"),
        ("nothing familiar here", None),
        (None, None),
        ("", None),
    ],
)
def test_match_modifier_phrase_detects_configured_substrings(app_config, text, expected):
    service = CommandSpeechService(Mock(), app_config)

    assert service._match_modifier_phrase(text) == expected


def test_match_modifier_phrase_prefers_longer_phrase(app_config):
    service = CommandSpeechService(Mock(), app_config)

    assert service._match_modifier_phrase("upper capitals mixed") == "capitals"
