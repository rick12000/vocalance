import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from vocalance.app.events.core_events import CommandAudioSegmentReadyEvent, CommandTextRecognizedEvent
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.stt.stt_service import SpeechToTextService


@pytest_asyncio.fixture
async def stt_service_with_mocked_engines(event_bus, app_config):
    """Create STT service with mocked engines."""
    service = SpeechToTextService(event_bus, app_config)

    service.vosk_engine = Mock()
    service.vosk_engine.recognize = AsyncMock(return_value="copy")
    service.moonshine_engine = Mock()
    service.moonshine_engine.recognize = AsyncMock(return_value="this is a test")

    yield service


@pytest.mark.asyncio
async def test_command_audio_processing_normal_mode(stt_service_with_mocked_engines, command_audio_bytes):
    """Test command audio processing in normal command mode."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(CommandTextRecognizedEvent, capture_event)

    event = CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert captured_events[0].text == "copy"
    assert captured_events[0].engine == "vosk"


@pytest.mark.asyncio
async def test_amber_trigger_detection_during_dictation(stt_service_with_mocked_engines, command_audio_bytes):
    """Test that amber triggers are detected during dictation mode."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus

    service.vosk_engine.recognize = AsyncMock(return_value="amber")

    await event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard"))
    await asyncio.sleep(0.05)

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(CommandTextRecognizedEvent, capture_event)

    event = CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert captured_events[0].text == "amber"


@pytest.mark.asyncio
async def test_non_amber_suppressed_during_dictation(stt_service_with_mocked_engines, command_audio_bytes):
    """Test that non-amber commands are suppressed during dictation mode."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus

    service.vosk_engine.recognize = AsyncMock(return_value="copy")

    await event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard"))
    await asyncio.sleep(0.05)

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(CommandTextRecognizedEvent, capture_event)

    event = CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 0


@pytest.mark.asyncio
async def test_dictation_mode_state_changes(stt_service_with_mocked_engines):
    """Test dictation mode state transitions."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus

    assert service._dictation_active is False

    await event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard"))
    await asyncio.sleep(0.05)
    assert service._dictation_active is True

    await event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive"))
    await asyncio.sleep(0.05)
    assert service._dictation_active is False


@pytest.mark.asyncio
async def test_duplicate_text_filtering(stt_service_with_mocked_engines, command_audio_bytes):
    """Test that duplicate text within threshold is filtered."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus

    service.vosk_engine.recognize = AsyncMock(return_value="copy")

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(CommandTextRecognizedEvent, capture_event)

    event = CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)

    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    # STT service doesn't apply command interval gating - CentralizedCommandParser handles it
    # So we expect 2 events
    assert len(captured_events) == 2


@pytest.mark.asyncio
async def test_empty_text_does_not_trigger_sound_recognition_from_stt(stt_service_with_mocked_engines, command_audio_bytes):
    """Test that empty recognition does NOT trigger sound recognition from STT service.

    Empty text forwarding is handled directly by the sound audio listener, not the STT service.
    This prevents duplicate events.
    """
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus

    service.vosk_engine.recognize = AsyncMock(return_value="")

    from vocalance.app.events.core_events import ProcessAudioChunkForSoundRecognitionEvent

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, capture_event)

    event = CommandAudioSegmentReadyEvent(audio_bytes=command_audio_bytes, sample_rate=16000)
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    # STT service should NOT forward empty text to sound recognition
    assert len(captured_events) == 0


def test_match_modifier_phrase_detects_configured_substrings(app_config):
    """Vosk path: substring match against sorted (longest-first) modifier phrases."""
    service = SpeechToTextService(Mock(), app_config)
    assert service._match_modifier_phrase("please toggle camel now") == "camel"
    assert service._match_modifier_phrase("use spelling mode") == "spelling"
    assert service._match_modifier_phrase("CAPITALS lock") == "capitals"
    assert service._match_modifier_phrase(None) is None
    assert service._match_modifier_phrase("") is None
    assert service._match_modifier_phrase("nothing familiar here") is None


def test_match_modifier_phrase_prefers_longer_phrase(app_config):
    """When multiple phrases match, the longest phrase wins (table order)."""
    service = SpeechToTextService(Mock(), app_config)
    assert service._match_modifier_phrase("upper capitals mixed") == "capitals"
