import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from vocalance.app.events.core_events import (
    CommandAudioSegmentReadyEvent,
    CommandTextRecognizedEvent,
    DictationAudioSegmentReadyEvent,
    DictationTextRecognizedEvent,
)
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.stt.stt_service import SpeechToTextService


@pytest_asyncio.fixture
async def stt_service_with_mocked_engines(event_bus, app_config):
    """Create STT service with mocked engines."""
    service = SpeechToTextService(event_bus, app_config)

    service.vosk_engine = Mock()
    service.vosk_engine.recognize = AsyncMock(return_value="copy")

    service.whisper_engine = Mock()
    service.whisper_engine.recognize = AsyncMock(return_value="this is a test")

    service._engines_initialized = True
    service.setup_subscriptions()

    await event_bus.start_worker()
    yield service
    await event_bus.stop_worker()


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
async def test_dictation_audio_processing(stt_service_with_mocked_engines, dictation_audio_bytes):
    """Test dictation audio processing with Whisper engine."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus

    # Set service to dictation mode
    service._dictation_active = True

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(DictationTextRecognizedEvent, capture_event)

    event = DictationAudioSegmentReadyEvent(audio_bytes=dictation_audio_bytes, sample_rate=16000)
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert captured_events[0].text == "this is a test"
    assert captured_events[0].engine == "whisper"
    assert captured_events[0].mode == "dictation"


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

    # STT service doesn't do duplicate filtering - CentralizedCommandParser handles it
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


@pytest.mark.asyncio
async def test_context_segments_management(stt_service_with_mocked_engines, dictation_audio_bytes):
    """Test that context segments are managed with maxlen of 10."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus

    for i in range(15):
        service.whisper_engine.recognize = AsyncMock(return_value=f"segment {i}")
        event = DictationAudioSegmentReadyEvent(audio_bytes=dictation_audio_bytes, sample_rate=16000)
        await event_bus.publish(event)
        await asyncio.sleep(0.05)

    async with service._context_lock:
        assert len(service._context_segments) <= 10
