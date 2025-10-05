"""
Tests for SpeechToTextService - core STT processing and mode management.
"""
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from iris.app.services.audio.stt_service import SpeechToTextService, STTMode
from iris.app.events.core_events import CommandAudioSegmentReadyEvent, DictationAudioSegmentReadyEvent
from iris.app.events.stt_events import CommandTextRecognizedEvent, DictationTextRecognizedEvent
from iris.app.events.dictation_events import DictationModeDisableOthersEvent
from iris.app.events.command_management_events import CommandMappingsUpdatedEvent


@pytest_asyncio.fixture
async def stt_service_with_mocked_engines(event_bus, app_config):
    """Create STT service with mocked engines."""
    service = SpeechToTextService(event_bus, app_config)
    
    service.vosk_engine = Mock()
    service.vosk_engine.recognize = Mock(return_value="copy")
    service.vosk_engine.set_smart_timeout_manager = Mock()
    
    service.whisper_engine = Mock()
    service.whisper_engine.recognize = Mock(return_value="this is a test")
    
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
    
    event = CommandAudioSegmentReadyEvent(
        audio_bytes=command_audio_bytes,
        sample_rate=16000
    )
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
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(DictationTextRecognizedEvent, capture_event)
    
    event = DictationAudioSegmentReadyEvent(
        audio_bytes=dictation_audio_bytes,
        sample_rate=16000
    )
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
    
    service.vosk_engine.recognize = Mock(return_value="amber")
    
    await event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=True))
    await asyncio.sleep(0.05)
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandTextRecognizedEvent, capture_event)
    
    event = CommandAudioSegmentReadyEvent(
        audio_bytes=command_audio_bytes,
        sample_rate=16000
    )
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    
    assert len(captured_events) == 1
    assert captured_events[0].text == "amber"


@pytest.mark.asyncio
async def test_non_amber_suppressed_during_dictation(stt_service_with_mocked_engines, command_audio_bytes):
    """Test that non-amber commands are suppressed during dictation mode."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus
    
    service.vosk_engine.recognize = Mock(return_value="copy")
    
    await event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=True))
    await asyncio.sleep(0.05)
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandTextRecognizedEvent, capture_event)
    
    event = CommandAudioSegmentReadyEvent(
        audio_bytes=command_audio_bytes,
        sample_rate=16000
    )
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    
    assert len(captured_events) == 0


@pytest.mark.asyncio
async def test_dictation_mode_state_changes(stt_service_with_mocked_engines):
    """Test dictation mode state transitions."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus
    
    assert service._dictation_active is False
    
    await event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=True))
    await asyncio.sleep(0.05)
    assert service._dictation_active is True
    
    await event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=False))
    await asyncio.sleep(0.05)
    assert service._dictation_active is False


@pytest.mark.asyncio
async def test_duplicate_text_filtering(stt_service_with_mocked_engines, command_audio_bytes):
    """Test that duplicate text within threshold is filtered."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus
    
    service.vosk_engine.recognize = Mock(return_value="copy")
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandTextRecognizedEvent, capture_event)
    
    event = CommandAudioSegmentReadyEvent(
        audio_bytes=command_audio_bytes,
        sample_rate=16000
    )
    
    await event_bus.publish(event)
    await asyncio.sleep(0.05)
    await event_bus.publish(event)
    await asyncio.sleep(0.05)
    
    assert len(captured_events) == 1


@pytest.mark.asyncio
async def test_command_mappings_update_propagation(stt_service_with_mocked_engines):
    """Test that command mappings updates propagate to smart timeout manager."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus
    
    service._smart_timeout_manager = Mock()
    service._smart_timeout_manager.update_command_action_map = Mock()
    
    updated_mappings = {"copy": None, "paste": None}
    event = CommandMappingsUpdatedEvent(updated_mappings=updated_mappings)
    
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    
    service._smart_timeout_manager.update_command_action_map.assert_called_once()


@pytest.mark.asyncio
async def test_empty_text_triggers_sound_recognition(stt_service_with_mocked_engines, command_audio_bytes):
    """Test that empty recognition triggers sound recognition event."""
    service = stt_service_with_mocked_engines
    event_bus = service.event_bus
    
    service.vosk_engine.recognize = Mock(return_value="")
    
    from iris.app.events.core_events import ProcessAudioChunkForSoundRecognitionEvent
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, capture_event)
    
    event = CommandAudioSegmentReadyEvent(
        audio_bytes=command_audio_bytes,
        sample_rate=16000
    )
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    
    assert len(captured_events) == 1
    assert captured_events[0].audio_chunk == command_audio_bytes

