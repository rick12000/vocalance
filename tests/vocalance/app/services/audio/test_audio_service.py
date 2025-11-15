import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vocalance.app.events.core_events import RecordingTriggerEvent
from vocalance.app.events.dictation_events import AudioModeChangeRequestEvent
from vocalance.app.services.audio.simple_audio_service import AudioService


@pytest.fixture
def mock_recorder():
    recorder = Mock()
    recorder.start = Mock()
    recorder.stop = Mock()
    recorder.is_recording = Mock(return_value=False)
    recorder.is_active = Mock(return_value=True)
    return recorder


@pytest.fixture
def mock_listener():
    listener = Mock()
    listener.setup_subscriptions = Mock()
    return listener


@pytest.fixture
def audio_service(event_bus, app_config):
    loop = asyncio.new_event_loop()
    with patch("vocalance.app.services.audio.simple_audio_service.AudioRecorder"), patch(
        "vocalance.app.services.audio.simple_audio_service.CommandAudioListener"
    ), patch("vocalance.app.services.audio.simple_audio_service.DictationAudioListener"), patch(
        "vocalance.app.services.audio.simple_audio_service.SoundAudioListener"
    ):
        service = AudioService(event_bus, app_config, main_event_loop=loop)
        yield service
    loop.close()


@pytest.mark.asyncio
async def test_shutdown_cleans_up_resources(audio_service):
    """Test that shutdown properly cleans up all resources."""
    await audio_service.shutdown()

    assert audio_service._recorder is None
    assert audio_service._command_listener is None
    assert audio_service._dictation_listener is None
    assert audio_service._sound_listener is None


@pytest.mark.asyncio
async def test_recording_trigger_handling(audio_service):
    """Test that recording triggers are handled without errors."""
    # Test start trigger
    event = RecordingTriggerEvent(trigger="start")
    await audio_service._handle_recording_trigger(event)

    # Test stop trigger
    event = RecordingTriggerEvent(trigger="stop")
    await audio_service._handle_recording_trigger(event)


@pytest.mark.asyncio
async def test_audio_mode_change_handling(audio_service):
    """Test that audio mode change requests are handled without errors."""
    event = AudioModeChangeRequestEvent(mode="dictation", reason="user_command")
    await audio_service._handle_audio_mode_change_request(event)


@pytest.mark.asyncio
async def test_dictation_silent_chunks_update_propagates(audio_service):
    """Test that dictation silent chunks updates are propagated to listener."""
    audio_service._dictation_listener.update_silent_chunks_threshold = AsyncMock()

    await audio_service.on_dictation_silent_chunks_updated(20)

    audio_service._dictation_listener.update_silent_chunks_threshold.assert_called_once_with(20)


@pytest.mark.asyncio
async def test_dictation_silent_chunks_update_handles_missing_listener(audio_service):
    """Test that dictation silent chunks updates handle missing listener gracefully."""
    audio_service._dictation_listener = None

    # Should not raise an exception
    await audio_service.on_dictation_silent_chunks_updated(20)
