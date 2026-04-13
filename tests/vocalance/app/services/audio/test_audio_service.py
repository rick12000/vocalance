import asyncio
from unittest.mock import Mock, patch

import pytest

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
    ), patch("vocalance.app.services.audio.simple_audio_service.SoundAudioListener"):
        service = AudioService(event_bus, app_config, main_event_loop=loop)
        yield service
    loop.close()


def test_shutdown_cleans_up_resources(audio_service):
    """Test that shutdown properly cleans up all resources."""
    audio_service.shutdown()

    assert audio_service._recorder is None
    assert audio_service._command_listener is None
    assert audio_service._sound_listener is None


def test_audio_mode_change_handling(audio_service):
    """Test that audio mode change requests are handled without errors."""
    event = AudioModeChangeRequestEvent(mode="dictation", reason="user_command")
    audio_service._handle_audio_mode_change_request(event)


def test_dictation_chunk_callback_registration(audio_service):
    """Moonshine PCM ingress is wired via set_dictation_chunk_callback (recorder thread, not the bus)."""

    def _cb(_b: bytes, _sr: int) -> None:
        pass

    audio_service.set_dictation_chunk_callback(_cb)
    assert audio_service._dictation_chunk_callback is _cb
    audio_service.set_dictation_chunk_callback(None)
    assert audio_service._dictation_chunk_callback is None
