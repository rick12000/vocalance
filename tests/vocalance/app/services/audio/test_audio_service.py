import asyncio
from unittest.mock import patch

import pytest

from vocalance.app.services.audio.simple_audio_service import AudioService


@pytest.fixture
def audio_service(event_bus, app_config):
    loop = asyncio.new_event_loop()
    with patch("vocalance.app.services.audio.simple_audio_service.AudioRecorder"), patch(
        "vocalance.app.services.audio.simple_audio_service.CommandAudioListener"
    ), patch("vocalance.app.services.audio.simple_audio_service.SoundAudioListener"):
        service = AudioService(event_bus, app_config, main_event_loop=loop)
        yield service
    loop.close()


@pytest.mark.asyncio
async def test_shutdown_cleans_up_resources(audio_service):
    """Shutdown nulls all heavyweight references."""
    await audio_service.shutdown()

    assert audio_service._recorder is None
    assert audio_service._command_listener is None
    assert audio_service._sound_listener is None


def test_dictation_chunk_callback_registration(audio_service):
    """Moonshine PCM ingress is wired via set_dictation_chunk_callback."""

    def _cb(_b: bytes, _sr: int) -> None:
        pass

    audio_service.set_dictation_chunk_callback(_cb)
    assert audio_service._dictation_chunk_callback is _cb
    audio_service.set_dictation_chunk_callback(None)
    assert audio_service._dictation_chunk_callback is None
