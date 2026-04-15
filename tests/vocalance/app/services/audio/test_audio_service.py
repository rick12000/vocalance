import asyncio
from unittest.mock import Mock, patch

import pytest

from vocalance.app.services.audio.simple_audio_service import AudioService


@pytest.fixture
def audio_service(event_bus, app_config):
    loop = asyncio.new_event_loop()
    dictation = Mock()
    dictation.feed_moonshine_audio_chunk = Mock()
    with patch("vocalance.app.services.audio.simple_audio_service.AudioRecorder"):
        service = AudioService(
            event_bus,
            app_config,
            main_event_loop=loop,
            dictation=dictation,
        )
        yield service
    loop.close()


@pytest.mark.asyncio
async def test_shutdown_cleans_up_resources(audio_service):
    await audio_service.shutdown()
    assert audio_service.recorder is None
    assert audio_service.command_segmenter is None
    assert audio_service.sound_segmenter is None
    assert audio_service.chunk_analyzer is None
