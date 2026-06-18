import asyncio
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vocalance.app.events.core_events import AudioChunkCapturedEvent, AudioDeviceErrorEvent
from vocalance.app.services.capture.audio_capture_service import AudioCaptureService


@pytest.mark.asyncio
async def test_publish_chunk_emits_event_with_chunk_payload(audio_capture_service, event_collector):
    received = event_collector(AudioChunkCapturedEvent)
    pcm = (np.zeros(800, dtype=np.int16) + 7).tobytes()

    audio_capture_service._publish_chunk(pcm, 1.5)
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].pcm_bytes == pcm
    assert received[0].timestamp == 1.5
    assert received[0].sample_rate == audio_capture_service.sample_rate


@pytest.mark.asyncio
@pytest.mark.parametrize("recording", [True, False])
async def test_portaudio_callback_respects_recording_flag(audio_capture_service, event_collector, recording):
    received = event_collector(AudioChunkCapturedEvent)
    audio_capture_service._recording = recording
    indata = np.zeros((800, 1), dtype=np.int16) + 3

    audio_capture_service._portaudio_callback(indata, 800, None, None)
    await asyncio.sleep(0.05)

    assert len(received) == (1 if recording else 0)


@pytest.mark.asyncio
async def test_device_error_published_only_once(audio_capture_service, event_collector):
    received = event_collector(AudioDeviceErrorEvent)

    audio_capture_service._publish_device_error("boom")
    audio_capture_service._publish_device_error("boom again")
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].error_message == "boom"


@pytest.mark.asyncio
async def test_start_failure_publishes_device_error_and_clears_recording(event_bus, app_config, event_collector):
    received = event_collector(AudioDeviceErrorEvent)
    loop = asyncio.get_running_loop()

    with patch(
        "vocalance.app.services.capture.audio_capture_service.sd.InputStream",
        side_effect=RuntimeError("nope"),
    ):
        service = AudioCaptureService(event_bus=event_bus, config=app_config, main_event_loop=loop)
        service.start()
        await asyncio.sleep(0.05)

    assert len(received) == 1
    assert service._recording is False


@pytest.mark.asyncio
async def test_shutdown_stops_active_stream(audio_capture_service):
    audio_capture_service._recording = True
    fake_stream = Mock()
    fake_stream.active = True
    audio_capture_service._stream = fake_stream

    await audio_capture_service.shutdown()

    fake_stream.stop.assert_called_once()
    fake_stream.close.assert_called_once()
    assert audio_capture_service._stream is None
