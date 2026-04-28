import asyncio
from unittest.mock import Mock, patch

import numpy as np
import pytest
import pytest_asyncio

from vocalance.app.events.core_events import AudioChunkCapturedEvent, AudioDeviceErrorEvent
from vocalance.app.services.audio.audio_capture_service import AudioCaptureService


@pytest_asyncio.fixture
async def audio_capture(event_bus, app_config):
    loop = asyncio.get_running_loop()
    with patch("vocalance.app.services.audio.audio_capture_service.sd.InputStream"):
        service = AudioCaptureService(event_bus=event_bus, config=app_config, main_event_loop=loop)
        yield service


@pytest.mark.asyncio
async def test_chunks_publish_audio_chunk_captured_event(audio_capture, event_bus):
    captured = []

    async def on_event(event):
        captured.append(event)

    event_bus.subscribe(AudioChunkCapturedEvent, on_event)

    pcm = (np.zeros(800, dtype=np.int16) + 7).tobytes()
    audio_capture._publish_chunk(pcm, 1.5)
    await asyncio.sleep(0.05)

    assert len(captured) == 1
    assert captured[0].pcm_bytes == pcm
    assert captured[0].timestamp == 1.5
    assert captured[0].sample_rate == audio_capture.sample_rate


@pytest.mark.asyncio
async def test_portaudio_callback_skips_when_not_recording(audio_capture, event_bus):
    captured = []

    async def on_event(event):
        captured.append(event)

    event_bus.subscribe(AudioChunkCapturedEvent, on_event)

    indata = np.zeros((800, 1), dtype=np.int16)
    audio_capture._portaudio_callback(indata, 800, None, None)
    await asyncio.sleep(0.05)

    assert captured == []


@pytest.mark.asyncio
async def test_portaudio_callback_publishes_when_recording(audio_capture, event_bus):
    captured = []

    async def on_event(event):
        captured.append(event)

    event_bus.subscribe(AudioChunkCapturedEvent, on_event)

    audio_capture._recording = True

    indata = np.zeros((800, 1), dtype=np.int16) + 3
    audio_capture._portaudio_callback(indata, 800, None, None)
    await asyncio.sleep(0.05)

    assert len(captured) == 1
    assert captured[0].pcm_bytes == indata.tobytes()


@pytest.mark.asyncio
async def test_device_error_published_once(audio_capture, event_bus):
    captured = []

    async def on_event(event):
        captured.append(event)

    event_bus.subscribe(AudioDeviceErrorEvent, on_event)

    audio_capture._publish_device_error("boom")
    audio_capture._publish_device_error("boom again")
    await asyncio.sleep(0.05)

    assert len(captured) == 1
    assert captured[0].error_message == "boom"


@pytest.mark.asyncio
async def test_start_failure_publishes_device_error(event_bus, app_config):
    loop = asyncio.get_running_loop()
    captured = []

    async def on_event(event):
        captured.append(event)

    event_bus.subscribe(AudioDeviceErrorEvent, on_event)

    with patch("vocalance.app.services.audio.audio_capture_service.sd.InputStream", side_effect=RuntimeError("nope")):
        service = AudioCaptureService(event_bus=event_bus, config=app_config, main_event_loop=loop)
        service.start()
        await asyncio.sleep(0.05)

    assert len(captured) == 1
    assert service._recording is False


@pytest.mark.asyncio
async def test_stop_idempotent(audio_capture):
    audio_capture.stop()
    audio_capture.stop()


@pytest.mark.asyncio
async def test_shutdown_stops_stream(audio_capture):
    audio_capture._recording = True

    fake_stream = Mock()
    fake_stream.active = True
    audio_capture._stream = fake_stream

    await audio_capture.shutdown()

    fake_stream.stop.assert_called_once()
    fake_stream.close.assert_called_once()
    assert audio_capture._stream is None
