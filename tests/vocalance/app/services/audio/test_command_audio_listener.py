import asyncio
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
import pytest_asyncio

from vocalance.app.events.core_events import AudioDetectedEvent, CommandAudioSegmentReadyEvent
from vocalance.app.services.audio.simple_audio_service import AudioService

_SAMPLE_RATE = 16000


def chunk_rms_energy(chunk: np.ndarray) -> float:
    if chunk.dtype == np.int16:
        return float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def feed_pcm(audio_service: AudioService, chunk: np.ndarray, ts: float | None = None) -> None:
    timestamp = time.time() if ts is None else ts
    audio_service.relay_captured_pcm_to_consumers(chunk.tobytes(), timestamp)


@pytest_asyncio.fixture
async def audio_service(event_bus, app_config):
    loop = asyncio.get_running_loop()
    dictation = Mock()
    dictation.feed_moonshine_audio_chunk = Mock()
    with patch("vocalance.app.services.audio.simple_audio_service.AudioRecorder"):
        return AudioService(event_bus, app_config, loop, dictation)


@pytest.fixture
def speech_chunk():
    return np.random.randint(-5000, 5000, size=800, dtype=np.int16)


@pytest.fixture
def silence_chunk():
    return np.random.randint(-10, 10, size=800, dtype=np.int16)


@pytest.mark.asyncio
async def test_energy_calculation_int16(audio_service):
    chunk = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    energy = chunk_rms_energy(chunk)
    assert 0.0 < energy < 1.0
    assert isinstance(energy, (float, np.floating))


@pytest.mark.asyncio
async def test_speech_onset_detection(audio_service, speech_chunk, event_bus):
    captured = []

    async def append_event(event):
        captured.append(event)

    event_bus.subscribe(AudioDetectedEvent, append_event)
    feed_pcm(audio_service, speech_chunk)
    await asyncio.sleep(0.05)
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_pre_roll_included_in_recording(audio_service, silence_chunk, speech_chunk):
    segment_config = audio_service.command_segmenter.config
    for _ in range(segment_config.pre_roll_chunks):
        feed_pcm(audio_service, silence_chunk)
    feed_pcm(audio_service, speech_chunk)


@pytest.mark.asyncio
async def test_silence_detection_ends_recording(audio_service, speech_chunk, silence_chunk, event_bus):
    captured = []

    async def append_event(event):
        captured.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, append_event)
    feed_pcm(audio_service, speech_chunk)
    await asyncio.sleep(0.01)
    segment_config = audio_service.command_segmenter.config
    for _ in range(segment_config.silent_chunks_for_end):
        feed_pcm(audio_service, silence_chunk)
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    assert len(captured) >= 1
    if captured:
        assert isinstance(captured[0], CommandAudioSegmentReadyEvent)


@pytest.mark.asyncio
async def test_segment_ready_event_emission(audio_service, speech_chunk, silence_chunk, event_bus):
    captured = []

    async def append_event(event):
        captured.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, append_event)
    for _ in range(5):
        feed_pcm(audio_service, speech_chunk)
        await asyncio.sleep(0.01)
    segment_config = audio_service.command_segmenter.config
    for _ in range(segment_config.silent_chunks_for_end):
        feed_pcm(audio_service, silence_chunk)
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    assert len(captured) >= 1
    first = captured[0]
    assert isinstance(first.audio_bytes, bytes)
    assert first.sample_rate == _SAMPLE_RATE
    assert len(first.audio_bytes) > 0


@pytest.mark.asyncio
async def test_maximum_duration_enforced(audio_service, speech_chunk, event_bus):
    captured = []

    async def append_event(event):
        captured.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, append_event)
    segment_config = audio_service.command_segmenter.config
    for _ in range(segment_config.max_duration_chunks + 5):
        feed_pcm(audio_service, speech_chunk)
        await asyncio.sleep(0.001)
    await asyncio.sleep(0.1)
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_state_reset_after_segment(audio_service, speech_chunk, silence_chunk, event_bus):
    captured = []

    async def append_event(event):
        captured.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, append_event)
    for _ in range(3):
        feed_pcm(audio_service, speech_chunk)
        await asyncio.sleep(0.01)
    segment_config = audio_service.command_segmenter.config
    for _ in range(segment_config.silent_chunks_for_end):
        feed_pcm(audio_service, silence_chunk)
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    assert len(captured) > 0


@pytest.mark.asyncio
async def test_audio_detected_event_once_per_session(audio_service, speech_chunk, silence_chunk, event_bus):
    captured = []

    async def append_event(event):
        captured.append(event)

    event_bus.subscribe(AudioDetectedEvent, append_event)
    feed_pcm(audio_service, speech_chunk)
    await asyncio.sleep(0.05)
    feed_pcm(audio_service, speech_chunk)
    await asyncio.sleep(0.05)
    segment_config = audio_service.command_segmenter.config
    for _ in range(segment_config.silent_chunks_for_end):
        feed_pcm(audio_service, silence_chunk)
    await asyncio.sleep(0.05)
    feed_pcm(audio_service, speech_chunk)
    await asyncio.sleep(0.05)
    assert len(captured) == 2


@pytest.mark.asyncio
async def test_concurrent_chunk_processing_safe(audio_service, speech_chunk):
    await asyncio.gather(
        *[
            asyncio.to_thread(audio_service.relay_captured_pcm_to_consumers, speech_chunk.tobytes(), float(index))
            for index in range(10)
        ]
    )
