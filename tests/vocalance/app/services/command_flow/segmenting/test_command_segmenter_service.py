import asyncio
import time

import numpy as np
import pytest
import pytest_asyncio

from vocalance.app.events.core_events import AudioChunkCapturedEvent, AudioDetectedEvent, CommandAudioSegmentReadyEvent
from vocalance.app.services.command_flow.segmenting.command_segmenter_service import CommandSegmenterService

_SAMPLE_RATE = 16000


def make_event(pcm: np.ndarray, ts: float | None = None) -> AudioChunkCapturedEvent:
    return AudioChunkCapturedEvent(
        pcm_bytes=pcm.tobytes(),
        timestamp=time.time() if ts is None else ts,
        sample_rate=_SAMPLE_RATE,
    )


@pytest_asyncio.fixture
async def segmenter(event_bus, app_config):
    return CommandSegmenterService(event_bus=event_bus, config=app_config)


@pytest.fixture
def speech_chunk():
    return np.random.randint(-5000, 5000, size=800, dtype=np.int16)


@pytest.fixture
def silence_chunk():
    return np.random.randint(-10, 10, size=800, dtype=np.int16)


@pytest.mark.asyncio
async def test_speech_onset_publishes_audio_detected(segmenter, speech_chunk, event_bus):
    captured = []

    async def on_event(event):
        captured.append(event)

    event_bus.subscribe(AudioDetectedEvent, on_event)

    await event_bus.publish(make_event(speech_chunk))
    await asyncio.sleep(0.05)

    assert len(captured) == 1


@pytest.mark.asyncio
async def test_silence_finalizes_clip(segmenter, speech_chunk, silence_chunk, event_bus):
    captured = []

    async def on_event(event):
        captured.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, on_event)

    await event_bus.publish(make_event(speech_chunk))
    for _ in range(segmenter.segmenter.config.silent_chunks_for_end):
        await event_bus.publish(make_event(silence_chunk))
    await asyncio.sleep(0.2)

    assert len(captured) >= 1
    first = captured[0]
    assert isinstance(first.audio_bytes, bytes)
    assert first.sample_rate == _SAMPLE_RATE


@pytest.mark.asyncio
async def test_max_duration_finalizes_clip(segmenter, speech_chunk, event_bus):
    captured = []

    async def on_event(event):
        captured.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, on_event)

    for _ in range(segmenter.segmenter.config.max_duration_chunks + 5):
        await event_bus.publish(make_event(speech_chunk))
    await asyncio.sleep(0.2)

    assert len(captured) == 1


@pytest.mark.asyncio
async def test_audio_detected_emitted_per_session(segmenter, speech_chunk, silence_chunk, event_bus):
    captured = []

    async def on_event(event):
        captured.append(event)

    event_bus.subscribe(AudioDetectedEvent, on_event)

    await event_bus.publish(make_event(speech_chunk))
    await asyncio.sleep(0.05)
    await event_bus.publish(make_event(speech_chunk))
    await asyncio.sleep(0.05)
    for _ in range(segmenter.segmenter.config.silent_chunks_for_end):
        await event_bus.publish(make_event(silence_chunk))
    await asyncio.sleep(0.05)
    await event_bus.publish(make_event(speech_chunk))
    await asyncio.sleep(0.05)

    assert len(captured) == 2
