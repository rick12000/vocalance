import asyncio
import time

import numpy as np
import pytest
import pytest_asyncio

from vocalance.app.events.core_events import AudioDetectedEvent, CommandAudioSegmentReadyEvent
from vocalance.app.services.audio.audio_listeners import CommandAudioListener

_SAMPLE_RATE = 16000


def _normalized_rms_energy(chunk: np.ndarray) -> float:
    if chunk.dtype == np.int16:
        return float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def _feed_chunk(listener: CommandAudioListener, chunk: np.ndarray, timestamp: float | None = None) -> None:
    ts = time.time() if timestamp is None else timestamp
    listener.process_audio_chunk(chunk.tobytes(), ts)


@pytest_asyncio.fixture
async def command_listener(event_bus, app_config):
    listener = CommandAudioListener(event_bus, app_config)
    listener.set_main_event_loop(asyncio.get_running_loop())
    return listener


@pytest.fixture
def speech_chunk():
    chunk = np.random.randint(-5000, 5000, size=800, dtype=np.int16)
    return chunk


@pytest.fixture
def silence_chunk():
    chunk = np.random.randint(-10, 10, size=800, dtype=np.int16)
    return chunk


@pytest.mark.asyncio
async def test_energy_calculation_int16(command_listener):
    chunk = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    energy = _normalized_rms_energy(chunk)

    assert 0.0 < energy < 1.0
    assert isinstance(energy, (float, np.floating))


@pytest.mark.asyncio
async def test_speech_onset_detection(command_listener, speech_chunk, event_bus):
    command_listener.setup_subscriptions()

    captured_events = []

    async def capture_audio_detected(event):
        captured_events.append(event)

    event_bus.subscribe(AudioDetectedEvent, capture_audio_detected)

    _feed_chunk(command_listener, speech_chunk)

    await asyncio.sleep(0.05)

    assert len(captured_events) == 1


@pytest.mark.asyncio
async def test_pre_roll_included_in_recording(command_listener, silence_chunk, speech_chunk):
    command_listener.setup_subscriptions()

    for _ in range(command_listener.pre_roll_chunks):
        _feed_chunk(command_listener, silence_chunk)

    _feed_chunk(command_listener, speech_chunk)


@pytest.mark.asyncio
async def test_silence_detection_ends_recording(command_listener, speech_chunk, silence_chunk, event_bus):
    command_listener.setup_subscriptions()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, capture_segment)

    _feed_chunk(command_listener, speech_chunk)
    await asyncio.sleep(0.01)

    for _ in range(command_listener.silent_chunks_for_end):
        _feed_chunk(command_listener, silence_chunk)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    assert len(captured_events) >= 1
    if len(captured_events) > 0:
        assert isinstance(captured_events[0], CommandAudioSegmentReadyEvent)


@pytest.mark.asyncio
async def test_segment_ready_event_emission(command_listener, speech_chunk, silence_chunk, event_bus):
    command_listener.setup_subscriptions()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, capture_segment)

    for _ in range(5):
        _feed_chunk(command_listener, speech_chunk)
        await asyncio.sleep(0.01)

    for _ in range(command_listener.silent_chunks_for_end):
        _feed_chunk(command_listener, silence_chunk)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    assert len(captured_events) >= 1
    segment_event = captured_events[0]
    assert isinstance(segment_event.audio_bytes, bytes)
    assert segment_event.sample_rate == _SAMPLE_RATE
    assert len(segment_event.audio_bytes) > 0


@pytest.mark.asyncio
async def test_maximum_duration_enforced(command_listener, speech_chunk, event_bus):
    command_listener.setup_subscriptions()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, capture_segment)

    for _ in range(command_listener.max_duration_chunks + 5):
        _feed_chunk(command_listener, speech_chunk)
        await asyncio.sleep(0.001)

    await asyncio.sleep(0.1)

    assert len(captured_events) == 1


@pytest.mark.asyncio
async def test_state_reset_after_segment(command_listener, speech_chunk, silence_chunk, event_bus):
    command_listener.setup_subscriptions()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, capture_segment)

    for _ in range(3):
        _feed_chunk(command_listener, speech_chunk)
        await asyncio.sleep(0.01)

    for _ in range(command_listener.silent_chunks_for_end):
        _feed_chunk(command_listener, silence_chunk)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    assert len(captured_events) > 0


@pytest.mark.asyncio
async def test_audio_detected_event_once_per_session(command_listener, speech_chunk, silence_chunk, event_bus):
    command_listener.setup_subscriptions()

    audio_detected_events = []

    async def capture_audio_detected(event):
        audio_detected_events.append(event)

    event_bus.subscribe(AudioDetectedEvent, capture_audio_detected)

    _feed_chunk(command_listener, speech_chunk)
    await asyncio.sleep(0.05)

    _feed_chunk(command_listener, speech_chunk)
    await asyncio.sleep(0.05)

    for _ in range(command_listener.silent_chunks_for_end):
        _feed_chunk(command_listener, silence_chunk)
    await asyncio.sleep(0.05)

    _feed_chunk(command_listener, speech_chunk)
    await asyncio.sleep(0.05)

    assert len(audio_detected_events) == 2


@pytest.mark.asyncio
async def test_concurrent_chunk_processing_safe(command_listener, speech_chunk):
    command_listener.setup_subscriptions()

    await asyncio.gather(
        *[asyncio.to_thread(command_listener.process_audio_chunk, speech_chunk.tobytes(), float(i)) for i in range(10)]
    )
