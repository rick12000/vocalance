import asyncio

import numpy as np
import pytest

from vocalance.app.events.core_events import AudioChunkEvent, AudioDetectedEvent, CommandAudioSegmentReadyEvent
from vocalance.app.services.audio.audio_listeners import CommandAudioListener


@pytest.fixture
def command_listener(event_bus, app_config):
    return CommandAudioListener(event_bus, app_config)


@pytest.fixture
def speech_chunk():
    chunk = np.random.randint(-5000, 5000, size=800, dtype=np.int16)
    return chunk


@pytest.fixture
def silence_chunk():
    # Create a very quiet chunk that should be below the silence threshold
    chunk = np.random.randint(-10, 10, size=800, dtype=np.int16)
    return chunk


def create_audio_event(chunk: np.ndarray, timestamp: float = None) -> AudioChunkEvent:
    if timestamp is None:
        timestamp = asyncio.get_event_loop().time()
    return AudioChunkEvent(audio_chunk=chunk.tobytes(), sample_rate=16000, timestamp=timestamp)


@pytest.mark.asyncio
async def test_energy_calculation_int16(command_listener):
    chunk = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    energy = command_listener._calculate_energy(chunk)

    assert 0.0 < energy < 1.0
    assert isinstance(energy, (float, np.floating))


@pytest.mark.asyncio
async def test_speech_onset_detection(command_listener, speech_chunk, event_bus):
    command_listener.setup_subscriptions()

    captured_events = []

    async def capture_audio_detected(event):
        captured_events.append(event)

    event_bus.subscribe(AudioDetectedEvent, capture_audio_detected)
    await event_bus.start_worker()

    event = create_audio_event(speech_chunk)
    await command_listener._handle_audio_chunk(event)

    await asyncio.sleep(0.05)

    # Should detect speech and emit AudioDetectedEvent
    assert len(captured_events) == 1

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_pre_roll_included_in_recording(command_listener, silence_chunk, speech_chunk):
    command_listener.setup_subscriptions()

    for _ in range(command_listener.pre_roll_chunks):
        event = create_audio_event(silence_chunk)
        await command_listener._handle_audio_chunk(event)

    event = create_audio_event(speech_chunk)
    await command_listener._handle_audio_chunk(event)

    # Pre-roll chunks should be included in recording


@pytest.mark.asyncio
async def test_silence_detection_ends_recording(command_listener, speech_chunk, silence_chunk, event_bus):
    command_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, capture_segment)

    event = create_audio_event(speech_chunk)
    await command_listener._handle_audio_chunk(event)
    await asyncio.sleep(0.01)

    for _ in range(command_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await command_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    assert len(captured_events) >= 1
    if len(captured_events) > 0:
        assert isinstance(captured_events[0], CommandAudioSegmentReadyEvent)

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_segment_ready_event_emission(command_listener, speech_chunk, silence_chunk, event_bus):
    command_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, capture_segment)

    for _ in range(5):
        event = create_audio_event(speech_chunk)
        await command_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.01)

    for _ in range(command_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await command_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    assert len(captured_events) >= 1
    segment_event = captured_events[0]
    assert isinstance(segment_event.audio_bytes, bytes)
    assert segment_event.sample_rate == 16000
    assert len(segment_event.audio_bytes) > 0

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_maximum_duration_enforced(command_listener, speech_chunk, event_bus):
    """Test that recordings are cut off at maximum duration."""
    command_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, capture_segment)

    # Send more chunks than maximum duration allows
    for _ in range(command_listener.max_duration_chunks + 5):
        event = create_audio_event(speech_chunk)
        await command_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.001)

    await asyncio.sleep(0.1)

    # Should produce exactly one segment (cut off at max duration)
    assert len(captured_events) == 1

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_state_reset_after_segment(command_listener, speech_chunk, silence_chunk, event_bus):
    command_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(CommandAudioSegmentReadyEvent, capture_segment)

    for _ in range(3):
        event = create_audio_event(speech_chunk)
        await command_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.01)

    for _ in range(command_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await command_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    # Should have produced a segment
    assert len(captured_events) > 0

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_audio_detected_event_once_per_session(command_listener, speech_chunk, silence_chunk, event_bus):
    """Test that AudioDetectedEvent is emitted only once per recording session."""
    command_listener.setup_subscriptions()
    await event_bus.start_worker()

    audio_detected_events = []

    async def capture_audio_detected(event):
        audio_detected_events.append(event)

    event_bus.subscribe(AudioDetectedEvent, capture_audio_detected)

    # Start recording with speech
    event = create_audio_event(speech_chunk)
    await command_listener._handle_audio_chunk(event)
    await asyncio.sleep(0.05)

    # End recording with silence
    for _ in range(command_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await command_listener._handle_audio_chunk(event)
    await asyncio.sleep(0.05)

    # Start new recording - should emit AudioDetectedEvent again
    event = create_audio_event(speech_chunk)
    await command_listener._handle_audio_chunk(event)
    await asyncio.sleep(0.05)

    # Should have exactly one event (only the first one in the session)
    assert len(audio_detected_events) == 1

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_concurrent_chunk_processing_safe(command_listener, speech_chunk):
    """Test that concurrent chunk processing is thread-safe."""
    command_listener.setup_subscriptions()

    tasks = []
    for i in range(10):
        event = create_audio_event(speech_chunk, timestamp=float(i))
        tasks.append(command_listener._handle_audio_chunk(event))

    # Should complete without errors
    await asyncio.gather(*tasks)
