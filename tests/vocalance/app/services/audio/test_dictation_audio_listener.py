import asyncio

import numpy as np
import pytest

from vocalance.app.events.core_events import AudioChunkEvent, DictationAudioSegmentReadyEvent
from vocalance.app.services.audio.audio_listeners import DictationAudioListener


@pytest.fixture
def dictation_listener(event_bus, app_config):
    return DictationAudioListener(event_bus, app_config)


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
async def test_longer_silence_timeout_than_command(dictation_listener, app_config):
    command_timeout = app_config.vad.command_silent_chunks_for_end
    dictation_timeout = dictation_listener.silent_chunks_for_end

    assert dictation_timeout > command_timeout


@pytest.mark.asyncio
async def test_pre_roll_buffer_maintenance(dictation_listener, silence_chunk):
    dictation_listener.setup_subscriptions()

    for i in range(10):
        event = create_audio_event(silence_chunk, timestamp=float(i))
        await dictation_listener._handle_audio_chunk(event)

    async with dictation_listener._state_lock:
        assert len(dictation_listener._pre_roll_buffer) == dictation_listener.pre_roll_chunks


@pytest.mark.asyncio
async def test_speech_onset_detection(dictation_listener, speech_chunk):
    dictation_listener.setup_subscriptions()

    event = create_audio_event(speech_chunk)
    await dictation_listener._handle_audio_chunk(event)

    async with dictation_listener._state_lock:
        assert dictation_listener._is_recording
        assert len(dictation_listener._audio_buffer) > 0


@pytest.mark.asyncio
async def test_pre_roll_included_in_recording(dictation_listener, silence_chunk, speech_chunk):
    dictation_listener.setup_subscriptions()

    for _ in range(dictation_listener.pre_roll_chunks):
        event = create_audio_event(silence_chunk)
        await dictation_listener._handle_audio_chunk(event)

    event = create_audio_event(speech_chunk)
    await dictation_listener._handle_audio_chunk(event)

    async with dictation_listener._state_lock:
        assert len(dictation_listener._audio_buffer) >= dictation_listener.pre_roll_chunks + 1


@pytest.mark.asyncio
async def test_silence_detection_ends_recording(dictation_listener, speech_chunk, silence_chunk, event_bus):
    dictation_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(DictationAudioSegmentReadyEvent, capture_segment)

    event = create_audio_event(speech_chunk)
    await dictation_listener._handle_audio_chunk(event)

    for _ in range(dictation_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await dictation_listener._handle_audio_chunk(event)

    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    assert isinstance(captured_events[0], DictationAudioSegmentReadyEvent)

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_segment_ready_event_emission(dictation_listener, speech_chunk, silence_chunk, event_bus):
    dictation_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(DictationAudioSegmentReadyEvent, capture_segment)

    for _ in range(10):
        event = create_audio_event(speech_chunk)
        await dictation_listener._handle_audio_chunk(event)

    for _ in range(dictation_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await dictation_listener._handle_audio_chunk(event)

    await asyncio.sleep(0.1)

    assert len(captured_events) == 1
    segment_event = captured_events[0]
    assert isinstance(segment_event.audio_bytes, bytes)
    assert segment_event.sample_rate == 16000
    assert len(segment_event.audio_bytes) > 0

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_minimum_duration_filtering(dictation_listener, speech_chunk, silence_chunk, event_bus):
    dictation_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(DictationAudioSegmentReadyEvent, capture_segment)

    event = create_audio_event(speech_chunk)
    await dictation_listener._handle_audio_chunk(event)

    for _ in range(dictation_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await dictation_listener._handle_audio_chunk(event)

    await asyncio.sleep(0.1)

    async with dictation_listener._state_lock:
        total_chunks = dictation_listener.pre_roll_chunks + 1
        if total_chunks >= dictation_listener.min_duration_chunks:
            assert len(captured_events) == 1
        else:
            assert len(captured_events) == 0

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_maximum_duration_cutoff(dictation_listener, speech_chunk, event_bus):
    dictation_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(DictationAudioSegmentReadyEvent, capture_segment)

    for _ in range(dictation_listener.max_duration_chunks + 5):
        event = create_audio_event(speech_chunk)
        await dictation_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.001)

    await asyncio.sleep(0.1)

    assert len(captured_events) == 1

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_state_reset_after_segment(dictation_listener, speech_chunk, silence_chunk, event_bus):
    dictation_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(DictationAudioSegmentReadyEvent, capture_segment)

    for _ in range(5):
        event = create_audio_event(speech_chunk)
        await dictation_listener._handle_audio_chunk(event)

    for _ in range(dictation_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await dictation_listener._handle_audio_chunk(event)

    await asyncio.sleep(0.1)

    async with dictation_listener._state_lock:
        assert not dictation_listener._is_recording
        assert len(dictation_listener._audio_buffer) == 0
        assert dictation_listener._consecutive_silent_chunks == 0

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_update_silent_chunks_threshold(dictation_listener):
    initial_threshold = dictation_listener.silent_chunks_for_end
    new_threshold = 20

    await dictation_listener.update_silent_chunks_threshold(new_threshold)

    assert dictation_listener.silent_chunks_for_end == new_threshold
    assert dictation_listener.silent_chunks_for_end != initial_threshold


@pytest.mark.asyncio
async def test_update_silent_chunks_during_recording(dictation_listener, speech_chunk, silence_chunk):
    dictation_listener.setup_subscriptions()

    event = create_audio_event(speech_chunk)
    await dictation_listener._handle_audio_chunk(event)

    await dictation_listener.update_silent_chunks_threshold(5)

    async with dictation_listener._state_lock:
        assert dictation_listener.silent_chunks_for_end == 5


@pytest.mark.asyncio
async def test_consecutive_silent_chunks_reset_on_speech(dictation_listener, speech_chunk, silence_chunk):
    dictation_listener.setup_subscriptions()

    event = create_audio_event(speech_chunk)
    await dictation_listener._handle_audio_chunk(event)

    for _ in range(5):
        event = create_audio_event(silence_chunk)
        await dictation_listener._handle_audio_chunk(event)

    async with dictation_listener._state_lock:
        assert dictation_listener._consecutive_silent_chunks == 5

    event = create_audio_event(speech_chunk)
    await dictation_listener._handle_audio_chunk(event)

    async with dictation_listener._state_lock:
        assert dictation_listener._consecutive_silent_chunks == 0


@pytest.mark.asyncio
async def test_long_dictation_session(dictation_listener, speech_chunk, event_bus):
    dictation_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_segment(event):
        captured_events.append(event)

    event_bus.subscribe(DictationAudioSegmentReadyEvent, capture_segment)

    for _ in range(100):
        event = create_audio_event(speech_chunk)
        await dictation_listener._handle_audio_chunk(event)

    async with dictation_listener._state_lock:
        assert dictation_listener._is_recording
        assert len(dictation_listener._audio_buffer) > 0

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_noise_floor_update(dictation_listener, silence_chunk):
    dictation_listener.setup_subscriptions()

    initial_noise_samples = len(dictation_listener._noise_samples)

    for _ in range(10):
        event = create_audio_event(silence_chunk)
        await dictation_listener._handle_audio_chunk(event)

    async with dictation_listener._state_lock:
        assert len(dictation_listener._noise_samples) > initial_noise_samples


@pytest.mark.asyncio
async def test_concurrent_chunk_processing(dictation_listener, speech_chunk):
    dictation_listener.setup_subscriptions()

    tasks = []
    for i in range(10):
        event = create_audio_event(speech_chunk, timestamp=float(i))
        tasks.append(dictation_listener._handle_audio_chunk(event))

    await asyncio.gather(*tasks)

    async with dictation_listener._state_lock:
        assert dictation_listener._is_recording


@pytest.mark.asyncio
async def test_silence_threshold_calculation(dictation_listener):
    expected_silence = dictation_listener.energy_threshold * dictation_listener.config.vad.silence_threshold_multiplier
    assert dictation_listener.silence_threshold == expected_silence


@pytest.mark.asyncio
async def test_energy_calculation_consistency(dictation_listener):
    chunk1 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    chunk2 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)

    energy1 = dictation_listener._calculate_energy(chunk1)
    energy2 = dictation_listener._calculate_energy(chunk2)

    assert energy1 == energy2
