import asyncio

import numpy as np
import pytest

from vocalance.app.events.core_events import AudioChunkEvent, ProcessAudioChunkForSoundRecognitionEvent
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.audio_listeners import SoundAudioListener


@pytest.fixture
def sound_listener(event_bus, app_config):
    return SoundAudioListener(event_bus, app_config)


@pytest.fixture
def sound_chunk():
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
async def test_sound_onset_detection_triggers_recording(sound_listener, sound_chunk):
    """Test that sound onset detection triggers recording state."""
    sound_listener.setup_subscriptions()

    event = create_audio_event(sound_chunk)
    await sound_listener._handle_audio_chunk(event)

    async with sound_listener._state_lock:
        assert sound_listener._is_recording


@pytest.mark.asyncio
async def test_sound_segment_creation_and_emission(sound_listener, sound_chunk, silence_chunk, event_bus):
    """Test that sound segments are properly created and emitted."""
    sound_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_sound_event(event):
        captured_events.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, capture_sound_event)

    # Trigger sound recording
    event = create_audio_event(sound_chunk)
    await sound_listener._handle_audio_chunk(event)
    await asyncio.sleep(0.01)

    # End with silence
    for _ in range(sound_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await sound_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    # Should have emitted one sound recognition event
    assert len(captured_events) == 1
    sound_event = captured_events[0]
    assert isinstance(sound_event.audio_chunk, bytes)
    assert sound_event.sample_rate == 16000
    assert len(sound_event.audio_chunk) > 0

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_maximum_duration_enforced(sound_listener, sound_chunk, event_bus):
    """Test that sound recordings are cut off at maximum duration."""
    sound_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_sound_event(event):
        captured_events.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, capture_sound_event)

    # Send more chunks than maximum duration allows
    for _ in range(sound_listener.max_duration_chunks + 5):
        event = create_audio_event(sound_chunk)
        await sound_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.001)

    await asyncio.sleep(0.1)

    # Should produce exactly one segment (cut off at max duration)
    assert len(captured_events) == 1

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_dictation_mode_disables_and_reenables_sound_processing(sound_listener, sound_chunk, silence_chunk, event_bus):
    """Test that dictation mode properly disables and re-enables sound processing."""
    sound_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_sound_event(event):
        captured_events.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, capture_sound_event)

    # Enable dictation mode - should disable sound processing
    dictation_event = DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard")
    await sound_listener._handle_dictation_mode_change(dictation_event)

    # Send sound chunks - should not trigger processing
    for _ in range(5):
        event = create_audio_event(sound_chunk)
        await sound_listener._handle_audio_chunk(event)

    await asyncio.sleep(0.1)
    assert len(captured_events) == 0
    assert sound_listener._dictation_active

    # Disable dictation mode - should re-enable sound processing
    dictation_event = DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive")
    await sound_listener._handle_dictation_mode_change(dictation_event)

    # Send sound chunks again - should now trigger processing
    for _ in range(5):
        event = create_audio_event(sound_chunk)
        await sound_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.01)

    for _ in range(sound_listener.silent_chunks_for_end):
        event = create_audio_event(silence_chunk)
        await sound_listener._handle_audio_chunk(event)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    assert len(captured_events) == 1
    assert not sound_listener._dictation_active

    await event_bus.stop_worker()
