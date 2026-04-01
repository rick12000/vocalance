import asyncio
import time

import numpy as np
import pytest
import pytest_asyncio

from vocalance.app.events.core_events import ProcessAudioChunkForSoundRecognitionEvent
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.audio_listeners import SoundAudioListener

_SAMPLE_RATE = 16000


def _feed_chunk(listener: SoundAudioListener, chunk: np.ndarray, timestamp: float | None = None) -> None:
    ts = time.time() if timestamp is None else timestamp
    listener.process_audio_chunk(chunk.tobytes(), ts)


@pytest_asyncio.fixture
async def sound_listener(event_bus, app_config):
    listener = SoundAudioListener(event_bus, app_config)
    listener.set_main_event_loop(asyncio.get_running_loop())
    return listener


@pytest.fixture
def sound_chunk():
    chunk = np.random.randint(-5000, 5000, size=800, dtype=np.int16)
    return chunk


@pytest.fixture
def silence_chunk():
    chunk = np.random.randint(-10, 10, size=800, dtype=np.int16)
    return chunk


@pytest.mark.asyncio
async def test_sound_onset_detection_triggers_recording(sound_listener, sound_chunk):
    sound_listener.setup_subscriptions()

    _feed_chunk(sound_listener, sound_chunk)

    with sound_listener._vad_lock:
        assert sound_listener._is_recording


@pytest.mark.asyncio
async def test_sound_segment_creation_and_emission(sound_listener, sound_chunk, silence_chunk, event_bus):
    sound_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_sound_event(event):
        captured_events.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, capture_sound_event)

    _feed_chunk(sound_listener, sound_chunk)
    await asyncio.sleep(0.01)

    for _ in range(sound_listener.silent_chunks_for_end):
        _feed_chunk(sound_listener, silence_chunk)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    assert len(captured_events) == 1
    sound_event = captured_events[0]
    assert isinstance(sound_event.audio_chunk, bytes)
    assert sound_event.sample_rate == _SAMPLE_RATE
    assert len(sound_event.audio_chunk) > 0

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_maximum_duration_enforced(sound_listener, sound_chunk, event_bus):
    sound_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_sound_event(event):
        captured_events.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, capture_sound_event)

    for _ in range(sound_listener.max_duration_chunks + 5):
        _feed_chunk(sound_listener, sound_chunk)
        await asyncio.sleep(0.001)

    await asyncio.sleep(0.1)

    assert len(captured_events) == 1

    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_dictation_mode_disables_and_reenables_sound_processing(sound_listener, sound_chunk, silence_chunk, event_bus):
    sound_listener.setup_subscriptions()
    await event_bus.start_worker()

    captured_events = []

    async def capture_sound_event(event):
        captured_events.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, capture_sound_event)

    dictation_event = DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard")
    await sound_listener._handle_dictation_mode_change(dictation_event)

    for _ in range(5):
        _feed_chunk(sound_listener, sound_chunk)

    await asyncio.sleep(0.1)
    assert len(captured_events) == 0
    assert sound_listener._dictation_active

    dictation_event = DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive")
    await sound_listener._handle_dictation_mode_change(dictation_event)

    for _ in range(5):
        _feed_chunk(sound_listener, sound_chunk)
        await asyncio.sleep(0.01)

    for _ in range(sound_listener.silent_chunks_for_end):
        _feed_chunk(sound_listener, silence_chunk)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.2)

    assert len(captured_events) == 1
    assert not sound_listener._dictation_active

    await event_bus.stop_worker()
