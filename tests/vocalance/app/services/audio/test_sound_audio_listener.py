import asyncio
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
import pytest_asyncio

from vocalance.app.events.core_events import ProcessAudioChunkForSoundRecognitionEvent
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.simple_audio_service import AudioService

_SAMPLE_RATE = 16000


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
def sound_chunk():
    return np.random.randint(-5000, 5000, size=800, dtype=np.int16)


@pytest.fixture
def silence_chunk():
    return np.random.randint(-10, 10, size=800, dtype=np.int16)


@pytest.mark.asyncio
async def test_sound_onset_detection_triggers_recording(audio_service, sound_chunk):
    feed_pcm(audio_service, sound_chunk)
    segmenter = audio_service.sound_segmenter
    with segmenter.state_lock:
        assert segmenter.capturing


@pytest.mark.asyncio
async def test_sound_segment_creation_and_emission(audio_service, sound_chunk, silence_chunk, event_bus):
    captured = []

    async def append_event(event):
        captured.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, append_event)
    feed_pcm(audio_service, sound_chunk)
    await asyncio.sleep(0.01)
    segment_config = audio_service.sound_segmenter.config
    for _ in range(segment_config.silent_chunks_for_end):
        feed_pcm(audio_service, silence_chunk)
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    assert len(captured) == 1
    first = captured[0]
    assert isinstance(first.audio_chunk, bytes)
    assert first.sample_rate == _SAMPLE_RATE
    assert len(first.audio_chunk) > 0


@pytest.mark.asyncio
async def test_maximum_duration_enforced(audio_service, sound_chunk, event_bus):
    captured = []

    async def append_event(event):
        captured.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, append_event)
    segment_config = audio_service.sound_segmenter.config
    for _ in range(segment_config.max_duration_chunks + 5):
        feed_pcm(audio_service, sound_chunk)
        await asyncio.sleep(0.001)
    await asyncio.sleep(0.1)
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_dictation_mode_disables_and_reenables_sound_processing(audio_service, sound_chunk, silence_chunk, event_bus):
    captured = []

    async def append_event(event):
        captured.append(event)

    event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, append_event)
    audio_service.apply_dictation(DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard"))
    for _ in range(5):
        feed_pcm(audio_service, sound_chunk)
    await asyncio.sleep(0.1)
    assert len(captured) == 0
    assert audio_service.sound_input_muted
    audio_service.apply_dictation(DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive"))
    for _ in range(5):
        feed_pcm(audio_service, sound_chunk)
        await asyncio.sleep(0.01)
    segment_config = audio_service.sound_segmenter.config
    for _ in range(segment_config.silent_chunks_for_end):
        feed_pcm(audio_service, silence_chunk)
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    assert len(captured) == 1
    assert not audio_service.sound_input_muted
