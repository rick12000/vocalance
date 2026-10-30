import asyncio
from unittest.mock import Mock

import numpy as np
import pytest

from vocalance.app.events.core_events import (
    AudioChunkCapturedEvent,
    AudioDetectedEvent,
    CommandAudioSegmentReadyEvent,
    SettingsChangedEvent,
)
from vocalance.app.services.capture.vad import Clip, Onset


@pytest.mark.asyncio
async def test_segmenter_hits_dispatch_to_matching_events(command_segmenter_service, event_collector):
    detected = event_collector(AudioDetectedEvent)
    ready = event_collector(CommandAudioSegmentReadyEvent)
    clip = Clip(pcm_bytes=b"\x01\x02\x03\x04", sample_rate=16000)
    command_segmenter_service.segmenter.feed_pcm_chunk = Mock(return_value=[Onset(ts=12.5), clip])

    command_segmenter_service._handle_audio_chunk(
        AudioChunkCapturedEvent(pcm_bytes=b"\x00\x00", timestamp=12.5, sample_rate=16000)
    )
    await asyncio.sleep(0.05)

    assert len(detected) == 1
    assert detected[0].timestamp == 12.5
    assert len(ready) == 1
    assert ready[0].audio_bytes == clip.pcm_bytes
    assert ready[0].sample_rate == clip.sample_rate


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updated_settings, should_apply",
    [
        ({"vad.command_pickup": "quiet"}, True),
        ({"vad.command_end_pause": "long"}, True),
        ({"vad.command_energy_threshold": 0.01}, False),
    ],
)
async def test_voice_gate_applied_only_for_relevant_setting(command_segmenter_service, updated_settings, should_apply):
    command_segmenter_service.segmenter.set_voice_gate = Mock()

    command_segmenter_service._handle_settings_changed(SettingsChangedEvent(updated_settings=updated_settings, all_settings={}))

    if should_apply:
        command_segmenter_service.segmenter.set_voice_gate.assert_called_once()
    else:
        command_segmenter_service.segmenter.set_voice_gate.assert_not_called()


@pytest.mark.asyncio
async def test_one_loud_chunk_still_closes_into_a_command(command_segmenter_service, silence_chunk_bytes, speech_chunk_bytes):
    sequence = [silence_chunk_bytes] * 2 + [speech_chunk_bytes] + [silence_chunk_bytes] * 6
    hits = []
    for ts, chunk in enumerate(sequence):
        hits.extend(command_segmenter_service.segmenter.feed_pcm_chunk(chunk, float(ts)))

    clips = [hit for hit in hits if isinstance(hit, Clip)]
    assert len(clips) == 1
    assert len(clips[0].pcm_bytes) % 2 == 0


@pytest.mark.asyncio
async def test_quiet_speech_is_captured_without_being_boosted(command_segmenter_service, silence_chunk_bytes):
    quiet = np.tile([250, -250], 400).astype(np.int16).tobytes()
    sequence = [silence_chunk_bytes] * 25 + [quiet] * 3 + [silence_chunk_bytes] * 6
    hits = []
    for ts, chunk in enumerate(sequence):
        hits.extend(command_segmenter_service.segmenter.feed_pcm_chunk(chunk, float(ts)))

    clips = [hit for hit in hits if isinstance(hit, Clip)]
    assert len(clips) == 1
    samples = np.frombuffer(clips[0].pcm_bytes, dtype=np.int16)
    assert len(samples) > 0
    assert int(np.max(np.abs(samples))) < 1000
