import asyncio
from unittest.mock import Mock

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
        ({"vad.command_silent_chunks_for_end": 9}, True),
        ({"vad.command_energy_threshold": 0.01}, False),
    ],
)
async def test_silence_tail_applied_only_for_relevant_setting(command_segmenter_service, updated_settings, should_apply):
    command_segmenter_service.segmenter.set_silence_tail = Mock()

    command_segmenter_service._handle_settings_changed(SettingsChangedEvent(updated_settings=updated_settings, all_settings={}))

    if should_apply:
        command_segmenter_service.segmenter.set_silence_tail.assert_called_once_with(
            command_segmenter_service.config.vad.command_silent_chunks_for_end
        )
    else:
        command_segmenter_service.segmenter.set_silence_tail.assert_not_called()
