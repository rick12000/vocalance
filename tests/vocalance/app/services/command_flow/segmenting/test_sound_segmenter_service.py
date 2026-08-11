import asyncio
from unittest.mock import Mock

import pytest

from vocalance.app.events.core_events import AudioChunkCapturedEvent, ProcessAudioChunkForSoundRecognitionEvent
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.capture.vad import Clip


@pytest.mark.asyncio
async def test_finalized_clip_published_for_sound_recognition(sound_segmenter_service, event_collector):
    received = event_collector(ProcessAudioChunkForSoundRecognitionEvent)
    clip = Clip(pcm_bytes=b"\x05\x06\x07\x08", sample_rate=16000)
    sound_segmenter_service.segmenter.feed_pcm_chunk = Mock(return_value=[clip])

    sound_segmenter_service._handle_audio_chunk(AudioChunkCapturedEvent(pcm_bytes=b"\x00\x00", timestamp=1.0, sample_rate=16000))
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].audio_chunk == clip.pcm_bytes
    assert received[0].sample_rate == clip.sample_rate


@pytest.mark.asyncio
@pytest.mark.parametrize("dictation_active", [True, False])
async def test_dictation_mute_forwarded_to_segmenter(sound_segmenter_service, dictation_active):
    sound_segmenter_service._handle_dictation_mode(
        DictationModeDisableOthersEvent(dictation_mode_active=dictation_active, dictation_mode="standard")
    )
    sound_segmenter_service.segmenter.feed_pcm_chunk = Mock(return_value=[])

    sound_segmenter_service._handle_audio_chunk(AudioChunkCapturedEvent(pcm_bytes=b"\x00\x00", timestamp=1.0, sample_rate=16000))

    assert sound_segmenter_service.muted == dictation_active
    assert sound_segmenter_service.segmenter.feed_pcm_chunk.call_args.args[2] == dictation_active
