from __future__ import annotations

import asyncio
import logging

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioChunkCapturedEvent, ProcessAudioChunkForSoundRecognitionEvent
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.audio_utils import AudioProcessor, Clip, SegmentConfig, UtteranceSegmenter
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class SoundSegmenterService(Service):
    """Segments the captured audio stream into short, transient sound clips.

    Subscribes to :class:`AudioChunkCapturedEvent`, runs each chunk through a
    sound-tuned :class:`UtteranceSegmenter`, and publishes
    :class:`ProcessAudioChunkForSoundRecognitionEvent` for each finalized
    clip. Self-mutes whenever dictation is active so dictated words do not
    produce false-positive sound clips.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus)
        self.config = config
        self._muted = False

        self.audio_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )
        self.segmenter = self._build_segmenter()

        self.subscribe(AudioChunkCapturedEvent, self._handle_audio_chunk)
        self.subscribe(DictationModeDisableOthersEvent, self._handle_dictation_mode)

    def _build_segmenter(self) -> UtteranceSegmenter:
        vad = self.config.vad
        chunk_seconds = float(self.config.audio.capture_chunk_duration_seconds)
        chunks_per_second = 1.0 / chunk_seconds if chunk_seconds > 0 else 1.0 / 0.03
        segment_config = SegmentConfig(
            speech_multiplier=vad.sound_adaptive_margin_multiplier,
            silence_multiplier=vad.sound_adaptive_margin_multiplier * vad.silence_threshold_multiplier,
            min_threshold=vad.sound_energy_threshold,
            max_threshold=vad.sound_max_threshold,
            silent_chunks_for_end=vad.sound_silent_chunks_for_end,
            pre_roll_chunks=vad.sound_pre_roll_buffers,
            min_duration_chunks=int(vad.sound_min_recording_duration * chunks_per_second),
            max_duration_chunks=int(vad.sound_max_recording_duration * chunks_per_second),
            min_peak_ratio=vad.sound_min_peak_ratio,
        )
        return UtteranceSegmenter(segment_config, self.audio_processor, self.config.audio.sample_rate)

    def _handle_audio_chunk(self, event: AudioChunkCapturedEvent) -> None:
        for hit in self.segmenter.feed_pcm_chunk(event.pcm_bytes, event.timestamp, self._muted):
            if isinstance(hit, Clip):
                asyncio.create_task(
                    self.event_bus.publish(
                        ProcessAudioChunkForSoundRecognitionEvent(audio_chunk=hit.pcm_bytes, sample_rate=hit.sample_rate)
                    )
                )

    def _handle_dictation_mode(self, event: DictationModeDisableOthersEvent) -> None:
        self._muted = event.dictation_mode_active

    @property
    def muted(self) -> bool:
        return self._muted
