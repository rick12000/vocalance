from __future__ import annotations

import asyncio
import logging

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import (
    AudioChunkCapturedEvent,
    AudioDetectedEvent,
    CommandAudioSegmentReadyEvent,
    SettingsChangedEvent,
)
from vocalance.app.services.audio.audio_utils import AudioProcessor, Clip, Onset, SegmentConfig, SegmentHit, UtteranceSegmenter
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class CommandSegmenterService(Service):
    """Segments the captured audio stream into command-length speech clips.

    Subscribes to :class:`AudioChunkCapturedEvent`, runs each chunk through a
    speech-tuned :class:`UtteranceSegmenter`, and publishes
    :class:`AudioDetectedEvent` on each speech onset and
    :class:`CommandAudioSegmentReadyEvent` for each finalized clip.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus)
        self.config = config

        self.audio_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )
        self.segmenter = self._build_segmenter()

        self.subscribe(AudioChunkCapturedEvent, self._handle_audio_chunk)
        self.subscribe(SettingsChangedEvent, self._handle_settings_changed)

    def _build_segmenter(self) -> UtteranceSegmenter:
        vad = self.config.vad
        chunk_seconds = float(self.config.audio.capture_chunk_duration_seconds)
        chunks_per_second = 1.0 / chunk_seconds if chunk_seconds > 0 else 1.0 / 0.03
        segment_config = SegmentConfig(
            speech_multiplier=vad.command_adaptive_margin_multiplier,
            silence_multiplier=vad.command_adaptive_margin_multiplier * vad.silence_threshold_multiplier,
            min_threshold=vad.command_energy_threshold,
            max_threshold=vad.command_max_threshold,
            silent_chunks_for_end=vad.command_silent_chunks_for_end,
            pre_roll_chunks=vad.command_pre_roll_buffers,
            min_duration_chunks=int(vad.command_min_recording_duration * chunks_per_second),
            max_duration_chunks=int(vad.command_max_recording_duration * chunks_per_second),
            emit_onset=True,
        )
        return UtteranceSegmenter(segment_config, self.audio_processor, self.config.audio.sample_rate)

    def _handle_audio_chunk(self, event: AudioChunkCapturedEvent) -> None:
        for hit in self.segmenter.feed_pcm_chunk(event.pcm_bytes, event.timestamp, False):
            self._dispatch_hit(hit)

    def _dispatch_hit(self, hit: SegmentHit) -> None:
        if isinstance(hit, Onset):
            asyncio.create_task(self.event_bus.publish(AudioDetectedEvent(timestamp=hit.ts)))
        elif isinstance(hit, Clip):
            asyncio.create_task(
                self.event_bus.publish(CommandAudioSegmentReadyEvent(audio_bytes=hit.pcm_bytes, sample_rate=hit.sample_rate))
            )

    def _handle_settings_changed(self, event: SettingsChangedEvent) -> None:
        if "vad.command_silent_chunks_for_end" in event.updated_settings:
            self.segmenter.set_silence_tail(self.config.vad.command_silent_chunks_for_end)
