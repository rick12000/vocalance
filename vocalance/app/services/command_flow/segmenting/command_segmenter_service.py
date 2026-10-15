from __future__ import annotations

import logging

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import (
    AudioChunkCapturedEvent,
    AudioDetectedEvent,
    CommandAudioSegmentReadyEvent,
    SettingsChangedEvent,
)
from vocalance.app.services.base_service import Service
from vocalance.app.services.capture.vad import (
    AudioProcessor,
    Clip,
    Onset,
    SegmentConfig,
    SegmentHit,
    UtteranceSegmenter,
)

logger = logging.getLogger(__name__)

PICKUP_MARGIN = {"quiet": 1.6, "normal": 2.2, "noisy": 3.5}
PAUSE_CHUNKS = {"short": 5, "medium": 8, "long": 12}
_VOICE_SETTING_PATHS = frozenset({"vad.command_pickup", "vad.command_end_pause"})


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
            enable_normalization=False,
        )
        self.audio_processor.noise_floor_percentile = 50
        self.segmenter = self._build_segmenter()

        self.subscribe(AudioChunkCapturedEvent, self._handle_audio_chunk)
        self.subscribe(SettingsChangedEvent, self._handle_settings_changed)

    def _build_segmenter(self) -> UtteranceSegmenter:
        vad = self.config.vad
        chunk_seconds = float(self.config.audio.capture_chunk_duration_seconds)
        chunks_per_second = 1.0 / chunk_seconds if chunk_seconds > 0 else 1.0 / 0.03
        margin = PICKUP_MARGIN[vad.command_pickup]
        segment_config = SegmentConfig(
            speech_multiplier=margin,
            silence_multiplier=margin * vad.silence_threshold_multiplier,
            min_threshold=vad.command_energy_threshold,
            max_threshold=vad.command_max_threshold,
            silent_chunks_for_end=PAUSE_CHUNKS[vad.command_end_pause],
            pre_roll_chunks=vad.command_pre_roll_buffers,
            min_duration_chunks=int(
                vad.command_min_recording_duration * chunks_per_second
            ),
            max_duration_chunks=int(
                vad.command_max_recording_duration * chunks_per_second
            ),
            emit_onset=True,
        )
        return UtteranceSegmenter(
            segment_config, self.audio_processor, self.config.audio.sample_rate
        )

    def _handle_audio_chunk(self, event: AudioChunkCapturedEvent) -> None:
        for hit in self.segmenter.feed_pcm_chunk(
            event.pcm_bytes, event.timestamp, False
        ):
            self._dispatch_hit(hit)

    def _dispatch_hit(self, hit: SegmentHit) -> None:
        if isinstance(hit, Onset):
            self.event_bus.submit(AudioDetectedEvent(timestamp=hit.ts))
        elif isinstance(hit, Clip):
            self.event_bus.submit(
                CommandAudioSegmentReadyEvent(audio_bytes=hit.pcm_bytes, sample_rate=hit.sample_rate)
            )

    def _handle_settings_changed(self, event: SettingsChangedEvent) -> None:
        if _VOICE_SETTING_PATHS.intersection(event.updated_settings):
            vad = self.config.vad
            margin = PICKUP_MARGIN[vad.command_pickup]
            self.segmenter.set_voice_gate(
                margin,
                margin * vad.silence_threshold_multiplier,
                PAUSE_CHUNKS[vad.command_end_pause],
            )
