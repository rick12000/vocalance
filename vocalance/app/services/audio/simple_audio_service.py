import asyncio

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import (
    AudioDetectedEvent,
    CommandAudioSegmentReadyEvent,
    MicLevelMeterPcmChunkEvent,
    ProcessAudioChunkForSoundRecognitionEvent,
    SettingsChangedEvent,
)
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.audio_utils import AudioProcessor, Clip, Onset, SegmentConfig, SegmentHit, UtteranceSegmenter
from vocalance.app.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator
from vocalance.app.services.audio.recorder import AudioRecorder
from vocalance.app.services.base_service import Service


class AudioService(Service):
    """Captures microphone audio, feeds dictation, meters levels, and publishes segment events."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        main_event_loop: asyncio.AbstractEventLoop,
        dictation: DictationCoordinator,
    ) -> None:
        super().__init__(event_bus)
        self.config = config
        self.main_event_loop = main_event_loop
        self.dictation = dictation

        self.chunk_analyzer = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )
        self.command_segmenter = self.create_command_segmenter()
        self.sound_segmenter = self.create_sound_segmenter()
        self.sound_input_muted = False
        self.subscribe(DictationModeDisableOthersEvent, self.apply_dictation)
        self.subscribe(SettingsChangedEvent, self._handle_settings_changed)

        self.recorder = AudioRecorder(
            app_config=config,
            loop=main_event_loop,
            event_bus=event_bus,
            on_audio_chunk=self.relay_captured_pcm_to_consumers,
        )

    def create_command_segmenter(self) -> UtteranceSegmenter:
        app_config = self.config
        chunk_seconds = float(app_config.audio.capture_chunk_duration_seconds)
        chunks_per_second = 1.0 / chunk_seconds if chunk_seconds > 0 else 1.0 / 0.03
        vad = app_config.vad
        segment_config = SegmentConfig(
            speech_multiplier=vad.command_adaptive_margin_multiplier,
            silence_multiplier=vad.command_adaptive_margin_multiplier * vad.silence_threshold_multiplier,
            min_threshold=vad.command_energy_threshold,
            max_threshold=0.1,
            silent_chunks_for_end=vad.command_silent_chunks_for_end,
            pre_roll_chunks=vad.command_pre_roll_buffers,
            min_duration_chunks=int(vad.command_min_recording_duration * chunks_per_second),
            max_duration_chunks=int(vad.command_max_recording_duration * chunks_per_second),
            emit_onset=True,
        )
        return UtteranceSegmenter(segment_config, self.chunk_analyzer, app_config.audio.sample_rate)

    def create_sound_segmenter(self) -> UtteranceSegmenter:
        app_config = self.config
        vad = app_config.vad
        segment_config = SegmentConfig(
            speech_multiplier=vad.sound_adaptive_margin_multiplier,
            silence_multiplier=vad.sound_adaptive_margin_multiplier * vad.silence_threshold_multiplier,
            min_threshold=vad.sound_energy_threshold,
            max_threshold=0.15,
            silent_chunks_for_end=5,
            pre_roll_chunks=5,
            min_duration_chunks=5,
            max_duration_chunks=34,
            min_peak_ratio=1.5,
        )
        return UtteranceSegmenter(segment_config, self.chunk_analyzer, app_config.audio.sample_rate)

    def apply_dictation(self, event: DictationModeDisableOthersEvent) -> None:
        self.sound_input_muted = event.dictation_mode_active

    async def publish_mic_level_meter_chunk(self, pcm: bytes) -> None:
        await self.event_bus.publish(MicLevelMeterPcmChunkEvent(audio_chunk=pcm))

    async def publish_audio_detected(self, ts: float) -> None:
        await self.event_bus.publish(AudioDetectedEvent(timestamp=ts))

    async def publish_command_segment_ready(self, clip: Clip) -> None:
        await self.event_bus.publish(CommandAudioSegmentReadyEvent(audio_bytes=clip.pcm_bytes, sample_rate=clip.sample_rate))

    async def publish_sound_recognition_chunk(self, clip: Clip) -> None:
        await self.event_bus.publish(
            ProcessAudioChunkForSoundRecognitionEvent(audio_chunk=clip.pcm_bytes, sample_rate=clip.sample_rate)
        )

    def schedule_command_hit(self, hit: SegmentHit) -> None:
        if isinstance(hit, Onset):
            asyncio.create_task(self.publish_audio_detected(hit.ts))
        elif isinstance(hit, Clip):
            asyncio.create_task(self.publish_command_segment_ready(hit))

    def schedule_sound_hit(self, hit: SegmentHit) -> None:
        if isinstance(hit, Clip):
            asyncio.create_task(self.publish_sound_recognition_chunk(hit))

    def relay_captured_pcm_to_consumers(self, pcm_bytes: bytes, ts: float) -> None:
        self.dictation.feed_moonshine_audio_chunk(pcm_bytes, self.config.audio.sample_rate)
        asyncio.create_task(self.publish_mic_level_meter_chunk(pcm_bytes))
        for hit in self.command_segmenter.feed_pcm_chunk(pcm_bytes, ts, False):
            self.schedule_command_hit(hit)
        for hit in self.sound_segmenter.feed_pcm_chunk(pcm_bytes, ts, self.sound_input_muted):
            self.schedule_sound_hit(hit)

    def start_processing(self) -> None:
        self.recorder.start()

    def stop_processing(self) -> None:
        self.recorder.stop()

    async def wait_for_capture_pipeline_idle(self, timeout_s: float = 3.0) -> None:
        if self.recorder is not None:
            await self.recorder.wait_deliveries_drained(timeout_s)

    def _handle_settings_changed(self, event: SettingsChangedEvent) -> None:
        """Re-read VAD silence-tail from the shared config when it changes."""
        if "vad.command_silent_chunks_for_end" in event.updated_settings:
            self.command_segmenter.set_silence_tail(self.config.vad.command_silent_chunks_for_end)

    async def shutdown(self) -> None:
        self.stop_processing()
        if self.recorder is not None:
            await self.recorder.wait_deliveries_drained(timeout_s=3.0)
        self.recorder = None
        self.command_segmenter = None
        self.sound_segmenter = None
        self.chunk_analyzer = None
        await super().shutdown()
