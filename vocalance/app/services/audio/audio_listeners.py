"""Command and sound VAD listeners: PCM from AudioService → segment events on the bus."""

import asyncio
import logging
import threading
from typing import Any, Coroutine, Optional

import numpy as np

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import (
    AudioDetectedEvent,
    CommandAudioSegmentReadyEvent,
    ProcessAudioChunkForSoundRecognitionEvent,
)
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.audio_processor import AdaptiveVADThreshold, AudioProcessor

logger = logging.getLogger(__name__)


def _schedule_publish_on_loop(loop: Optional[asyncio.AbstractEventLoop], coro: Coroutine[Any, Any, Any], label: str) -> None:
    if loop is None:
        logger.error("%s: main event loop not set; dropping publish", label)
        return

    def _log_done(fut: asyncio.Future[Any]) -> None:
        try:
            fut.result()
        except Exception as e:
            logger.error("%s: publish failed: %s", label, e, exc_info=True)

    try:
        asyncio.run_coroutine_threadsafe(coro, loop).add_done_callback(_log_done)
    except RuntimeError as e:
        logger.debug("%s: schedule publish failed: %s", label, e)


class CommandAudioListener:
    """VAD for command segments; publishes ``CommandAudioSegmentReadyEvent``."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        shared_audio_processor: Optional[AudioProcessor] = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config
        self.sample_rate = config.audio.sample_rate

        self._audio_processor = shared_audio_processor or AudioProcessor(
            sample_rate=self.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )

        self._adaptive_threshold = AdaptiveVADThreshold(
            speech_multiplier=config.vad.command_adaptive_margin_multiplier,
            silence_multiplier=config.vad.command_adaptive_margin_multiplier * config.vad.silence_threshold_multiplier,
            min_threshold=config.vad.command_energy_threshold,
        )

        self.silent_chunks_for_end = config.vad.command_silent_chunks_for_end
        self.pre_roll_chunks = config.vad.command_pre_roll_buffers
        self.min_duration_chunks = int(config.vad.command_min_recording_duration / 0.03)
        self.max_duration_chunks = int(config.vad.command_max_recording_duration / 0.03)

        self._pre_roll_buffer: list[np.ndarray] = []
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._first_speech_in_buffer = True

        self._vad_lock = threading.Lock()
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    def set_main_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._main_loop = loop

    def setup_subscriptions(self) -> None:
        pass

    def process_audio_chunk(self, audio_chunk: bytes, timestamp: float) -> None:
        try:
            normalized_chunk, energy = self._audio_processor.process_chunk(audio_chunk)
        except Exception as e:
            logger.error("Command listener preprocess error: %s", e, exc_info=True)
            return

        with self._vad_lock:
            try:
                is_likely_speech = energy > self._adaptive_threshold.speech_threshold
                noise_estimate = self._audio_processor.update_noise_floor(energy, is_likely_speech)

                if noise_estimate.is_stable:
                    self._adaptive_threshold.update(noise_estimate.value)

                if not self._is_recording:
                    self._pre_roll_buffer.append(normalized_chunk)
                    if len(self._pre_roll_buffer) > self.pre_roll_chunks:
                        self._pre_roll_buffer.pop(0)

                    if self._adaptive_threshold.is_speech(energy):
                        self._is_recording = True

                        self._audio_buffer.extend(self._pre_roll_buffer)
                        self._audio_buffer.append(normalized_chunk)
                        self._consecutive_silent_chunks = 0

                        if self._first_speech_in_buffer:
                            _schedule_publish_on_loop(
                                self._main_loop,
                                self.event_bus.publish(AudioDetectedEvent(timestamp=timestamp)),
                                "CommandAudioListener",
                            )
                            self._first_speech_in_buffer = False

                else:
                    self._audio_buffer.append(normalized_chunk)

                    if self._adaptive_threshold.is_silence(energy):
                        self._consecutive_silent_chunks += 1

                        if self._consecutive_silent_chunks >= self.silent_chunks_for_end:
                            self._finalize_segment_sync()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    if len(self._audio_buffer) >= self.max_duration_chunks:
                        self._finalize_segment_sync()

            except Exception as e:
                logger.error("Command listener VAD error: %s", e, exc_info=True)

    def _finalize_segment_sync(self) -> None:
        if not self._audio_buffer:
            self._reset_state()
            return

        if len(self._audio_buffer) < self.min_duration_chunks:
            self._reset_state()
            return

        audio_float32 = np.concatenate(self._audio_buffer)
        audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        duration = len(audio_float32) / self.sample_rate

        event = CommandAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=self.sample_rate)
        _schedule_publish_on_loop(self._main_loop, self.event_bus.publish(event), "CommandAudioListener")
        logger.info("Command segment ready: %.3fs, %s chunks, %s bytes", duration, len(self._audio_buffer), len(audio_bytes))

        self._reset_state()

    def _reset_state(self) -> None:
        self._audio_buffer.clear()
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._first_speech_in_buffer = True

    async def update_silent_chunks_threshold(self, chunks: int) -> None:
        def _apply() -> None:
            with self._vad_lock:
                self.silent_chunks_for_end = chunks

        await asyncio.to_thread(_apply)

    @property
    def energy_threshold(self) -> float:
        return self._adaptive_threshold.speech_threshold

    @property
    def silence_threshold(self) -> float:
        return self._adaptive_threshold.silence_threshold


class SoundAudioListener:
    """VAD for sound training/recognition; publishes ``ProcessAudioChunkForSoundRecognitionEvent``."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        shared_audio_processor: Optional[AudioProcessor] = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config
        self.sample_rate = config.audio.sample_rate

        self._audio_processor = shared_audio_processor or AudioProcessor(
            sample_rate=self.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )

        self._adaptive_threshold = AdaptiveVADThreshold(
            speech_multiplier=config.vad.sound_adaptive_margin_multiplier,
            silence_multiplier=config.vad.sound_adaptive_margin_multiplier * config.vad.silence_threshold_multiplier,
            min_threshold=config.vad.sound_energy_threshold,
            max_threshold=0.15,
        )

        self.silent_chunks_for_end = 5
        self.min_duration_chunks = 5
        self.max_duration_chunks = 34
        self.pre_roll_chunks = 5

        self._segment_peak_energy = 0.0

        self._pre_roll_buffer: list[np.ndarray] = []
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording = False
        self._consecutive_silent_chunks = 0

        self._dictation_active = False

        self._vad_lock = threading.Lock()
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    def set_main_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._main_loop = loop

    def setup_subscriptions(self) -> None:
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)

    def process_audio_chunk(self, audio_chunk: bytes, _timestamp: float) -> None:
        try:
            normalized_chunk, energy = self._audio_processor.process_chunk(audio_chunk)
        except Exception as e:
            logger.error("Sound listener preprocess error: %s", e, exc_info=True)
            return

        with self._vad_lock:
            try:
                if self._dictation_active:
                    if self._audio_buffer or self._pre_roll_buffer:
                        self._audio_buffer.clear()
                        self._pre_roll_buffer.clear()
                        self._is_recording = False
                        self._consecutive_silent_chunks = 0
                    return

                is_likely_sound = energy > self._adaptive_threshold.speech_threshold
                noise_estimate = self._audio_processor.update_noise_floor(energy, is_likely_sound)

                if noise_estimate.is_stable:
                    self._adaptive_threshold.update(noise_estimate.value)

                if not self._is_recording:
                    self._pre_roll_buffer.append(normalized_chunk)
                    if len(self._pre_roll_buffer) > self.pre_roll_chunks:
                        self._pre_roll_buffer.pop(0)

                    if self._adaptive_threshold.is_speech(energy):
                        self._is_recording = True
                        self._segment_peak_energy = energy
                        self._audio_buffer.extend(self._pre_roll_buffer)
                        self._audio_buffer.append(normalized_chunk)
                        self._consecutive_silent_chunks = 0
                else:
                    self._audio_buffer.append(normalized_chunk)

                    if energy > self._segment_peak_energy:
                        self._segment_peak_energy = energy

                    if self._adaptive_threshold.is_silence(energy):
                        self._consecutive_silent_chunks += 1

                        if self._consecutive_silent_chunks >= self.silent_chunks_for_end:
                            self._finalize_segment_sync()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    if len(self._audio_buffer) >= self.max_duration_chunks:
                        self._finalize_segment_sync()

            except Exception as e:
                logger.error("Sound listener VAD error: %s", e, exc_info=True)
                self._reset_state()

    def _finalize_segment_sync(self) -> None:
        if not self._audio_buffer:
            self._reset_state()
            return

        if len(self._audio_buffer) < self.min_duration_chunks:
            self._reset_state()
            return

        min_peak_ratio = 1.5
        if self._segment_peak_energy < self._adaptive_threshold.speech_threshold * min_peak_ratio:
            self._reset_state()
            return

        audio_float32 = np.concatenate(self._audio_buffer)
        audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        duration = len(audio_float32) / self.sample_rate

        sound_event = ProcessAudioChunkForSoundRecognitionEvent(audio_chunk=audio_bytes, sample_rate=self.sample_rate)
        _schedule_publish_on_loop(self._main_loop, self.event_bus.publish(sound_event), "SoundAudioListener")

        logger.info(
            "Sound segment ready: %.3fs, %s chunks, peak_energy=%.6f",
            duration,
            len(self._audio_buffer),
            self._segment_peak_energy,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self._audio_buffer.clear()
        self._pre_roll_buffer.clear()
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._segment_peak_energy = 0.0

    async def _handle_dictation_mode_change(self, event: DictationModeDisableOthersEvent) -> None:
        def _apply() -> None:
            with self._vad_lock:
                self._dictation_active = event.dictation_mode_active
                if self._dictation_active:
                    self._reset_state()

        await asyncio.to_thread(_apply)

    @property
    def energy_threshold(self) -> float:
        return self._adaptive_threshold.speech_threshold

    @property
    def silence_threshold(self) -> float:
        return self._adaptive_threshold.silence_threshold
