"""VAD listeners for command and sound: PCM chunks from AudioService → segment events on the bus."""

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


class CommandAudioListener:
    """VAD listener for command/stop word detection with low latency."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        shared_audio_processor: Optional[AudioProcessor] = None,
    ) -> None:
        """Initialize command audio listener.

        Args:
            event_bus: EventBus for publishing segment and detection events.
            config: Global application configuration.
            shared_audio_processor: Optional ``AudioProcessor``; if None, a local instance is created.
        """
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

        logger.debug(
            f"CommandAudioListener initialized: "
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 30}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks (~{self.pre_roll_chunks * 30}ms)"
        )

    def set_main_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Loop used to schedule publishes from the VAD worker thread."""
        self._main_loop = loop

    def _schedule_publish(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule ``coro`` on the main asyncio loop from the VAD worker thread."""
        if self._main_loop is None:
            logger.error("CommandAudioListener: main event loop not set; dropping publish")
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
            fut.add_done_callback(self._log_publish_result)
        except RuntimeError as e:
            logger.debug("CommandAudioListener: schedule publish failed: %s", e)

    @staticmethod
    def _log_publish_result(fut: asyncio.Future[Any]) -> None:
        """Log exceptions from a publish future scheduled via ``run_coroutine_threadsafe``."""
        try:
            fut.result()
        except Exception as e:
            logger.error("CommandAudioListener: publish failed: %s", e, exc_info=True)

    def setup_subscriptions(self) -> None:
        """No bus subscriptions; PCM is delivered via ``process_audio_chunk``."""
        logger.debug("CommandAudioListener subscriptions ready")

    def process_audio_chunk(self, audio_chunk: bytes, timestamp: float) -> None:
        """Synchronous VAD entry point from AudioService VAD worker thread."""
        try:
            normalized_chunk, energy = self._audio_processor.process_chunk(audio_chunk)
        except Exception as e:
            logger.error(f"Error preprocessing audio chunk in CommandListener: {e}", exc_info=True)
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
                            audio_detected_event = AudioDetectedEvent(timestamp=timestamp)
                            self._schedule_publish(self.event_bus.publish(audio_detected_event))
                            self._first_speech_in_buffer = False

                        logger.debug(
                            f"Command: Speech detected (energy={energy:.6f}, "
                            f"threshold={self._adaptive_threshold.speech_threshold:.6f})"
                        )

                else:
                    self._audio_buffer.append(normalized_chunk)

                    if self._adaptive_threshold.is_silence(energy):
                        self._consecutive_silent_chunks += 1

                        if self._consecutive_silent_chunks >= self.silent_chunks_for_end:
                            logger.debug(f"Command: Silence detected ({self._consecutive_silent_chunks} chunks)")
                            self._finalize_segment_sync()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    if len(self._audio_buffer) >= self.max_duration_chunks:
                        logger.debug("Command: Max duration reached")
                        self._finalize_segment_sync()

            except Exception as e:
                logger.error(f"Error handling audio chunk in CommandAudioListener: {e}", exc_info=True)

    def _finalize_segment_sync(self) -> None:
        """Finalize current recording and emit CommandAudioSegmentReadyEvent."""
        if not self._audio_buffer:
            self._reset_state()
            return

        if len(self._audio_buffer) < self.min_duration_chunks:
            logger.debug(f"Command segment too short: {len(self._audio_buffer)} chunks < {self.min_duration_chunks} minimum")
            self._reset_state()
            return

        audio_float32 = np.concatenate(self._audio_buffer)
        audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        duration = len(audio_float32) / self.sample_rate

        event = CommandAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=self.sample_rate)
        self._schedule_publish(self.event_bus.publish(event))
        logger.info(f"Command segment ready: {duration:.3f}s, {len(self._audio_buffer)} chunks, {len(audio_bytes)} bytes")

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset buffering state for next segment."""
        self._audio_buffer.clear()
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._first_speech_in_buffer = True

    async def update_silent_chunks_threshold(self, chunks: int) -> None:
        """Update command silent chunks threshold dynamically during runtime.

        Allows real-time adjustment of silence detection sensitivity.
        Thread-safe: Acquires VAD lock for atomic update.

        Args:
            chunks: New number of consecutive silent chunks required to end recording.
        """

        def _apply() -> None:
            with self._vad_lock:
                self.silent_chunks_for_end = chunks
                logger.info(f"Command: Updated silent_chunks_for_end to {chunks} (~{chunks * 30}ms)")

        await asyncio.to_thread(_apply)

    @property
    def energy_threshold(self) -> float:
        """Current adaptive speech-onset threshold."""
        return self._adaptive_threshold.speech_threshold

    @property
    def silence_threshold(self) -> float:
        """Current adaptive silence threshold."""
        return self._adaptive_threshold.silence_threshold


class SoundAudioListener:
    """VAD listener for sound recognition with stricter thresholds."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        shared_audio_processor: Optional[AudioProcessor] = None,
    ) -> None:
        """Initialize sound audio listener.

        Args:
            event_bus: EventBus for publishing sound-segment and dictation gating events.
            config: Global application configuration.
            shared_audio_processor: Optional ``AudioProcessor``; if None, a local instance is created.
        """
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

        logger.debug(
            f"SoundAudioListener initialized: "
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 30}ms), "
            f"min_duration={self.min_duration_chunks} chunks (~{self.min_duration_chunks * 30}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks (~{self.pre_roll_chunks * 30}ms)"
        )

    def set_main_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Loop used to schedule publishes from the VAD worker thread."""
        self._main_loop = loop

    def _schedule_publish(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule ``coro`` on the main asyncio loop from the VAD worker thread."""
        if self._main_loop is None:
            logger.error("SoundAudioListener: main event loop not set; dropping publish")
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
            fut.add_done_callback(self._log_publish_result_sound)
        except RuntimeError as e:
            logger.debug("SoundAudioListener: schedule publish failed: %s", e)

    @staticmethod
    def _log_publish_result_sound(fut: asyncio.Future[Any]) -> None:
        """Log exceptions from a publish future scheduled via ``run_coroutine_threadsafe``."""
        try:
            fut.result()
        except Exception as e:
            logger.error("SoundAudioListener: publish failed: %s", e, exc_info=True)

    def setup_subscriptions(self) -> None:
        """Subscribe to dictation gating; PCM uses ``process_audio_chunk``."""
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)
        logger.debug("SoundAudioListener subscribed to DictationModeDisableOthersEvent")

    def process_audio_chunk(self, audio_chunk: bytes, _timestamp: float) -> None:
        """Synchronous VAD entry point from AudioService VAD worker thread."""
        try:
            normalized_chunk, energy = self._audio_processor.process_chunk(audio_chunk)
        except Exception as e:
            logger.error(f"Error preprocessing audio chunk in SoundListener: {e}", exc_info=True)
            return

        with self._vad_lock:
            try:
                if self._dictation_active:
                    if self._audio_buffer or self._pre_roll_buffer:
                        self._audio_buffer.clear()
                        self._pre_roll_buffer.clear()
                        self._is_recording = False
                        self._consecutive_silent_chunks = 0
                        logger.debug("Sound: Cleared buffer due to dictation mode activation")
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
                        logger.debug(
                            f"Sound: Detected sound onset (energy={energy:.6f}, "
                            f"threshold={self._adaptive_threshold.speech_threshold:.6f}), "
                            f"included {len(self._pre_roll_buffer)} pre-roll chunks"
                        )
                else:
                    self._audio_buffer.append(normalized_chunk)

                    if energy > self._segment_peak_energy:
                        self._segment_peak_energy = energy

                    if self._adaptive_threshold.is_silence(energy):
                        self._consecutive_silent_chunks += 1

                        if self._consecutive_silent_chunks >= self.silent_chunks_for_end:
                            logger.debug(f"Sound: Silence detected ({self._consecutive_silent_chunks} chunks)")
                            self._finalize_segment_sync()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    if len(self._audio_buffer) >= self.max_duration_chunks:
                        logger.debug("Sound: Max duration reached")
                        self._finalize_segment_sync()

            except Exception as e:
                logger.error(f"Error handling audio chunk in SoundAudioListener: {e}", exc_info=True)
                self._reset_state()

    def _finalize_segment_sync(self) -> None:
        """Finalize current recording and emit ProcessAudioChunkForSoundRecognitionEvent."""
        if not self._audio_buffer:
            self._reset_state()
            return

        if len(self._audio_buffer) < self.min_duration_chunks:
            logger.debug(f"Sound segment too short: {len(self._audio_buffer)} chunks < {self.min_duration_chunks} minimum")
            self._reset_state()
            return

        min_peak_ratio = 1.5
        if self._segment_peak_energy < self._adaptive_threshold.speech_threshold * min_peak_ratio:
            logger.debug(
                f"Sound segment rejected: peak energy {self._segment_peak_energy:.6f} "
                f"< {self._adaptive_threshold.speech_threshold * min_peak_ratio:.6f} (threshold * {min_peak_ratio})"
            )
            self._reset_state()
            return

        audio_float32 = np.concatenate(self._audio_buffer)
        audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        duration = len(audio_float32) / self.sample_rate

        sound_event = ProcessAudioChunkForSoundRecognitionEvent(audio_chunk=audio_bytes, sample_rate=self.sample_rate)
        self._schedule_publish(self.event_bus.publish(sound_event))

        logger.info(
            f"Sound segment ready: {duration:.3f}s, {len(self._audio_buffer)} chunks, "
            f"peak_energy={self._segment_peak_energy:.6f}"
        )

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset buffering state for next segment."""
        self._audio_buffer.clear()
        self._pre_roll_buffer.clear()
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._segment_peak_energy = 0.0

    async def _handle_dictation_mode_change(self, event: DictationModeDisableOthersEvent) -> None:
        """Track dictation mode and skip sound recognition during dictation.

        Args:
            event: DictationModeDisableOthersEvent containing mode state.
        """

        def _apply() -> None:
            with self._vad_lock:
                old_state = self._dictation_active
                self._dictation_active = event.dictation_mode_active

                if old_state != self._dictation_active:
                    logger.debug(f"Sound: Dictation mode changed: {old_state} -> {self._dictation_active}")

                    if self._dictation_active:
                        self._reset_state()
                        logger.debug("Sound: Reset state on dictation mode activation")

        await asyncio.to_thread(_apply)

    @property
    def energy_threshold(self) -> float:
        """Current speech detection threshold."""
        return self._adaptive_threshold.speech_threshold

    @property
    def silence_threshold(self) -> float:
        """Current silence detection threshold."""
        return self._adaptive_threshold.silence_threshold
