import asyncio
import logging
from typing import Optional

import numpy as np

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import (
    AudioChunkEvent,
    AudioDetectedEvent,
    CommandAudioSegmentReadyEvent,
    DictationAudioSegmentReadyEvent,
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
            event_bus: EventBus for subscribing to AudioChunkEvent.
            config: Global application configuration.
            shared_audio_processor: Optional shared AudioProcessor. If None, creates local instance.
        """
        self.event_bus = event_bus
        self.config = config
        self.sample_rate = config.audio.sample_rate

        # Use shared processor if provided, otherwise create local instance
        self._audio_processor = shared_audio_processor or AudioProcessor(
            sample_rate=self.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )
        self._owns_processor = shared_audio_processor is None

        # Adaptive thresholds based on noise floor
        # Command mode uses higher multipliers for more sensitive detection
        self._adaptive_threshold = AdaptiveVADThreshold(
            speech_multiplier=config.vad.command_adaptive_margin_multiplier,
            silence_multiplier=config.vad.command_adaptive_margin_multiplier * config.vad.silence_threshold_multiplier,
            min_threshold=config.vad.command_energy_threshold,
        )

        # Timing parameters (30ms chunks)
        self.silent_chunks_for_end = config.vad.command_silent_chunks_for_end
        self.pre_roll_chunks = config.vad.command_pre_roll_buffers
        self.min_duration_chunks = int(config.vad.command_min_recording_duration / 0.03)
        self.max_duration_chunks = int(config.vad.command_max_recording_duration / 0.03)

        # Buffering state
        self._pre_roll_buffer: list[np.ndarray] = []
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._speech_detected_timestamp = None
        self._first_speech_in_buffer = True

        self._state_lock = asyncio.Lock()

        logger.debug(
            f"CommandAudioListener initialized: "
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 30}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks (~{self.pre_roll_chunks * 30}ms)"
        )

    def setup_subscriptions(self) -> None:
        """Subscribe to AudioChunkEvent for processing."""
        self.event_bus.subscribe(event_type=AudioChunkEvent, handler=self._handle_audio_chunk)
        logger.debug("CommandAudioListener subscribed to AudioChunkEvent")

    async def _handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        """Process audio chunk and apply VAD logic.

        Args:
            event: AudioChunkEvent containing audio data and timestamp.
        """
        try:
            normalized_chunk, energy = self._audio_processor.process_chunk(event.audio_chunk)
        except Exception as e:
            logger.error(f"Error preprocessing audio chunk in CommandListener: {e}", exc_info=True)
            return

        async with self._state_lock:
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
                        self._speech_detected_timestamp = event.timestamp

                        self._audio_buffer.extend(self._pre_roll_buffer)
                        self._audio_buffer.append(normalized_chunk)
                        self._consecutive_silent_chunks = 0

                        if self._first_speech_in_buffer:
                            audio_detected_event = AudioDetectedEvent(timestamp=event.timestamp)
                            await self.event_bus.publish(audio_detected_event)
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
                            await self._finalize_segment()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    if len(self._audio_buffer) >= self.max_duration_chunks:
                        logger.debug("Command: Max duration reached")
                        await self._finalize_segment()

            except Exception as e:
                logger.error(f"Error handling audio chunk in CommandAudioListener: {e}", exc_info=True)

    async def _finalize_segment(self) -> None:
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
        await self.event_bus.publish(event)
        logger.info(f"Command segment ready: {duration:.3f}s, {len(self._audio_buffer)} chunks, {len(audio_bytes)} bytes")

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset buffering state for next segment."""
        self._audio_buffer.clear()
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._speech_detected_timestamp = None
        self._first_speech_in_buffer = True

    async def update_silent_chunks_threshold(self, chunks: int) -> None:
        """Update command silent chunks threshold dynamically during runtime.

        Allows real-time adjustment of silence detection sensitivity.
        Thread-safe: Acquires state lock for atomic update.

        Args:
            chunks: New number of consecutive silent chunks required to end recording.
        """
        async with self._state_lock:
            self.silent_chunks_for_end = chunks
            logger.info(f"Command: Updated silent_chunks_for_end to {chunks} (~{chunks * 50}ms)")

    # Legacy methods and properties for backward compatibility
    @property
    def energy_threshold(self) -> float:
        """Current speech detection threshold (for backward compatibility)."""
        return self._adaptive_threshold.speech_threshold

    @property
    def silence_threshold(self) -> float:
        """Current silence detection threshold (for backward compatibility)."""
        return self._adaptive_threshold.silence_threshold

    def _calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk (legacy method for backward compatibility).

        Args:
            chunk: Numpy array of int16 audio samples.

        Returns:
            RMS energy normalized to [0, 1] range.
        """
        if chunk.dtype == np.int16:
            return float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


class DictationAudioListener:
    """VAD listener for dictation with longer silence tolerance."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        shared_audio_processor: Optional[AudioProcessor] = None,
    ) -> None:
        """Initialize dictation audio listener.

        Args:
            event_bus: EventBus for subscribing to AudioChunkEvent.
            config: Global application configuration.
            shared_audio_processor: Optional shared AudioProcessor. If None, creates local instance.
        """
        self.event_bus = event_bus
        self.config = config
        self.sample_rate = config.audio.sample_rate

        self._audio_processor = shared_audio_processor or AudioProcessor(
            sample_rate=self.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )
        self._owns_processor = shared_audio_processor is None

        self._adaptive_threshold = AdaptiveVADThreshold(
            speech_multiplier=config.vad.dictation_adaptive_margin_multiplier,
            silence_multiplier=config.vad.dictation_adaptive_margin_multiplier * config.vad.silence_threshold_multiplier,
            min_threshold=config.vad.dictation_energy_threshold,
        )

        # Timing parameters (30ms chunks)
        self.silent_chunks_for_end = config.vad.dictation_silent_chunks_for_end
        self.pre_roll_chunks = config.vad.dictation_pre_roll_buffers
        self.min_duration_chunks = int(config.vad.dictation_min_recording_duration / 0.03)
        self.max_duration_chunks = int(config.vad.dictation_max_recording_duration / 0.03)

        # Buffering state
        self._pre_roll_buffer: list[np.ndarray] = []
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._speech_detected_timestamp = None

        self._state_lock = asyncio.Lock()

        logger.debug(
            f"DictationAudioListener initialized: "
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 30}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks (~{self.pre_roll_chunks * 30}ms)"
        )

    def setup_subscriptions(self) -> None:
        """Subscribe to AudioChunkEvent and DictationModeDisableOthersEvent for processing."""
        self.event_bus.subscribe(event_type=AudioChunkEvent, handler=self._handle_audio_chunk)
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)
        logger.debug("DictationAudioListener subscribed to AudioChunkEvent and DictationModeDisableOthersEvent")

    async def _handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        """Process incoming audio chunk and apply VAD logic.

        Thread-safe: Minimizes lock hold time by computing energy outside lock.
        Stores normalized float32 audio in buffers for consistent STT input.

        Args:
            event: AudioChunkEvent containing 30ms audio chunk.
        """
        try:
            # Preprocess audio and get energy (outside lock for performance)
            # normalized_chunk is float32 with DC offset removed and peak normalized
            normalized_chunk, energy = self._audio_processor.process_chunk(event.audio_chunk)
        except Exception as e:
            logger.error(f"Error preprocessing audio chunk in DictationListener: {e}", exc_info=True)
            return

        async with self._state_lock:
            try:
                # Determine if this is likely speech for noise floor update
                is_likely_speech = energy > self._adaptive_threshold.speech_threshold

                # Update noise floor continuously (only with silence samples)
                noise_estimate = self._audio_processor.update_noise_floor(energy, is_likely_speech)

                # Update adaptive thresholds based on noise floor
                if noise_estimate.is_stable:
                    self._adaptive_threshold.update(noise_estimate.value)

                if not self._is_recording:
                    # Maintain pre-roll buffer with normalized audio
                    self._pre_roll_buffer.append(normalized_chunk)
                    if len(self._pre_roll_buffer) > self.pre_roll_chunks:
                        self._pre_roll_buffer.pop(0)

                    # Check for speech onset
                    if self._adaptive_threshold.is_speech(energy):
                        self._is_recording = True
                        self._speech_detected_timestamp = event.timestamp
                        self._audio_buffer.extend(self._pre_roll_buffer)
                        self._audio_buffer.append(normalized_chunk)
                        self._consecutive_silent_chunks = 0
                        logger.debug(
                            f"Dictation: Speech detected (energy={energy:.6f}, "
                            f"threshold={self._adaptive_threshold.speech_threshold:.6f})"
                        )

                else:
                    self._audio_buffer.append(normalized_chunk)

                    if self._adaptive_threshold.is_silence(energy):
                        self._consecutive_silent_chunks += 1

                        if self._consecutive_silent_chunks >= self.silent_chunks_for_end:
                            logger.debug(f"Dictation: Silence detected ({self._consecutive_silent_chunks} chunks)")
                            await self._finalize_segment()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    if len(self._audio_buffer) >= self.max_duration_chunks:
                        logger.debug("Dictation: Max duration reached")
                        await self._finalize_segment()

            except Exception as e:
                logger.error(f"Error handling audio chunk in DictationAudioListener: {e}", exc_info=True)

    async def _handle_dictation_mode_change(self, event: DictationModeDisableOthersEvent) -> None:
        """Clear buffers when dictation mode activates.

        Args:
            event: DictationModeDisableOthersEvent containing mode state.
        """
        async with self._state_lock:
            if event.dictation_mode_active:
                self._reset_state()
                self._pre_roll_buffer.clear()
                logger.debug("DictationAudioListener: Cleared buffers on dictation activation")

    async def _finalize_segment(self) -> None:
        """Finalize current recording and emit DictationAudioSegmentReadyEvent."""
        if not self._audio_buffer:
            self._reset_state()
            return

        if len(self._audio_buffer) < self.min_duration_chunks:
            logger.debug(f"Dictation segment too short: {len(self._audio_buffer)} chunks < {self.min_duration_chunks} minimum")
            self._reset_state()
            return

        audio_float32 = np.concatenate(self._audio_buffer)
        audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        duration = len(audio_float32) / self.sample_rate

        event = DictationAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=self.sample_rate)
        await self.event_bus.publish(event)
        logger.info(f"Dictation segment ready: {duration:.3f}s, {len(self._audio_buffer)} chunks, {len(audio_bytes)} bytes")

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset buffering state for next segment."""
        self._audio_buffer.clear()
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._speech_detected_timestamp = None

    async def update_silent_chunks_threshold(self, chunks: int) -> None:
        """Update dictation silent chunks threshold dynamically during runtime.

        Allows real-time adjustment of silence detection sensitivity.
        Thread-safe: Acquires state lock for atomic update.

        Args:
            chunks: New number of consecutive silent chunks required to end recording.
        """
        async with self._state_lock:
            self.silent_chunks_for_end = chunks
            logger.info(f"Dictation: Updated silent_chunks_for_end to {chunks} (~{chunks * 30}ms)")

    async def force_flush_buffer(self) -> None:
        """Force flush the current audio buffer, bypassing minimum duration check.

        Used by hidden dictation mode to capture trailing audio when the user
        says the stop word, ensuring no audio is lost even if it doesn't meet
        normal publishing criteria.

        Thread-safe: Acquires state lock for atomic buffer access.
        Converts normalized float32 audio back to int16 for STT compatibility.
        """
        async with self._state_lock:
            logger.info(
                f"DictationAudioListener: force_flush_buffer called - buffer has {len(self._audio_buffer)} chunks, "
                f"is_recording: {self._is_recording}, consecutive_silent: {self._consecutive_silent_chunks}"
            )

            if not self._audio_buffer:
                logger.info("DictationAudioListener: No buffer to flush - buffer is empty")
                return

            duration_chunks = len(self._audio_buffer)

            logger.info(
                f"DictationAudioListener: Force flushing buffer with {duration_chunks} chunks "
                f"(min normally required: {self.min_duration_chunks})"
            )

            # Concatenate normalized float32 chunks and convert to int16 for STT
            audio_float32 = np.concatenate(self._audio_buffer)
            audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            duration_seconds = len(audio_float32) / self.sample_rate

            event = DictationAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=self.sample_rate)
            await self.event_bus.publish(event)
            logger.info(
                f"Dictation buffer force-flushed: {duration_seconds:.3f}s, "
                f"{len(self._audio_buffer)} chunks, {len(audio_bytes)} bytes"
            )

            self._reset_state()

    @property
    def energy_threshold(self) -> float:
        """Current speech detection threshold."""
        return self._adaptive_threshold.speech_threshold

    @property
    def silence_threshold(self) -> float:
        """Current silence detection threshold."""
        return self._adaptive_threshold.silence_threshold

    def _calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk.

        Args:
            chunk: Numpy array of int16 audio samples.

        Returns:
            RMS energy normalized to [0, 1] range.
        """
        if chunk.dtype == np.int16:
            return float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


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
            event_bus: EventBus for subscribing to AudioChunkEvent.
            config: Global application configuration.
            shared_audio_processor: Optional shared AudioProcessor. If None, creates local instance.
        """
        self.event_bus = event_bus
        self.config = config
        self.sample_rate = config.audio.sample_rate

        # Use shared processor if provided, otherwise create local instance
        self._audio_processor = shared_audio_processor or AudioProcessor(
            sample_rate=self.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )
        self._owns_processor = shared_audio_processor is None

        # Adaptive thresholds based on noise floor
        # Sound recognition uses HIGHER thresholds than command/dictation
        # to reduce false triggers from ambient noise
        self._adaptive_threshold = AdaptiveVADThreshold(
            speech_multiplier=config.vad.sound_adaptive_margin_multiplier,
            silence_multiplier=config.vad.sound_adaptive_margin_multiplier * config.vad.silence_threshold_multiplier,
            min_threshold=config.vad.sound_energy_threshold,
            max_threshold=0.15,  # Higher max to allow for louder environments
        )

        # Timing parameters - stricter than command/dictation to reduce false triggers
        # At 30ms chunks: 5 chunks = 150ms
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

        self._state_lock = asyncio.Lock()

        logger.debug(
            f"SoundAudioListener initialized: "
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 30}ms), "
            f"min_duration={self.min_duration_chunks} chunks (~{self.min_duration_chunks * 30}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks (~{self.pre_roll_chunks * 30}ms)"
        )

    def setup_subscriptions(self) -> None:
        """Subscribe to AudioChunkEvent and DictationModeDisableOthersEvent."""
        self.event_bus.subscribe(event_type=AudioChunkEvent, handler=self._handle_audio_chunk)
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)
        logger.debug("SoundAudioListener subscribed to AudioChunkEvent and DictationModeDisableOthersEvent")

    async def _handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        """Process audio chunk with VAD filtering for sound detection.

        Args:
            event: AudioChunkEvent containing audio data and timestamp.
        """
        try:
            normalized_chunk, energy = self._audio_processor.process_chunk(event.audio_chunk)
        except Exception as e:
            logger.error(f"Error preprocessing audio chunk in SoundListener: {e}", exc_info=True)
            return

        async with self._state_lock:
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
                            await self._finalize_segment()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    if len(self._audio_buffer) >= self.max_duration_chunks:
                        logger.debug("Sound: Max duration reached")
                        await self._finalize_segment()

            except Exception as e:
                logger.error(f"Error handling audio chunk in SoundAudioListener: {e}", exc_info=True)
                self._reset_state()

    async def _finalize_segment(self) -> None:
        """Finalize current recording and emit ProcessAudioChunkForSoundRecognitionEvent."""
        if not self._audio_buffer:
            self._reset_state()
            return

        if len(self._audio_buffer) < self.min_duration_chunks:
            logger.debug(f"Sound segment too short: {len(self._audio_buffer)} chunks < {self.min_duration_chunks} minimum")
            self._reset_state()
            return

        # Quality check: peak energy must be 1.5x above threshold to avoid spurious triggers
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
        await self.event_bus.publish(sound_event)

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
        async with self._state_lock:
            old_state = self._dictation_active
            self._dictation_active = event.dictation_mode_active

            if old_state != self._dictation_active:
                logger.debug(f"Sound: Dictation mode changed: {old_state} -> {self._dictation_active}")

                if self._dictation_active:
                    self._reset_state()
                    logger.debug("Sound: Reset state on dictation mode activation")

    @property
    def energy_threshold(self) -> float:
        """Current speech detection threshold."""
        return self._adaptive_threshold.speech_threshold

    @property
    def silence_threshold(self) -> float:
        """Current silence detection threshold."""
        return self._adaptive_threshold.silence_threshold

    def _calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk.

        Args:
            chunk: Numpy array of int16 audio samples.

        Returns:
            RMS energy normalized to [0, 1] range.
        """
        if chunk.dtype == np.int16:
            return float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
