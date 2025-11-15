import asyncio
import logging

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

logger = logging.getLogger(__name__)


class CommandAudioListener:
    """Listens to AudioChunkEvents and accumulates them for command recognition.

    Applies VAD logic with command-specific parameters (low latency, short silence
    timeout) to detect speech segments and emit CommandAudioSegmentReadyEvent.

    This listener operates independently from dictation, enabling simultaneous
    segment detection with different timeouts from the same audio stream.

    Thread-safe: All state access protected by asyncio.Lock for event handler concurrency.

    Parameters expressed as multiples of 50ms base unit:
    - Command silence timeout: 3 chunks (150ms) - for responsive stop word detection
    - Pre-roll buffer: 5 chunks (250ms) - captures word attack
    - Min duration: 1 chunk (50ms)
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig):
        """Initialize command audio listener.

        Args:
            event_bus: EventBus for subscribing to AudioChunkEvent and publishing
                      CommandAudioSegmentReadyEvent.
            config: Global application configuration.
        """
        self.event_bus = event_bus
        self.config = config
        self.sample_rate = config.audio.sample_rate

        # VAD parameters (expressed as multiples of 50ms base unit)
        # Command mode: responsive, short timeouts
        self.energy_threshold = config.vad.command_energy_threshold
        self.silence_threshold = self.energy_threshold * config.vad.silence_threshold_multiplier
        self.silent_chunks_for_end = config.vad.command_silent_chunks_for_end
        self.pre_roll_chunks = config.vad.command_pre_roll_buffers
        self.min_duration_chunks = int(config.vad.command_min_recording_duration / 0.05)  # Convert to chunks
        self.max_duration_chunks = int(config.vad.command_max_recording_duration / 0.05)

        # Adaptive noise floor
        self.adaptive_margin_multiplier = config.vad.command_adaptive_margin_multiplier
        self._noise_floor = config.vad.noise_floor_initial_value
        self._noise_samples = []
        self._max_noise_samples = config.vad.max_noise_samples

        # Buffering state (protected by _state_lock)
        self._pre_roll_buffer = []
        self._audio_buffer = []
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._speech_detected_timestamp = None
        self._first_speech_in_buffer = True

        # Async lock for state protection (event handlers are async)
        self._state_lock = asyncio.Lock()

        logger.debug(
            f"CommandAudioListener initialized: "
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 50}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks"
        )

    def setup_subscriptions(self) -> None:
        """Subscribe to AudioChunkEvent for processing."""
        self.event_bus.subscribe(event_type=AudioChunkEvent, handler=self._handle_audio_chunk)
        logger.debug("CommandAudioListener subscribed to AudioChunkEvent")

    async def _handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        """Process incoming audio chunk and apply VAD logic.

        Thread-safe: Minimizes lock hold time by computing energy outside lock.

        Args:
            event: AudioChunkEvent containing 50ms audio chunk.
        """
        try:
            # Convert bytes and calculate energy OUTSIDE lock for better performance
            chunk = np.frombuffer(event.audio_chunk, dtype=np.int16)
            energy = self._calculate_energy(chunk)
        except Exception as e:
            logger.error(f"Error preprocessing audio chunk in CommandListener: {e}", exc_info=True)
            return

        async with self._state_lock:
            try:
                # Update noise floor if still collecting samples
                if energy <= self.energy_threshold and len(self._noise_samples) < self._max_noise_samples:
                    self._update_noise_floor(energy)

                if not self._is_recording:
                    # STATE: Waiting for speech
                    # Maintain pre-roll buffer
                    self._pre_roll_buffer.append(chunk)
                    if len(self._pre_roll_buffer) > self.pre_roll_chunks:
                        self._pre_roll_buffer.pop(0)

                    # Check for speech onset
                    if energy > self.energy_threshold:
                        self._is_recording = True
                        self._speech_detected_timestamp = event.timestamp

                        # Include full pre-roll buffer - Vosk needs onset context
                        self._audio_buffer.extend(self._pre_roll_buffer)
                        self._audio_buffer.append(chunk)
                        self._consecutive_silent_chunks = 0

                        # Emit AudioDetectedEvent for Markov prediction (once per audio segment)
                        if self._first_speech_in_buffer:
                            audio_detected_event = AudioDetectedEvent(timestamp=event.timestamp)
                            await self.event_bus.publish(audio_detected_event)
                            self._first_speech_in_buffer = False

                        logger.debug("Command: Speech detected, started recording")

                else:
                    # STATE: Recording active speech
                    self._audio_buffer.append(chunk)

                    # Check for silence
                    if energy < self.silence_threshold:
                        self._consecutive_silent_chunks += 1

                        # Check if silence timeout reached
                        if self._consecutive_silent_chunks >= self.silent_chunks_for_end:
                            logger.debug(f"Command: Silence detected ({self._consecutive_silent_chunks} chunks)")
                            await self._finalize_segment()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    # Check max duration
                    if len(self._audio_buffer) >= self.max_duration_chunks:
                        logger.debug("Command: Max duration reached")
                        await self._finalize_segment()

            except Exception as e:
                logger.error(f"Error handling audio chunk in CommandAudioListener: {e}", exc_info=True)

    async def _finalize_segment(self) -> None:
        """Finalize current recording and emit CommandAudioSegmentReadyEvent.

        Preserves natural leading and trailing silence for optimal Vosk recognition.
        Leading silence is captured via pre-roll buffer, trailing silence is preserved
        as detected by the silence timeout threshold.
        """
        if not self._audio_buffer:
            self._reset_state()
            return

        # Check minimum duration
        if len(self._audio_buffer) < self.min_duration_chunks:
            logger.debug(f"Command segment too short: {len(self._audio_buffer)} chunks " f"< {self.min_duration_chunks} minimum")
            self._reset_state()
            return

        # Preserve all audio including natural trailing silence
        # No trimming of trailing silence - let Vosk handle natural audio patterns
        trimmed_buffer = self._audio_buffer

        # Concatenate buffer and convert to bytes
        audio_data = np.concatenate(trimmed_buffer)
        audio_bytes = audio_data.tobytes()
        duration = len(audio_data) / self.sample_rate

        # Emit event
        event = CommandAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=self.sample_rate)
        await self.event_bus.publish(event)
        logger.info(f"Command segment ready: {duration:.3f}s, " f"{len(trimmed_buffer)} chunks, {len(audio_bytes)} bytes")

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset buffering state for next segment."""
        self._audio_buffer.clear()
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._speech_detected_timestamp = None
        self._first_speech_in_buffer = True

    def _calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk.

        Args:
            chunk: Numpy array of int16 audio samples.

        Returns:
            RMS energy normalized to [0, 1] range.
        """
        if chunk.dtype == np.int16:
            return np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
        return np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

    def _update_noise_floor(self, energy: float) -> None:
        """Update adaptive noise floor estimation.

        Args:
            energy: RMS energy from a low-energy chunk.
        """
        if len(self._noise_samples) < self._max_noise_samples:
            self._noise_samples.append(energy)

            if len(self._noise_samples) == self._max_noise_samples:
                self._noise_floor = np.percentile(self._noise_samples, self.config.vad.noise_floor_percentile)
                adaptive_threshold = self._noise_floor * self.adaptive_margin_multiplier

                if adaptive_threshold > self.energy_threshold * self.config.vad.adaptive_threshold_max_multiplier:
                    old_threshold = self.energy_threshold
                    self.energy_threshold = adaptive_threshold
                    self.silence_threshold = self.energy_threshold * self.config.vad.adaptive_silence_threshold_multiplier
                    logger.debug(f"Command: Adapted energy threshold: {old_threshold:.6f} -> {self.energy_threshold:.6f}")

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


class DictationAudioListener:
    """Listens to AudioChunkEvents and accumulates them for dictation recognition.

    Applies VAD logic with dictation-specific parameters (longer silence tolerance)
    to detect speech segments and emit DictationAudioSegmentReadyEvent.

    This listener operates independently from command recognition, enabling simultaneous
    segment detection with different timeouts from the same audio stream.

    Thread-safe: All state access protected by asyncio.Lock for event handler concurrency.

    Parameters expressed as multiples of 50ms base unit:
    - Dictation silence timeout: 16 chunks (800ms) - tolerant of pauses
    - Pre-roll buffer: 5 chunks (250ms) - captures word attack
    - Min duration: 2 chunks (100ms)
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig):
        """Initialize dictation audio listener.

        Args:
            event_bus: EventBus for subscribing to AudioChunkEvent and publishing
                      DictationAudioSegmentReadyEvent.
            config: Global application configuration.
        """
        self.event_bus = event_bus
        self.config = config
        self.sample_rate = config.audio.sample_rate

        # VAD parameters (expressed as multiples of 50ms base unit)
        # Dictation mode: longer timeouts, tolerant of pauses
        self.energy_threshold = config.vad.dictation_energy_threshold
        self.silence_threshold = self.energy_threshold * config.vad.silence_threshold_multiplier
        self.silent_chunks_for_end = config.vad.dictation_silent_chunks_for_end
        self.pre_roll_chunks = config.vad.dictation_pre_roll_buffers
        self.min_duration_chunks = int(config.vad.dictation_min_recording_duration / 0.05)
        self.max_duration_chunks = int(config.vad.dictation_max_recording_duration / 0.05)

        # Adaptive noise floor
        self.adaptive_margin_multiplier = config.vad.dictation_adaptive_margin_multiplier
        self._noise_floor = config.vad.noise_floor_initial_value
        self._noise_samples = []
        self._max_noise_samples = config.vad.max_noise_samples

        # Buffering state (protected by _state_lock)
        self._pre_roll_buffer = []
        self._audio_buffer = []
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._speech_detected_timestamp = None

        # Async lock for state protection (event handlers are async)
        self._state_lock = asyncio.Lock()

        logger.debug(
            f"DictationAudioListener initialized: "
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 50}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks"
        )

    def setup_subscriptions(self) -> None:
        """Subscribe to AudioChunkEvent and DictationModeDisableOthersEvent for processing."""
        self.event_bus.subscribe(event_type=AudioChunkEvent, handler=self._handle_audio_chunk)
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)
        logger.debug("DictationAudioListener subscribed to AudioChunkEvent and DictationModeDisableOthersEvent")

    async def _handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        """Process incoming audio chunk and apply VAD logic.

        Thread-safe: Minimizes lock hold time by computing energy outside lock.

        Args:
            event: AudioChunkEvent containing 50ms audio chunk.
        """
        try:
            # Convert bytes and calculate energy OUTSIDE lock for better performance
            chunk = np.frombuffer(event.audio_chunk, dtype=np.int16)
            energy = self._calculate_energy(chunk)
        except Exception as e:
            logger.error(f"Error preprocessing audio chunk in DictationListener: {e}", exc_info=True)
            return

        async with self._state_lock:
            try:
                # Update noise floor if still collecting samples
                if energy <= self.energy_threshold and len(self._noise_samples) < self._max_noise_samples:
                    self._update_noise_floor(energy)

                if not self._is_recording:
                    # STATE: Waiting for speech
                    # Maintain pre-roll buffer
                    self._pre_roll_buffer.append(chunk)
                    if len(self._pre_roll_buffer) > self.pre_roll_chunks:
                        self._pre_roll_buffer.pop(0)

                    # Check for speech onset
                    if energy > self.energy_threshold:
                        self._is_recording = True
                        self._speech_detected_timestamp = event.timestamp
                        self._audio_buffer.extend(self._pre_roll_buffer)
                        self._audio_buffer.append(chunk)
                        self._consecutive_silent_chunks = 0
                        logger.debug("Dictation: Speech detected, started recording")

                else:
                    # STATE: Recording active speech
                    self._audio_buffer.append(chunk)

                    # Check for silence
                    if energy < self.silence_threshold:
                        self._consecutive_silent_chunks += 1

                        # Check if silence timeout reached
                        if self._consecutive_silent_chunks >= self.silent_chunks_for_end:
                            logger.debug(f"Dictation: Silence detected ({self._consecutive_silent_chunks} chunks)")
                            await self._finalize_segment()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    # Check max duration
                    if len(self._audio_buffer) >= self.max_duration_chunks:
                        logger.debug("Dictation: Max duration reached")
                        await self._finalize_segment()

            except Exception as e:
                logger.error(f"Error handling audio chunk in DictationAudioListener: {e}", exc_info=True)

    async def _handle_dictation_mode_change(self, event: DictationModeDisableOthersEvent) -> None:
        """Clear pre-trigger audio when dictation mode activates.

        When dictation is activated, clear all accumulated audio buffers to ensure only
        audio spoken AFTER the activation trigger is transcribed. This prevents pre-trigger
        words from being included in the dictation output.

        Args:
            event: Event containing dictation mode activation state.
        """
        async with self._state_lock:
            if event.dictation_mode_active:
                # Clear all buffers when entering dictation mode
                self._reset_state()
                # Also explicitly clear pre-roll buffer to ensure no stale pre-trigger audio
                self._pre_roll_buffer.clear()
                logger.debug("DictationAudioListener: Cleared buffers on dictation activation - pre-trigger audio discarded")

    async def _finalize_segment(self) -> None:
        """Finalize current recording and emit DictationAudioSegmentReadyEvent."""
        if not self._audio_buffer:
            self._reset_state()
            return

        # Check minimum duration
        if len(self._audio_buffer) < self.min_duration_chunks:
            logger.debug(f"Dictation segment too short: {len(self._audio_buffer)} chunks " f"< {self.min_duration_chunks} minimum")
            self._reset_state()
            return

        # Concatenate buffer and convert to bytes
        audio_data = np.concatenate(self._audio_buffer)
        audio_bytes = audio_data.tobytes()
        duration = len(audio_data) / self.sample_rate

        # Emit event
        event = DictationAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=self.sample_rate)
        await self.event_bus.publish(event)
        logger.info(f"Dictation segment ready: {duration:.3f}s, " f"{len(self._audio_buffer)} chunks, {len(audio_bytes)} bytes")

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset buffering state for next segment."""
        self._audio_buffer.clear()
        self._is_recording = False
        self._consecutive_silent_chunks = 0
        self._speech_detected_timestamp = None

    def _calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk.

        Args:
            chunk: Numpy array of int16 audio samples.

        Returns:
            RMS energy normalized to [0, 1] range.
        """
        if chunk.dtype == np.int16:
            return np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
        return np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

    def _update_noise_floor(self, energy: float) -> None:
        """Update adaptive noise floor estimation.

        Args:
            energy: RMS energy from a low-energy chunk.
        """
        if len(self._noise_samples) < self._max_noise_samples:
            self._noise_samples.append(energy)

            if len(self._noise_samples) == self._max_noise_samples:
                self._noise_floor = np.percentile(self._noise_samples, self.config.vad.noise_floor_percentile)
                adaptive_threshold = self._noise_floor * self.adaptive_margin_multiplier

                if adaptive_threshold > self.energy_threshold * self.config.vad.adaptive_threshold_max_multiplier:
                    old_threshold = self.energy_threshold
                    self.energy_threshold = adaptive_threshold
                    self.silence_threshold = self.energy_threshold * self.config.vad.adaptive_silence_threshold_multiplier
                    logger.debug(f"Dictation: Adapted energy threshold: {old_threshold:.6f} -> {self.energy_threshold:.6f}")

    async def update_silent_chunks_threshold(self, chunks: int) -> None:
        """Update dictation silent chunks threshold dynamically during runtime.

        Allows real-time adjustment of silence detection sensitivity.
        Thread-safe: Acquires state lock for atomic update.

        Args:
            chunks: New number of consecutive silent chunks required to end recording.
        """
        async with self._state_lock:
            self.silent_chunks_for_end = chunks
            logger.info(f"Dictation: Updated silent_chunks_for_end to {chunks} (~{chunks * 50}ms)")


class SoundAudioListener:
    """Listens to AudioChunkEvents and accumulates them for sound recognition.

    Applies VAD logic with sound-specific parameters to detect sound segments
    and emit ProcessAudioChunkForSoundRecognitionEvent. Only processes audio
    segments that contain actual sound energy, preventing continuous processing
    of silence during training and recognition.

    Thread-safe: All state access protected by asyncio.Lock for event handler concurrency.

    Parameters expressed as multiples of 50ms base unit:
    - Sound silence timeout: 2 chunks (100ms) - quick detection for brief sounds
    - Min duration: 2 chunks (100ms) - minimum sound length
    - Max duration: 20 chunks (1000ms) - prevent memory buildup
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig):
        """Initialize sound audio listener with VAD filtering.

        Args:
            event_bus: EventBus for subscribing to AudioChunkEvent and publishing
                      ProcessAudioChunkForSoundRecognitionEvent.
            config: Global application configuration.
        """
        self.event_bus = event_bus
        self.config = config
        self.sample_rate = config.audio.sample_rate

        # VAD parameters (using sound-specific thresholds for optimal sound detection)
        self.energy_threshold = config.vad.sound_energy_threshold
        self.silence_threshold = self.energy_threshold * config.vad.silence_threshold_multiplier
        self.silent_chunks_for_end = 2  # 100ms of silence to end sound segment
        self.min_duration_chunks = 2  # 100ms minimum sound duration
        self.max_duration_chunks = 20  # 1000ms maximum to prevent memory buildup
        self.pre_roll_chunks = 2  # 100ms pre-roll to capture sound attack/onset

        # Adaptive noise floor
        self.adaptive_margin_multiplier = config.vad.sound_adaptive_margin_multiplier
        self._noise_floor = config.vad.noise_floor_initial_value
        self._noise_samples = []
        self._max_noise_samples = config.vad.max_noise_samples

        # Buffering state (protected by _state_lock)
        self._pre_roll_buffer = []  # NEW: Pre-roll buffer for capturing sound onset
        self._audio_buffer = []
        self._is_recording = False
        self._consecutive_silent_chunks = 0

        # Mode awareness to skip during dictation
        self._dictation_active = False

        # Async lock for state protection
        self._state_lock = asyncio.Lock()

        logger.debug(
            f"SoundAudioListener initialized with VAD: "
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 50}ms), "
            f"min_duration={self.min_duration_chunks} chunks (~{self.min_duration_chunks * 50}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks (~{self.pre_roll_chunks * 50}ms)"
        )

    def setup_subscriptions(self) -> None:
        """Subscribe to AudioChunkEvent and DictationModeDisableOthersEvent."""
        self.event_bus.subscribe(event_type=AudioChunkEvent, handler=self._handle_audio_chunk)
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)
        logger.debug("SoundAudioListener subscribed to AudioChunkEvent and DictationModeDisableOthersEvent")

    async def _handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        """Process incoming audio chunk with VAD filtering for sound detection.

        Thread-safe: Minimizes lock hold time by computing energy outside lock.

        Args:
            event: AudioChunkEvent containing 50ms audio chunk.
        """
        try:
            # Convert bytes and calculate energy OUTSIDE lock for better performance
            chunk = np.frombuffer(event.audio_chunk, dtype=np.int16)
            energy = self._calculate_energy(chunk)
        except Exception as e:
            logger.error(f"Error preprocessing audio chunk in SoundListener: {e}", exc_info=True)
            return

        async with self._state_lock:
            try:
                # Skip if dictation is active (don't interfere with dictation processing)
                if self._dictation_active:
                    if self._audio_buffer or self._pre_roll_buffer:
                        self._audio_buffer.clear()
                        self._pre_roll_buffer.clear()
                        self._is_recording = False
                        self._consecutive_silent_chunks = 0
                        logger.debug("Sound: Cleared buffer due to dictation mode activation")
                    return

                # Update noise floor if still collecting samples
                if energy <= self.energy_threshold and len(self._noise_samples) < self._max_noise_samples:
                    self._update_noise_floor(energy)

                if not self._is_recording:
                    # STATE: Waiting for sound
                    # Maintain pre-roll buffer to capture sound onset/attack
                    self._pre_roll_buffer.append(chunk)
                    if len(self._pre_roll_buffer) > self.pre_roll_chunks:
                        self._pre_roll_buffer.pop(0)

                    # Check for sound onset (above energy threshold)
                    if energy > self.energy_threshold:
                        self._is_recording = True
                        # Include pre-roll buffer to capture sound attack
                        self._audio_buffer.extend(self._pre_roll_buffer)
                        self._audio_buffer.append(chunk)
                        self._consecutive_silent_chunks = 0
                        logger.debug(
                            f"Sound: Detected sound onset (energy: {energy:.6f}), included {len(self._pre_roll_buffer)} pre-roll chunks"
                        )
                else:
                    # STATE: Recording active sound
                    self._audio_buffer.append(chunk)

                    # Check for silence
                    if energy < self.silence_threshold:
                        self._consecutive_silent_chunks += 1

                        # Check if silence timeout reached
                        if self._consecutive_silent_chunks >= self.silent_chunks_for_end:
                            logger.debug(f"Sound: Silence detected ({self._consecutive_silent_chunks} chunks)")
                            await self._finalize_segment()
                            return
                    else:
                        self._consecutive_silent_chunks = 0

                    # Check max duration to prevent memory buildup
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

        # Check minimum duration
        if len(self._audio_buffer) < self.min_duration_chunks:
            logger.debug(f"Sound segment too short: {len(self._audio_buffer)} chunks " f"< {self.min_duration_chunks} minimum")
            self._reset_state()
            return

        # Concatenate buffer and convert to bytes
        audio_segment = np.concatenate(self._audio_buffer)
        audio_bytes = audio_segment.tobytes()
        duration = len(audio_segment) / self.sample_rate

        # Emit sound recognition event
        sound_event = ProcessAudioChunkForSoundRecognitionEvent(audio_chunk=audio_bytes, sample_rate=self.sample_rate)
        await self.event_bus.publish(sound_event)

        logger.info(f"Sound segment ready: {duration:.3f}s, " f"{len(self._audio_buffer)} chunks, {len(audio_bytes)} bytes")

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset buffering state for next segment."""
        self._audio_buffer.clear()
        self._pre_roll_buffer.clear()
        self._is_recording = False
        self._consecutive_silent_chunks = 0

    def _calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk.

        Args:
            chunk: Numpy array of int16 audio samples.

        Returns:
            RMS energy normalized to [0, 1] range.
        """
        if chunk.dtype == np.int16:
            return np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
        return np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

    def _update_noise_floor(self, energy: float) -> None:
        """Update adaptive noise floor estimation.

        Args:
            energy: RMS energy from a low-energy chunk.
        """
        if len(self._noise_samples) < self._max_noise_samples:
            self._noise_samples.append(energy)

            if len(self._noise_samples) == self._max_noise_samples:
                self._noise_floor = np.percentile(self._noise_samples, self.config.vad.noise_floor_percentile)
                adaptive_threshold = self._noise_floor * self.adaptive_margin_multiplier

                if adaptive_threshold > self.energy_threshold * self.config.vad.adaptive_threshold_max_multiplier:
                    old_threshold = self.energy_threshold
                    self.energy_threshold = adaptive_threshold
                    self.silence_threshold = self.energy_threshold * self.config.vad.adaptive_silence_threshold_multiplier
                    logger.debug(f"Sound: Adapted energy threshold: {old_threshold:.6f} -> {self.energy_threshold:.6f}")

    async def _handle_dictation_mode_change(self, event: DictationModeDisableOthersEvent) -> None:
        """Track dictation mode to skip sound recognition during dictation.

        Args:
            event: Event containing dictation mode activation state.
        """
        async with self._state_lock:
            old_state = self._dictation_active
            self._dictation_active = event.dictation_mode_active

            if old_state != self._dictation_active:
                logger.debug(f"Sound: Dictation mode changed: {old_state} -> {self._dictation_active}")

                # Clear buffer and reset state when entering dictation mode
                if self._dictation_active:
                    self._reset_state()
                    logger.debug("Sound: Reset state on dictation mode activation")
