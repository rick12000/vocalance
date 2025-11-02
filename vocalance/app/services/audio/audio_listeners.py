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
)

logger = logging.getLogger(__name__)


class CommandAudioListener:
    """Listens to AudioChunkEvents and accumulates them for command recognition.

    Applies VAD logic with command-specific parameters (low latency, short silence
    timeout) to detect speech segments and emit CommandAudioSegmentReadyEvent.

    This listener operates independently from dictation, enabling simultaneous
    segment detection with different timeouts from the same audio stream.

    Thread-safe: All state access protected by asyncio.Lock for event handler concurrency.

    Parameters expressed as multiples of 30ms base unit:
    - Command silence timeout: 6 chunks (180ms)
    - Pre-roll buffer: 4 chunks (120ms)
    - Min duration: 2 chunks (60ms)
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

        # VAD parameters (expressed as multiples of 30ms base unit)
        # Command mode: responsive, short timeouts
        self.energy_threshold = config.vad.command_energy_threshold
        self.silence_threshold = self.energy_threshold * config.vad.silence_threshold_multiplier
        self.silent_chunks_for_end = config.vad.command_silent_chunks_for_end
        self.pre_roll_chunks = config.vad.command_pre_roll_buffers
        self.min_duration_chunks = int(config.vad.command_min_recording_duration / 0.03)  # Convert to chunks
        self.max_duration_chunks = int(config.vad.command_max_recording_duration / 0.03)

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
        self._first_speech_in_session = True

        # Async lock for state protection (event handlers are async)
        self._state_lock = asyncio.Lock()

        logger.debug(
            f"CommandAudioListener initialized: "
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 30}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks"
        )

    def setup_subscriptions(self) -> None:
        """Subscribe to AudioChunkEvent for processing."""
        self.event_bus.subscribe(event_type=AudioChunkEvent, handler=self._handle_audio_chunk)
        logger.debug("CommandAudioListener subscribed to AudioChunkEvent")

    async def _handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        """Process incoming audio chunk and apply VAD logic.

        Thread-safe: Acquires state lock for entire processing to ensure atomic state transitions.

        Args:
            event: AudioChunkEvent containing 30ms audio chunk.
        """
        async with self._state_lock:
            try:
                # Convert bytes back to numpy array
                chunk = np.frombuffer(event.audio_chunk, dtype=np.int16)
                energy = self._calculate_energy(chunk)

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

                        # Emit AudioDetectedEvent for Markov prediction (only first in session)
                        if self._first_speech_in_session:
                            audio_detected_event = AudioDetectedEvent(timestamp=event.timestamp)
                            await self.event_bus.publish(audio_detected_event)
                            self._first_speech_in_session = False

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
        """Finalize current recording and emit CommandAudioSegmentReadyEvent."""
        if not self._audio_buffer:
            self._reset_state()
            return

        # Check minimum duration
        if len(self._audio_buffer) < self.min_duration_chunks:
            logger.debug(f"Command segment too short: {len(self._audio_buffer)} chunks " f"< {self.min_duration_chunks} minimum")
            self._reset_state()
            return

        # Concatenate buffer and convert to bytes
        audio_data = np.concatenate(self._audio_buffer)
        audio_bytes = audio_data.tobytes()
        duration = len(audio_data) / self.sample_rate

        # Emit event
        event = CommandAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=self.sample_rate)
        await self.event_bus.publish(event)
        logger.info(f"Command segment ready: {duration:.3f}s, " f"{len(self._audio_buffer)} chunks, {len(audio_bytes)} bytes")

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
                    logger.debug(f"Command: Adapted energy threshold: {old_threshold:.6f} -> {self.energy_threshold:.6f}")


class DictationAudioListener:
    """Listens to AudioChunkEvents and accumulates them for dictation recognition.

    Applies VAD logic with dictation-specific parameters (longer silence tolerance)
    to detect speech segments and emit DictationAudioSegmentReadyEvent.

    This listener operates independently from command recognition, enabling simultaneous
    segment detection with different timeouts from the same audio stream.

    Thread-safe: All state access protected by asyncio.Lock for event handler concurrency.

    Parameters expressed as multiples of 30ms base unit:
    - Dictation silence timeout: 27 chunks (810ms)
    - Pre-roll buffer: 8 chunks (240ms)
    - Min duration: 3 chunks (90ms)
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

        # VAD parameters (expressed as multiples of 30ms base unit)
        # Dictation mode: longer timeouts, tolerant of pauses
        self.energy_threshold = config.vad.dictation_energy_threshold
        self.silence_threshold = self.energy_threshold * config.vad.silence_threshold_multiplier
        self.silent_chunks_for_end = config.vad.dictation_silent_chunks_for_end
        self.pre_roll_chunks = config.vad.dictation_pre_roll_buffers
        self.min_duration_chunks = int(config.vad.dictation_min_recording_duration / 0.03)
        self.max_duration_chunks = int(config.vad.dictation_max_recording_duration / 0.03)

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
            f"silent_chunks={self.silent_chunks_for_end} (~{self.silent_chunks_for_end * 30}ms), "
            f"pre_roll={self.pre_roll_chunks} chunks"
        )

    def setup_subscriptions(self) -> None:
        """Subscribe to AudioChunkEvent for processing."""
        self.event_bus.subscribe(event_type=AudioChunkEvent, handler=self._handle_audio_chunk)
        logger.debug("DictationAudioListener subscribed to AudioChunkEvent")

    async def _handle_audio_chunk(self, event: AudioChunkEvent) -> None:
        """Process incoming audio chunk and apply VAD logic.

        Thread-safe: Acquires state lock for entire processing to ensure atomic state transitions.

        Args:
            event: AudioChunkEvent containing 30ms audio chunk.
        """
        async with self._state_lock:
            try:
                # Convert bytes back to numpy array
                chunk = np.frombuffer(event.audio_chunk, dtype=np.int16)
                energy = self._calculate_energy(chunk)

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
            logger.info(f"Dictation: Updated silent_chunks_for_end to {chunks} (~{chunks * 30}ms)")
