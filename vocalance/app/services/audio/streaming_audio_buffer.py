"""Streaming audio buffer for real-time dictation with timestamp offset tracking.

Implements WhisperLive-inspired offset system to prevent silence from drowning out
speech. Tracks both physical buffer position and logical transcription position.
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class StreamingAudioBuffer:
    """Timestamp-offset based audio buffer for streaming dictation.

    Uses separate tracking of physical buffer position (frames_offset) and
    logical transcription position (timestamp_offset) to ensure only untranscribed
    audio is sent to Whisper. This prevents silence periods from filling the
    context window and drowning out actual speech.

    Inspired by WhisperLive's offset system for robust streaming transcription.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        _buffer: Numpy array containing accumulated audio samples.
        _buffer_lock: Asyncio lock for thread-safe operations.
        _timestamp_offset: Logical position (in seconds) of last transcribed audio.
        _frames_offset: Physical starting position (in seconds) of current buffer.
        _last_chunk_time: Timestamp of last chunk addition.
    """

    def __init__(self, sample_rate: int = 16000):
        """Initialize streaming audio buffer with offset tracking.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self.sample_rate = sample_rate

        # Physical buffer - holds up to 45 seconds, trims oldest 30s when full
        self._max_buffer_seconds = 45.0
        self._max_buffer_samples = int(self._max_buffer_seconds * sample_rate)
        self._trim_seconds = 30.0

        self._buffer: Optional[np.ndarray] = None
        self._buffer_lock = asyncio.Lock()

        # Offset tracking (in seconds)
        self._timestamp_offset = 0.0  # Logical transcription position
        self._frames_offset = 0.0  # Physical buffer start position

        self._last_chunk_time = time.time()
        self._total_chunks_added = 0

        logger.debug(
            f"StreamingAudioBuffer initialized with offset tracking: "
            f"max_buffer={self._max_buffer_seconds}s, sample_rate={sample_rate}Hz"
        )

    async def add_chunk(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer and trim if necessary.

        Thread-safe method to append new audio data. Automatically trims oldest
        audio when buffer exceeds 45 seconds to prevent memory bloat.

        Args:
            audio_chunk: Numpy array of int16 or float32 audio samples.
        """
        async with self._buffer_lock:
            self._last_chunk_time = time.time()
            self._total_chunks_added += 1

            # Convert to float32 if needed
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0

            # Initialize or append to buffer
            if self._buffer is None:
                self._buffer = audio_chunk.copy()
            else:
                self._buffer = np.concatenate([self._buffer, audio_chunk])

            # Trim oldest audio if buffer exceeds 45 seconds
            if len(self._buffer) > self._max_buffer_samples:
                trim_samples = int(self._trim_seconds * self.sample_rate)
                self._buffer = self._buffer[trim_samples:]
                self._frames_offset += self._trim_seconds

                # CRITICAL: If no speech was transcribed, sync timestamp forward
                # This prevents feeding old silence repeatedly to Whisper
                if self._timestamp_offset < self._frames_offset:
                    logger.debug(
                        f"No transcription progress detected. Advancing timestamp_offset "
                        f"from {self._timestamp_offset:.2f}s to {self._frames_offset:.2f}s"
                    )
                    self._timestamp_offset = self._frames_offset

                logger.debug(
                    f"Trimmed {self._trim_seconds}s from buffer. "
                    f"frames_offset={self._frames_offset:.2f}s, "
                    f"timestamp_offset={self._timestamp_offset:.2f}s"
                )

    async def get_audio_for_transcription(self) -> Optional[tuple[bytes, float]]:
        """Get audio for Whisper processing starting from timestamp_offset.

        Returns audio FROM timestamp_offset onwards (includes recent transcribed audio
        as context for coherent predictions). This matches WhisperLive's approach.

        Returns:
            Tuple of (audio_bytes, duration) or None if no audio available.
            audio_bytes: Audio data as int16 PCM bytes.
            duration: Duration of returned audio in seconds.
        """
        async with self._buffer_lock:
            if self._buffer is None or len(self._buffer) == 0:
                return None

            # Calculate starting position (timestamp_offset is our "read head")
            samples_take = max(0, int((self._timestamp_offset - self._frames_offset) * self.sample_rate))

            # Return audio FROM timestamp_offset onwards (includes context)
            audio_chunk = self._buffer[samples_take:].copy()

            if len(audio_chunk) == 0:
                return None

            duration = len(audio_chunk) / self.sample_rate

            # Convert float32 to int16 for Whisper
            audio_int16 = (audio_chunk * 32768.0).astype(np.int16)

            logger.debug(
                f"Returning audio from timestamp_offset: duration={duration:.2f}s, "
                f"samples_take={samples_take} ({samples_take/self.sample_rate:.2f}s)"
            )

            return audio_int16.tobytes(), duration

    async def advance_timestamp(self, seconds: float) -> None:
        """Advance transcription timestamp after successful recognition.

        Call this after Whisper successfully transcribes audio to mark it as
        "processed" and prevent re-transcription.

        Args:
            seconds: Duration in seconds to advance the timestamp.
        """
        async with self._buffer_lock:
            old_offset = self._timestamp_offset
            self._timestamp_offset += seconds
            logger.debug(f"Advanced timestamp_offset: {old_offset:.2f}s -> {self._timestamp_offset:.2f}s " f"(+{seconds:.2f}s)")

    async def get_timestamp_offset(self) -> float:
        """Get current transcription timestamp offset.

        Returns:
            Current timestamp_offset in seconds.
        """
        async with self._buffer_lock:
            return self._timestamp_offset

    async def get_untranscribed_duration(self) -> float:
        """Get duration of untranscribed audio in buffer.

        Returns:
            Duration in seconds of audio after timestamp_offset.
        """
        async with self._buffer_lock:
            if self._buffer is None:
                return 0.0

            total_duration = len(self._buffer) / self.sample_rate
            current_position = self._frames_offset + total_duration
            untranscribed = current_position - self._timestamp_offset

            return max(0.0, untranscribed)

    async def clear(self) -> None:
        """Clear all audio and reset offsets.

        Thread-safe method to reset buffer state.
        """
        async with self._buffer_lock:
            self._buffer = None
            self._timestamp_offset = 0.0
            self._frames_offset = 0.0
            self._total_chunks_added = 0
            logger.debug("StreamingAudioBuffer cleared and offsets reset")

    def get_last_chunk_time(self) -> float:
        """Get timestamp of last chunk addition (non-async for quick checks).

        Returns:
            Unix timestamp of last add_chunk() call.
        """
        return self._last_chunk_time

    def get_stats(self) -> dict:
        """Get buffer statistics for debugging.

        Returns:
            Dictionary with buffer statistics.
        """
        duration = 0.0
        sample_count = 0

        if self._buffer is not None:
            sample_count = len(self._buffer)
            duration = sample_count / self.sample_rate

        return {
            "buffer_duration_seconds": duration,
            "sample_count": sample_count,
            "max_buffer_samples": self._max_buffer_samples,
            "timestamp_offset": self._timestamp_offset,
            "frames_offset": self._frames_offset,
            "total_chunks_added": self._total_chunks_added,
            "last_chunk_time": self._last_chunk_time,
        }
