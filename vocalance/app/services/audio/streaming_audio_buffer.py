"""Streaming audio buffer with dual offset tracking.

Implements offset management for streaming audio:
- frames_offset: How much audio has been trimmed from the buffer
- timestamp_offset: How much audio has been processed/finalized

Audio returned for transcription is: buffer[timestamp_offset - frames_offset:]
This ensures we only transcribe unprocessed audio.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamingSegment:
    """Represents a finalized segment with its audio timestamp.

    Attributes:
        text: The finalized text of this segment.
        start_time: Audio timestamp where this segment starts (seconds from buffer start).
        end_time: Audio timestamp where this segment ends (seconds from buffer start).
    """

    text: str
    start_time: float
    end_time: float


class StreamingAudioBuffer:
    """Streaming audio buffer with dual offset tracking.

    Key design principles:
    1. frames_offset tracks how much audio has been trimmed from buffer start
    2. timestamp_offset tracks how much audio has been processed/finalized
    3. get_audio_for_transcription() returns buffer[timestamp_offset - frames_offset:]
    4. After finalization, timestamp_offset += segment_end_time
    5. Buffer auto-trims when > 45s, advancing frames_offset

    This ensures:
    - Only unprocessed audio is sent to Whisper
    - No re-transcription of already-finalized content
    - Proper context preservation

    Attributes:
        sample_rate: Audio sample rate in Hz.
        timestamp_offset: Seconds of audio that have been processed/finalized.
        frames_offset: Seconds of audio that have been trimmed from buffer.
    """

    RATE = 16000

    # Trim when > 45s, keep last 15s (trim 30s)
    MAX_BUFFER_SECONDS = 45.0
    TRIM_SECONDS = 30.0

    def __init__(self, sample_rate: int = 16000):
        """Initialize buffer with dual offset tracking.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self.sample_rate = sample_rate
        self.RATE = sample_rate

        self._buffer: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()
        self._last_chunk_time = time.time()

        # Dual offset tracking
        self.timestamp_offset: float = 0.0  # How much audio has been processed
        self.frames_offset: float = 0.0  # How much audio has been trimmed

        logger.debug("StreamingAudioBuffer initialized with dual offset tracking")

    async def add_chunk(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer.

        Auto-trims buffer if it exceeds MAX_BUFFER_SECONDS, advancing frames_offset.
        Also syncs timestamp_offset if it falls behind frames_offset.

        Args:
            audio_chunk: Numpy array of int16 or float32 audio samples.
        """
        async with self._lock:
            self._last_chunk_time = time.time()

            # Convert to float32 if needed
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0

            if self._buffer is not None and len(self._buffer) > self.MAX_BUFFER_SECONDS * self.RATE:
                # Trim oldest TRIM_SECONDS of audio
                self.frames_offset += self.TRIM_SECONDS
                trim_samples = int(self.TRIM_SECONDS * self.RATE)
                self._buffer = self._buffer[trim_samples:]

                # Sync timestamp_offset if it fell behind (no speech detected)
                if self.timestamp_offset < self.frames_offset:
                    self.timestamp_offset = self.frames_offset

                logger.debug(
                    f"Buffer auto-trimmed: frames_offset={self.frames_offset:.2f}s, timestamp_offset={self.timestamp_offset:.2f}s"
                )

            # Append to buffer
            if self._buffer is None:
                self._buffer = audio_chunk.copy()
            else:
                self._buffer = np.concatenate([self._buffer, audio_chunk])

    async def get_audio_for_transcription(self) -> Optional[tuple[bytes, float]]:
        """Get unprocessed audio for transcription.

        Returns audio from timestamp_offset onwards, NOT the entire buffer.
        This is the key difference from our previous implementation.

        Returns:
            Tuple of (audio_bytes, duration) or None if not enough audio.
            - audio_bytes: Audio data from timestamp_offset onwards
            - duration: Duration of returned audio in seconds
        """
        async with self._lock:
            if self._buffer is None or len(self._buffer) == 0:
                return None

            # Calculate samples to skip (already processed)
            samples_to_skip = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))

            # Get unprocessed portion
            if samples_to_skip >= len(self._buffer):
                return None

            unprocessed_audio = self._buffer[samples_to_skip:].copy()
            duration = len(unprocessed_audio) / self.RATE

            # Convert to int16 bytes
            audio_int16 = (unprocessed_audio * 32768.0).astype(np.int16)

            return audio_int16.tobytes(), duration

    async def advance_timestamp_offset(self, offset_advance: float) -> None:
        """Advance timestamp_offset after finalizing segments.

        This is called after segments are finalized.

        Args:
            offset_advance: Seconds to advance the timestamp offset.
        """
        async with self._lock:
            self.timestamp_offset += offset_advance
            logger.debug(f"Advanced timestamp_offset by {offset_advance:.2f}s, now: {self.timestamp_offset:.2f}s")

    async def get_audio(self) -> Optional[tuple[bytes, float]]:
        """Get unprocessed audio for transcription (legacy API compatibility).

        Returns:
            Tuple of (audio_bytes, duration) or None if empty.
        """
        return await self.get_audio_for_transcription()

    async def clear(self) -> None:
        """Clear buffer and reset both offsets."""
        async with self._lock:
            self._buffer = None
            self.timestamp_offset = 0.0
            self.frames_offset = 0.0
            self._last_chunk_time = time.time()
            logger.debug("Buffer cleared, offsets reset")

    def get_silence_duration(self) -> float:
        """Get seconds since last audio chunk.

        Returns:
            Seconds since last add_chunk() call.
        """
        return time.time() - self._last_chunk_time

    def get_last_chunk_time(self) -> float:
        """Get timestamp of last chunk.

        Returns:
            Unix timestamp of last add_chunk() call.
        """
        return self._last_chunk_time

    async def get_duration(self) -> float:
        """Get current buffer duration.

        Returns:
            Buffer duration in seconds.
        """
        async with self._lock:
            if self._buffer is None:
                return 0.0
            return len(self._buffer) / self.RATE

    async def get_unprocessed_duration(self) -> float:
        """Get duration of unprocessed audio in buffer.

        Returns:
            Duration of audio that hasn't been processed yet.
        """
        async with self._lock:
            if self._buffer is None:
                return 0.0
            total_duration = len(self._buffer) / self.RATE
            processed_in_buffer = self.timestamp_offset - self.frames_offset
            return max(0, total_duration - processed_in_buffer)

    async def get_timestamp_offset(self) -> float:
        """Get current timestamp offset.

        Returns:
            Seconds of audio that have been processed/finalized.
        """
        async with self._lock:
            return self.timestamp_offset

    async def get_frames_offset(self) -> float:
        """Get current frames offset.

        Returns:
            Seconds of audio that have been trimmed from buffer.
        """
        async with self._lock:
            return self.frames_offset

    async def clip_audio_if_no_valid_segment(self, duration: float) -> None:
        """Clip audio if no valid segment for too long.

        If unprocessed audio is very long (>20s), advance timestamp_offset to skip most of it.
        This prevents losing speech that VAD might have filtered.

        Args:
            duration: Current audio chunk duration.
        """
        async with self._lock:
            if self._buffer is None:
                return

            samples_to_skip = int((self.timestamp_offset - self.frames_offset) * self.RATE)
            unprocessed_samples = len(self._buffer) - samples_to_skip

            # Only clip if > 20s unprocessed
            if unprocessed_samples > 20 * self.RATE:
                total_duration = len(self._buffer) / self.RATE
                new_offset = self.frames_offset + total_duration - 8
                advance_amount = new_offset - self.timestamp_offset
                self.timestamp_offset = new_offset
                logger.warning(
                    f"Clipped {advance_amount:.2f}s of audio due to no valid segment, new timestamp_offset: {self.timestamp_offset:.2f}s"
                )

    async def advance_timestamp_offset_conservative(self, max_advance: float = 0.5) -> float:
        """Conservatively advance timestamp offset for silence handling.

        Unlike advance_timestamp_offset which adds a specific amount,
        this method advances by a small amount to prevent audio loss
        during silence while still making progress.

        Args:
            max_advance: Maximum seconds to advance (default 0.5s).

        Returns:
            Actual amount advanced.
        """
        async with self._lock:
            if self._buffer is None:
                return 0.0

            unprocessed = await self._get_unprocessed_duration_unlocked()

            # Only advance if we have significant unprocessed audio
            # and leave at least 2s of audio for next transcription
            if unprocessed > 3.0:
                advance = min(max_advance, unprocessed - 2.0)
                if advance > 0:
                    self.timestamp_offset += advance
                    logger.debug(f"Conservative advance: {advance:.2f}s, unprocessed remaining: {unprocessed - advance:.2f}s")
                    return advance
            return 0.0

    async def _get_unprocessed_duration_unlocked(self) -> float:
        """Get unprocessed duration without acquiring lock (for internal use)."""
        if self._buffer is None:
            return 0.0
        total_duration = len(self._buffer) / self.RATE
        processed_in_buffer = self.timestamp_offset - self.frames_offset
        return max(0, total_duration - processed_in_buffer)
