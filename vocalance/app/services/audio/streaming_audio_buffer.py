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
    """Streaming audio buffer for continuous speech-to-text transcription.

    Manages dual offset tracking for streaming audio:
    1. frames_np grows continuously with add_chunk()
    2. When buffer > 45s, trim 30s and advance frames_offset by 30
    3. If timestamp_offset < frames_offset after trim, sync them
    4. get_audio_for_transcription() returns buffer[(timestamp_offset - frames_offset):]
    5. timestamp_offset advances ONLY when segments are finalized

    This ensures:
    - Only unprocessed audio is sent to transcriber
    - Incomplete segments get re-transcribed with more audio context
    - No artificial overlap parameter needed

    Attributes:
        sample_rate: Audio sample rate in Hz.
        timestamp_offset: Seconds of audio that have been processed/finalized.
        frames_offset: Seconds of audio that have been trimmed from buffer.
    """

    RATE = 16000

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

        self.timestamp_offset: float = 0.0
        self.frames_offset: float = 0.0

        self._forced_finalization_triggered: bool = False
        self._unprocessed_audio_duration: float = 0.0

        logger.debug("StreamingAudioBuffer initialized")

    async def add_chunk(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer.

        Buffer management logic:
        1. If buffer > 45s, trim 30s and advance frames_offset
        2. If timestamp_offset < frames_offset, sync them (no speech detected)
        3. Append new audio to buffer

        Detects if unprocessed audio is about to be lost (forced finalization check).

        Args:
            audio_chunk: Numpy array of int16 or float32 audio samples.
        """
        async with self._lock:
            self._last_chunk_time = time.time()

            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0

            if self._buffer is not None:
                unprocessed_offset_in_buffer = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
                self._unprocessed_audio_duration = (len(self._buffer) - unprocessed_offset_in_buffer) / self.RATE
            else:
                self._unprocessed_audio_duration = 0.0

            if self._buffer is not None and len(self._buffer) > self.MAX_BUFFER_SECONDS * self.RATE:
                if self.timestamp_offset < self.frames_offset + self.TRIM_SECONDS:
                    lost_audio_duration = (self.frames_offset + self.TRIM_SECONDS) - self.timestamp_offset
                    if lost_audio_duration > 0.5:
                        logger.warning(
                            f"Unprocessed audio about to be discarded: {lost_audio_duration:.2f}s "
                            f"(timestamp_offset={self.timestamp_offset:.2f}s, frames_offset={self.frames_offset:.2f}s)"
                        )
                        self._forced_finalization_triggered = True

                self.frames_offset += self.TRIM_SECONDS
                trim_samples = int(self.TRIM_SECONDS * self.RATE)
                self._buffer = self._buffer[trim_samples:]

                if self.timestamp_offset < self.frames_offset:
                    logger.debug(
                        f"timestamp_offset ({self.timestamp_offset:.2f}s) behind frames_offset "
                        f"({self.frames_offset:.2f}s) - syncing (no speech detected)"
                    )
                    self.timestamp_offset = self.frames_offset

                logger.debug(
                    f"Buffer trimmed: frames_offset={self.frames_offset:.2f}s, " f"timestamp_offset={self.timestamp_offset:.2f}s"
                )

            if self._buffer is None:
                self._buffer = audio_chunk.copy()
            else:
                self._buffer = np.concatenate([self._buffer, audio_chunk])

    async def get_audio_for_transcription(self) -> Optional[tuple[bytes, float]]:
        """Get unprocessed audio for transcription.

        Extraction logic:
            samples_take = max(0, (timestamp_offset - frames_offset) * RATE)
            input_bytes = frames_np[int(samples_take):].copy()
            duration = input_bytes.shape[0] / RATE
            return input_bytes, duration

        Natural overlap comes from:
        - Only complete segments advance timestamp_offset
        - Incomplete segments get re-transcribed with more context

        Returns:
            Tuple of (audio_bytes, duration) or None if not enough audio.
        """
        async with self._lock:
            if self._buffer is None:
                return None

            samples_take = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
            input_bytes = self._buffer[samples_take:].copy()
            duration = len(input_bytes) / self.RATE

            if len(input_bytes) == 0:
                return None

            audio_int16 = (input_bytes * 32768.0).astype(np.int16)

            logger.debug(
                f"get_audio_for_transcription: samples_take={samples_take}, "
                f"duration={duration:.2f}s, timestamp_offset={self.timestamp_offset:.2f}s"
            )

            return audio_int16.tobytes(), duration

    async def advance_timestamp_offset(self, offset_advance: float) -> None:
        """Advance timestamp_offset after finalizing segments.

        Called after segments are finalized. Advances by segment.end time (relative to chunk start),
        not by the full duration.

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

    async def get_unprocessed_audio_duration(self) -> float:
        """Get duration of unfinalized audio that could be lost on next trim.

        Returns:
            Duration in seconds of unprocessed audio.
        """
        async with self._lock:
            return self._unprocessed_audio_duration

    async def check_and_clear_forced_finalization_flag(self) -> bool:
        """Check if forced finalization was triggered, and clear the flag.

        Returns:
            True if forced finalization was triggered, False otherwise.
        """
        async with self._lock:
            triggered = self._forced_finalization_triggered
            self._forced_finalization_triggered = False
            return triggered

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

    async def clip_audio_if_no_valid_segment(self) -> None:
        """Clip audio if no valid segment for too long.

        Clips to keep only the last 5 seconds when unprocessed audio exceeds 25s.
        Prevents excessive accumulation when transcriber is not detecting valid segments.
        """
        async with self._lock:
            if self._buffer is None:
                return

            samples_to_skip = int((self.timestamp_offset - self.frames_offset) * self.RATE)
            unprocessed_samples = len(self._buffer) - samples_to_skip

            if unprocessed_samples > 25 * self.RATE:
                total_duration = len(self._buffer) / self.RATE
                new_timestamp_offset = self.frames_offset + total_duration - 5
                advance_amount = new_timestamp_offset - self.timestamp_offset
                self.timestamp_offset = new_timestamp_offset
                logger.warning(
                    f"Clipped {advance_amount:.2f}s of audio (no valid segment), "
                    f"new timestamp_offset: {self.timestamp_offset:.2f}s"
                )
