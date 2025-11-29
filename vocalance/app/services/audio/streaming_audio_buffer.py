"""Streaming audio buffer matching WhisperLive's methodology.

Implements WhisperLive's exact offset management for streaming audio:
- frames_np (buffer): The audio buffer (grows continuously, trims at 45s)
- frames_offset: How much audio has been trimmed from start (advances by 30s when buffer > 45s)
- timestamp_offset: How much audio has been "processed" (advances when segments finalize)

Key principle: NO OVERLAP PARAMETER
WhisperLive sends buffer[timestamp_offset:] to Whisper. The "overlap" comes naturally because:
1. Whisper returns segments with relative timestamps (0-30s)
2. Only COMPLETE segments advance timestamp_offset
3. The LAST segment (incomplete) does NOT advance the offset
4. Next transcription re-processes the incomplete segment's audio
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
    """Streaming audio buffer matching WhisperLive's ServeClientBase exactly.

    WhisperLive's approach (from base.py):
    1. frames_np grows continuously with add_frames()
    2. When buffer > 45s, trim 30s and advance frames_offset by 30
    3. If timestamp_offset < frames_offset after trim, sync them
    4. get_audio_chunk_for_processing() returns buffer[(timestamp_offset - frames_offset):]
    5. timestamp_offset advances ONLY when segments are finalized

    This ensures:
    - Only unprocessed audio is sent to Whisper
    - Incomplete segments get re-transcribed with more audio context
    - No artificial overlap parameter needed

    Attributes:
        sample_rate: Audio sample rate in Hz.
        timestamp_offset: Seconds of audio that have been processed/finalized.
        frames_offset: Seconds of audio that have been trimmed from buffer.
    """

    RATE = 16000

    # Match WhisperLive: Trim when > 45s, trim 30s (keep last 15s)
    MAX_BUFFER_SECONDS = 45.0
    TRIM_SECONDS = 30.0

    def __init__(self, sample_rate: int = 16000):
        """Initialize buffer with dual offset tracking (matches WhisperLive).

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self.sample_rate = sample_rate
        self.RATE = sample_rate

        self._buffer: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()
        self._last_chunk_time = time.time()

        # Dual offset tracking (matches WhisperLive's frames_offset and timestamp_offset)
        self.timestamp_offset: float = 0.0  # How much audio has been processed
        self.frames_offset: float = 0.0  # How much audio has been trimmed

        logger.debug("StreamingAudioBuffer initialized (WhisperLive methodology)")

    async def add_chunk(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer (matches WhisperLive's add_frames).

        WhisperLive's add_frames() logic:
        1. If buffer > 45s, trim 30s and advance frames_offset
        2. If timestamp_offset < frames_offset, sync them (no speech detected)
        3. Append new audio to buffer

        Args:
            audio_chunk: Numpy array of int16 or float32 audio samples.
        """
        async with self._lock:
            self._last_chunk_time = time.time()

            # Convert to float32 if needed (WhisperLive expects float32)
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0

            # WhisperLive: if frames_np.shape[0] > 45*RATE
            if self._buffer is not None and len(self._buffer) > self.MAX_BUFFER_SECONDS * self.RATE:
                # WhisperLive: frames_offset += 30.0
                self.frames_offset += self.TRIM_SECONDS
                # WhisperLive: frames_np = frames_np[int(30*RATE):]
                trim_samples = int(self.TRIM_SECONDS * self.RATE)
                self._buffer = self._buffer[trim_samples:]

                # WhisperLive: if timestamp_offset < frames_offset: timestamp_offset = frames_offset
                # "this basically means that there is no speech as timestamp offset hasnt updated"
                if self.timestamp_offset < self.frames_offset:
                    logger.debug(
                        f"timestamp_offset ({self.timestamp_offset:.2f}s) behind frames_offset "
                        f"({self.frames_offset:.2f}s) - syncing (no speech detected)"
                    )
                    self.timestamp_offset = self.frames_offset

                logger.debug(
                    f"Buffer trimmed: frames_offset={self.frames_offset:.2f}s, " f"timestamp_offset={self.timestamp_offset:.2f}s"
                )

            # WhisperLive: if frames_np is None: frames_np = frame_np.copy()
            # else: frames_np = np.concatenate((frames_np, frame_np), axis=0)
            if self._buffer is None:
                self._buffer = audio_chunk.copy()
            else:
                self._buffer = np.concatenate([self._buffer, audio_chunk])

    async def get_audio_for_transcription(self) -> Optional[tuple[bytes, float]]:
        """Get unprocessed audio for transcription (matches WhisperLive's get_audio_chunk_for_processing).

        WhisperLive's exact logic:
            samples_take = max(0, (timestamp_offset - frames_offset) * RATE)
            input_bytes = frames_np[int(samples_take):].copy()
            duration = input_bytes.shape[0] / RATE
            return input_bytes, duration

        NO OVERLAP PARAMETER - WhisperLive doesn't use one. The natural overlap comes from:
        - Only complete segments advance timestamp_offset
        - Incomplete segments get re-transcribed with more context

        Returns:
            Tuple of (audio_bytes, duration) or None if not enough audio.
        """
        async with self._lock:
            if self._buffer is None:
                return None

            # WhisperLive: samples_take = max(0, (timestamp_offset - frames_offset) * RATE)
            samples_take = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
            # WhisperLive: input_bytes = frames_np[int(samples_take):].copy()
            input_bytes = self._buffer[samples_take:].copy()
            # WhisperLive: duration = input_bytes.shape[0] / RATE
            duration = len(input_bytes) / self.RATE

            if len(input_bytes) == 0:
                return None

            # Convert to int16 bytes for our STT service
            audio_int16 = (input_bytes * 32768.0).astype(np.int16)

            logger.debug(
                f"get_audio_for_transcription: samples_take={samples_take}, "
                f"duration={duration:.2f}s, timestamp_offset={self.timestamp_offset:.2f}s"
            )

            return audio_int16.tobytes(), duration

    async def advance_timestamp_offset(self, offset_advance: float) -> None:
        """Advance timestamp_offset after finalizing segments.

        WhisperLive advances timestamp_offset by segment.end time (relative to chunk start),
        NOT by the full duration. This is called after segments are finalized.

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

    async def clip_audio_if_no_valid_segment(self) -> None:
        """Clip audio if no valid segment for too long (matches WhisperLive's clip_audio_if_no_valid_segment).

        WhisperLive's logic:
            if frames_np[(timestamp_offset - frames_offset)*RATE:].shape[0] > 25 * RATE:
                duration = frames_np.shape[0] / RATE
                timestamp_offset = frames_offset + duration - 5

        This clips to keep only the last 5 seconds when unprocessed audio exceeds 25s.
        """
        async with self._lock:
            if self._buffer is None:
                return

            # WhisperLive: frames_np[int((timestamp_offset - frames_offset)*RATE):].shape[0] > 25 * RATE
            samples_to_skip = int((self.timestamp_offset - self.frames_offset) * self.RATE)
            unprocessed_samples = len(self._buffer) - samples_to_skip

            if unprocessed_samples > 25 * self.RATE:
                # WhisperLive: duration = frames_np.shape[0] / RATE
                total_duration = len(self._buffer) / self.RATE
                # WhisperLive: timestamp_offset = frames_offset + duration - 5
                new_timestamp_offset = self.frames_offset + total_duration - 5
                advance_amount = new_timestamp_offset - self.timestamp_offset
                self.timestamp_offset = new_timestamp_offset
                logger.warning(
                    f"Clipped {advance_amount:.2f}s of audio (no valid segment), "
                    f"new timestamp_offset: {self.timestamp_offset:.2f}s"
                )
