"""Simple accumulating buffer for streaming audio.

Accumulates ALL audio until explicitly cleared. No automatic trimming.
This is critical for streaming dictation - we must preserve all audio
so Whisper can re-transcribe with full context.
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class StreamingAudioBuffer:
    """Accumulating buffer for streaming dictation.

    Key design: NEVER automatically trim audio. Trimming causes the
    "rolling window" problem where early speech is lost.

    Buffer only clears on explicit clear() call (finalization).
    Coordinator is responsible for calling clear() at appropriate times.

    Attributes:
        sample_rate: Audio sample rate in Hz.
    """

    # Warning threshold - log if buffer gets very large
    WARN_BUFFER_SECONDS = 60.0

    def __init__(self, sample_rate: int = 16000):
        """Initialize buffer.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self.sample_rate = sample_rate
        self._warn_samples = int(self.WARN_BUFFER_SECONDS * sample_rate)

        self._buffer: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()
        self._last_chunk_time = time.time()

        logger.debug("StreamingAudioBuffer initialized (no auto-trim)")

    async def add_chunk(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer. Never trims automatically.

        Args:
            audio_chunk: Numpy array of int16 or float32 audio samples.
        """
        async with self._lock:
            self._last_chunk_time = time.time()

            # Convert to float32 if needed
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0

            # Append to buffer
            if self._buffer is None:
                self._buffer = audio_chunk.copy()
            else:
                self._buffer = np.concatenate([self._buffer, audio_chunk])

            # Warn if buffer is getting very large (but don't trim!)
            if len(self._buffer) > self._warn_samples:
                duration = len(self._buffer) / self.sample_rate
                logger.warning(f"Buffer at {duration:.1f}s - consider finalizing")

    async def get_audio(self) -> Optional[tuple[bytes, float]]:
        """Get all buffered audio for transcription.

        Returns:
            Tuple of (audio_bytes, duration) or None if empty.
        """
        async with self._lock:
            if self._buffer is None or len(self._buffer) == 0:
                return None

            duration = len(self._buffer) / self.sample_rate
            audio_int16 = (self._buffer * 32768.0).astype(np.int16)

            return audio_int16.tobytes(), duration

    async def clear(self) -> None:
        """Clear buffer."""
        async with self._lock:
            self._buffer = None
            self._last_chunk_time = time.time()
            logger.debug("Buffer cleared")

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
            return len(self._buffer) / self.sample_rate

    async def trim_to_duration(self, keep_seconds: float) -> None:
        """Trim buffer to keep only the last N seconds.

        Used after progressive finalization to prevent buffer overflow
        while maintaining some context for Whisper.

        Args:
            keep_seconds: How many seconds to keep from the end.
        """
        async with self._lock:
            if self._buffer is None:
                return

            current_duration = len(self._buffer) / self.sample_rate
            if current_duration <= keep_seconds:
                return

            keep_samples = int(keep_seconds * self.sample_rate)
            trimmed_duration = current_duration - keep_seconds
            self._buffer = self._buffer[-keep_samples:]

            logger.debug(f"Buffer trimmed: kept last {keep_seconds}s, removed {trimmed_duration:.1f}s")
