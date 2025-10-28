import asyncio
import time
from collections import deque
from typing import Optional


class DuplicateTextFilter:
    """Duplicate text filter for STT engines using time-based and similarity detection.

    Prevents duplicate recognition events using exact matching within time window and
    high-similarity detection for longer texts to handle STT variations.

    Thread-safe using asyncio.Lock for use in async event handlers.
    """

    def __init__(self, cache_size: int = 5, duplicate_threshold_ms: float = 300) -> None:
        """Initialize duplicate filter with cache and time window configuration.

        Args:
            cache_size: Maximum number of recent texts to cache.
            duplicate_threshold_ms: Time window in milliseconds for duplicate detection.
        """
        self._text_cache: deque = deque(maxlen=cache_size)
        self._duplicate_threshold_ms: float = duplicate_threshold_ms
        self._last_recognized_text: str = ""
        self._last_text_time: float = 0.0
        self._lock = asyncio.Lock()

    async def is_duplicate(self, text: str, current_time_ms: Optional[float] = None) -> bool:
        """Check if text is duplicate using time-based and similarity detection.

        Args:
            text: Text to check for duplication
            current_time_ms: Current time in milliseconds (auto-generated if None)

        Returns:
            True if text is considered a duplicate
        """
        async with self._lock:
            if not text or not text.strip():
                return True

            normalized_text = text.lower().strip()

            if current_time_ms is None:
                current_time_ms = time.time() * 1000

            current_time_s = current_time_ms / 1000
            if (
                normalized_text == self._last_recognized_text.lower()
                and current_time_s - self._last_text_time < self._duplicate_threshold_ms / 1000
            ):
                return True

            for entry_timestamp, entry_text in reversed(self._text_cache):
                time_diff_ms = current_time_ms - entry_timestamp

                if time_diff_ms > self._duplicate_threshold_ms:
                    continue

                if entry_text.lower().strip() == normalized_text:
                    return True

                if len(normalized_text) > 10 and len(entry_text) > 10:
                    text_words = set(normalized_text.split())
                    entry_words = set(entry_text.lower().split())

                    if len(text_words) >= 3 and len(entry_words) >= 3:
                        overlap = text_words.intersection(entry_words)
                        overlap_ratio = len(overlap) / min(len(text_words), len(entry_words))

                        if overlap_ratio > 0.8 and time_diff_ms < self._duplicate_threshold_ms * 0.5:
                            return True

            self._text_cache.append((current_time_ms, text))
            self._last_recognized_text = text
            self._last_text_time = current_time_s

            return False
