"""
Shared utilities for Speech-to-Text processing.
"""
import time
from collections import deque
from typing import Optional


class DuplicateTextFilter:
    """Shared duplicate text detection for STT engines."""

    def __init__(self, cache_size: int = 5, duplicate_threshold_ms: float = 300):
        self._text_cache: deque = deque(maxlen=cache_size)
        self._duplicate_threshold_ms = duplicate_threshold_ms
        self._last_recognized_text = ""
        self._last_text_time = 0.0

    def is_duplicate(self, text: str, current_time_ms: Optional[float] = None) -> bool:
        """
        Check if text is a duplicate within the time threshold.

        Args:
            text: Text to check
            current_time_ms: Current time in milliseconds (optional, uses time.time() * 1000 if not provided)

        Returns:
            True if text is considered a duplicate
        """
        if not text or not text.strip():
            return True

        normalized_text = text.lower().strip()

        if current_time_ms is None:
            current_time_ms = time.time() * 1000

        # Simple recent duplicate check
        current_time_s = current_time_ms / 1000
        if (
            normalized_text == self._last_recognized_text.lower()
            and current_time_s - self._last_text_time < self._duplicate_threshold_ms / 1000
        ):
            return True

        # Cache-based duplicate check
        for entry_timestamp, entry_text in reversed(self._text_cache):
            time_diff_ms = current_time_ms - entry_timestamp

            # Skip if too old
            if time_diff_ms > self._duplicate_threshold_ms:
                continue

            # Exact match
            if entry_text.lower().strip() == normalized_text:
                return True

            # High similarity check for longer texts
            if len(normalized_text) > 10 and len(entry_text) > 10:
                text_words = set(normalized_text.split())
                entry_words = set(entry_text.lower().split())

                if len(text_words) >= 3 and len(entry_words) >= 3:
                    overlap = text_words.intersection(entry_words)
                    overlap_ratio = len(overlap) / min(len(text_words), len(entry_words))

                    if overlap_ratio > 0.8 and time_diff_ms < self._duplicate_threshold_ms * 0.5:
                        return True

        # Update cache and tracking
        self._text_cache.append((current_time_ms, text))
        self._last_recognized_text = text
        self._last_text_time = current_time_s

        return False
