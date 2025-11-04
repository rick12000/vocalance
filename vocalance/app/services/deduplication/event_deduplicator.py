"""Unified event deduplication for command events (Vosk, sound, Markov).

Single source of truth for command deduplication across all command sources.
Time-window based with exact match and similarity detection.
Thread-safe for concurrent access from multiple event handlers.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationEntry:
    """Represents a command event in deduplication history."""

    text: str
    source: str  # "vosk", "markov", "sound", etc.
    timestamp: float  # milliseconds


class EventDeduplicator:
    """Unified deduplicator for command events (Vosk, sound, Markov).

    Maintains a time-windowed history of commands and checks new events against it.
    Uses exact text matching within the window, plus similarity checking for longer texts.
    Thread-safe using RLock for concurrent access from multiple event handlers.

    Attributes:
        window_ms: Time window in milliseconds for deduplication checks (default 600ms).
        similarity_threshold: Word overlap ratio threshold for similarity detection (0.0-1.0).
        _history: Deque of recent command events.
        _last_allowed_text: Last command text that was not deduplicated.
        _last_allowed_time: Timestamp of last allowed command (seconds).
        _lock: RLock for thread-safe access to mutable state.
    """

    def __init__(self, window_ms: float = 600, similarity_threshold: float = 0.85) -> None:
        """Initialize deduplicator with configurable time window and similarity threshold.

        Args:
            window_ms: Time window in milliseconds to check for duplicates (default 600ms).
            similarity_threshold: Word overlap ratio threshold for similarity (0.0-1.0, default 0.85).
        """
        self.window_ms = window_ms
        self.similarity_threshold = similarity_threshold
        self._history: deque = deque(maxlen=50)  # Keep recent 50 events
        self._last_allowed_text: Optional[str] = None
        self._last_allowed_time: float = 0.0
        self._lock: threading.RLock = threading.RLock()

        logger.debug(f"EventDeduplicator initialized: window_ms={window_ms}, " f"similarity_threshold={similarity_threshold}")

    def should_deduplicate(self, text: str, source: str, current_time: Optional[float] = None) -> bool:
        """Check if a command event should be deduplicated.

        Thread-safe method using RLock for concurrent access.

        Args:
            text: Command text to check.
            source: Source of command ("vosk", "markov", "sound", etc.).
            current_time: Current timestamp in seconds (auto-generated if None).

        Returns:
            True if event is a duplicate and should be suppressed, False otherwise.
        """
        if not text or not text.strip():
            logger.debug("Empty text - treating as duplicate")
            return True

        if current_time is None:
            current_time = time.time()

        current_time_ms = current_time * 1000
        normalized_text = text.lower().strip()

        with self._lock:
            # Check against last allowed event
            if self._last_allowed_text is not None:
                if (
                    normalized_text == self._last_allowed_text
                    and current_time_ms - (self._last_allowed_time * 1000) < self.window_ms
                ):
                    logger.debug(
                        f"Duplicate detected vs last allowed: '{text}' (source={source}, "
                        f"age_ms={(current_time_ms - (self._last_allowed_time * 1000)):.0f})"
                    )
                    return True

            # Check against recent history within time window
            cutoff_ms = current_time_ms - self.window_ms
            for entry in self._history:
                if entry.timestamp < cutoff_ms:
                    continue

                # Exact match: always a duplicate
                if entry.text.lower().strip() == normalized_text:
                    logger.debug(
                        f"Exact match duplicate detected: '{text}' from {entry.source} "
                        f"(current source={source}, age_ms={(current_time_ms - entry.timestamp):.0f})"
                    )
                    return True

                # Similarity check: for longer commands
                if self._is_similar(normalized_text, entry.text):
                    logger.debug(
                        f"Similar duplicate detected: '{text}' vs '{entry.text}' "
                        f"(source={source}, age_ms={(current_time_ms - entry.timestamp):.0f})"
                    )
                    return True

            return False

    def record_event(self, text: str, source: str, current_time: Optional[float] = None) -> None:
        """Record a command event as allowed (not deduplicated).

        Thread-safe method using RLock for concurrent access.

        Args:
            text: Command text to record.
            source: Source of command.
            current_time: Current timestamp in seconds (auto-generated if None).
        """
        if current_time is None:
            current_time = time.time()

        current_time_ms = current_time * 1000

        with self._lock:
            self._history.append(DeduplicationEntry(text=text, source=source, timestamp=current_time_ms))
            self._last_allowed_text = text.lower().strip()
            self._last_allowed_time = current_time

        logger.debug(f"Event recorded: '{text}' (source={source})")

    def _is_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar based on word overlap.

        Only checks similarity for reasonably long commands (>10 chars each).

        Args:
            text1: First text (normalized).
            text2: Second text (un-normalized).

        Returns:
            True if word overlap ratio exceeds similarity_threshold.
        """
        text2_lower = text2.lower().strip()

        # Only check similarity for longer commands
        if len(text1) < 10 or len(text2_lower) < 10:
            return False

        words1 = set(text1.split())
        words2 = set(text2_lower.split())

        # Need at least 2 words in each to be meaningful
        if len(words1) < 2 or len(words2) < 2:
            return False

        overlap = len(words1.intersection(words2))
        ratio = overlap / min(len(words1), len(words2))

        return ratio >= self.similarity_threshold

    def get_stats(self) -> dict:
        """Return deduplication statistics for monitoring.

        Thread-safe method using RLock for concurrent access.

        Returns:
            Dict with history size, window size, and last event info.
        """
        with self._lock:
            return {
                "history_size": len(self._history),
                "window_ms": self.window_ms,
                "last_allowed_text": self._last_allowed_text,
                "last_allowed_age_ms": (
                    (time.time() * 1000 - self._last_allowed_time * 1000) if self._last_allowed_time else None
                ),
            }
