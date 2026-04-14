"""Coordinates graceful application shutdown from any thread.

Uses an ``asyncio.Event`` rather than a raw ``Future`` — cleaner and avoids
manually retrieving the event loop to schedule ``set_result``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class ShutdownCoordinator:
    """Thread-safe, idempotent shutdown signal.

    Any component — UI thread, OS signal handler, async service — calls
    ``request_shutdown()``.  The application waits on ``wait()`` and proceeds
    to cleanup once it resolves.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = False
        self._event = asyncio.Event()
        self._init_task: Optional[asyncio.Task] = None

    def request_shutdown(self, *, reason: str, source: str) -> bool:
        """Signal shutdown.  Thread-safe and idempotent.

        Returns True on the first call, False on duplicates.
        """
        with self._lock:
            if self._requested:
                return False
            self._requested = True

        logger.info("Shutdown requested by %s: %s", source, reason)

        if self._init_task and not self._init_task.done():
            self._init_task.cancel()

        # asyncio.Event.set() is not thread-safe; schedule it on the loop.
        try:
            loop = self._event.get_loop()  # type: ignore[attr-defined]
        except AttributeError:
            loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(self._event.set)
        return True

    def is_shutdown_requested(self) -> bool:
        with self._lock:
            return self._requested

    async def wait(self) -> None:
        """Await until shutdown is requested."""
        await self._event.wait()

    def register_initialization_task(self, task: asyncio.Task) -> None:
        self._init_task = task

    def unregister_initialization_task(self) -> None:
        self._init_task = None
