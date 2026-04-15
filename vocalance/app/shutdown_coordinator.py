"""Application-wide graceful shutdown signal (same layer as ``EventBus``).

``EventBus`` routes domain events; this type coordinates process lifecycle. Both
live under ``vocalance.app`` because they are cross-cutting infrastructure, not
domain services.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PySide6.QtCore import QTimer

logger = logging.getLogger(__name__)


class ShutdownCoordinator:
    """Thread-safe, idempotent shutdown gate tied to the GUI asyncio loop.

    Must be constructed from a coroutine running on ``loop`` (e.g. the first lines of
    ``async def main()`` under QtAsyncio) so the internal ``asyncio.Event`` is bound
    to that loop and matches ``loop`` used for ``call_soon_threadsafe``.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError as e:
            raise RuntimeError("ShutdownCoordinator must be constructed from a coroutine running on the target loop.") from e
        if running is not loop:
            raise ValueError("loop must be the asyncio loop returned by get_running_loop() in this context.")
        self._loop = loop
        self._lock = threading.Lock()
        self._requested = False
        self._event = asyncio.Event()
        self._init_task: Optional[asyncio.Task] = None
        self._signal_poll_timer: Optional["QTimer"] = None

    def attach_signal_poll_timer(self, timer: "QTimer") -> None:
        """Retain the OS-signal poll timer so it is not garbage-collected."""

        self._signal_poll_timer = timer

    def request_shutdown(self, *, reason: str, source: str) -> bool:
        """Signal shutdown. Thread-safe and idempotent.

        Returns:
            True on the first request, False if shutdown was already requested.
        """
        with self._lock:
            if self._requested:
                return False
            self._requested = True

        logger.info("Shutdown requested by %s: %s", source, reason)

        if self._init_task and not self._init_task.done():
            self._init_task.cancel()

        self._loop.call_soon_threadsafe(self._event.set)
        return True

    def is_shutdown_requested(self) -> bool:
        with self._lock:
            return self._requested

    async def wait(self) -> None:
        await self._event.wait()

    def register_initialization_task(self, task: asyncio.Task) -> None:
        self._init_task = task

    def unregister_initialization_task(self) -> None:
        self._init_task = None
