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
        self.loop = loop
        self.coord_lock = threading.Lock()
        self.requested = False
        self.shutdown_event = asyncio.Event()
        self.init_task: Optional[asyncio.Task] = None
        self.signal_poll_timer: Optional["QTimer"] = None

    def attach_signal_poll_timer(self, timer: "QTimer") -> None:
        """Retain the OS-signal poll timer so it is not garbage-collected.

        Args:
            timer: Timer instance owned by the application entry point.
        """
        self.signal_poll_timer = timer

    def request_shutdown(self, *, reason: str, source: str) -> bool:
        """Request process shutdown. Thread-safe and idempotent.

        Args:
            reason: Human-readable reason for logging.
            source: Component name that initiated shutdown.

        Returns:
            True on the first request, False if shutdown was already requested.
        """
        with self.coord_lock:
            if self.requested:
                return False
            self.requested = True

        logger.info("Shutdown requested by %s: %s", source, reason)

        if self.init_task and not self.init_task.done():
            self.init_task.cancel()

        self.loop.call_soon_threadsafe(self.shutdown_event.set)
        return True

    def is_shutdown_requested(self) -> bool:
        """Return whether ``request_shutdown`` has been called successfully."""
        with self.coord_lock:
            return self.requested

    async def wait(self) -> None:
        """Block until shutdown has been requested."""
        await self.shutdown_event.wait()

    def register_initialization_task(self, task: asyncio.Task) -> None:
        """Track the bootstrap task so it can be cancelled on shutdown."""
        self.init_task = task

    def unregister_initialization_task(self) -> None:
        """Clear the bootstrap task reference after it completes."""
        self.init_task = None
