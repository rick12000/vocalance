import asyncio
import logging
import threading
from typing import Optional


class ShutdownCoordinator:
    """Production-ready shutdown coordinator for graceful application shutdown.

    Provides a single source of truth for shutdown state, coordinates graceful shutdown
    across all components, prevents race conditions, and enables clean cancellation of
    async initialization. Thread-safe shutdown signaling with idempotent behavior.
    """

    def __init__(self, shutdown_future: asyncio.Future, logger: Optional[logging.Logger] = None) -> None:
        """Initialize shutdown coordinator with application dependencies.

        Args:
            shutdown_future: A future that will be resolved to trigger the main application shutdown.
            logger: Optional logger instance (uses module logger if None).
        """
        self.shutdown_future = shutdown_future
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        self._shutdown_requested: bool = False
        self._shutdown_lock: threading.Lock = threading.Lock()
        self._initialization_task: Optional[asyncio.Task] = None

    def request_shutdown(self, reason: str, source: str) -> bool:
        """Request application shutdown.

        Thread-safe, idempotent operation that can be called from any thread.
        Returns True if this is the first shutdown request, False if already shutting down.

        Args:
            reason: Reason for shutdown request.
            source: Source of shutdown request.

        Returns:
            True if this is the first shutdown request, False if already shutting down.
        """
        with self._shutdown_lock:
            if self._shutdown_requested:
                self.logger.debug(f"Shutdown already in progress. Ignoring duplicate request from {source}")
                return False

            self._shutdown_requested = True

        self.logger.info(f"Shutdown requested: {reason} (source: {source})")

        if self._initialization_task and not self._initialization_task.done():
            self.logger.debug("Cancelling initialization task due to shutdown request")
            self._initialization_task.cancel()

        if not self.shutdown_future.done():
            loop = self.shutdown_future.get_loop()
            loop.call_soon_threadsafe(self.shutdown_future.set_result, None)

        return True

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested (thread-safe).

        Returns:
            True if shutdown was requested, False otherwise.
        """
        with self._shutdown_lock:
            return self._shutdown_requested

    def register_initialization_task(self, task: asyncio.Task) -> None:
        """Register the initialization task so it can be cancelled on shutdown."""
        self._initialization_task = task

    def unregister_initialization_task(self) -> None:
        """Clear the initialization task reference after it completes."""
        self._initialization_task = None
