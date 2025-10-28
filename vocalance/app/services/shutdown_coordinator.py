import asyncio
import logging
import threading
from typing import Optional

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import ApplicationShutdownRequestedEvent


class ShutdownCoordinator:
    """Production-ready shutdown coordinator for graceful application shutdown.

    Provides a single source of truth for shutdown state, coordinates graceful shutdown
    across all components, prevents race conditions, and enables clean cancellation of
    async initialization. Thread-safe shutdown signaling with idempotent behavior.
    """

    def __init__(
        self,
        event_bus: EventBus,
        root_window,
        logger: Optional[logging.Logger] = None,
        gui_event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Initialize shutdown coordinator with application dependencies.

        Args:
            event_bus: EventBus for publishing shutdown events.
            root_window: Root Tkinter window for destruction.
            logger: Optional logger instance (uses module logger if None).
            gui_event_loop: Optional GUI event loop reference.
        """
        self.event_bus: EventBus = event_bus
        self.root_window = root_window
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.gui_event_loop: Optional[asyncio.AbstractEventLoop] = gui_event_loop

        self._shutdown_requested: bool = False
        self._shutdown_lock: threading.Lock = threading.Lock()
        self._shutdown_event: threading.Event = threading.Event()
        self._initialization_task: Optional[asyncio.Task] = None

    def set_gui_event_loop(self, gui_event_loop: asyncio.AbstractEventLoop) -> None:
        """Set the GUI event loop reference after initialization.

        Args:
            gui_event_loop: GUI event loop instance.
        """
        self.gui_event_loop = gui_event_loop

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
            self._shutdown_event.set()

        self.logger.info(f"Shutdown requested: {reason} (source: {source})")

        if self._initialization_task and not self._initialization_task.done():
            self.logger.debug("Cancelling initialization task due to shutdown request")
            self._initialization_task.cancel()

        if self.gui_event_loop and not self.gui_event_loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.event_bus.publish(ApplicationShutdownRequestedEvent(reason=reason, source=source)), self.gui_event_loop
                )
            except Exception as e:
                self.logger.debug(f"Could not publish shutdown event: {e}")
        else:
            self.logger.warning("GUI event loop not available for shutdown event publication")

        try:
            self.root_window.quit()
        except Exception as e:
            self.logger.error(f"Error calling quit() on root window: {e}")

        return True

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested (thread-safe).

        Returns:
            True if shutdown was requested, False otherwise.
        """
        with self._shutdown_lock:
            return self._shutdown_requested

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to be requested.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            True if shutdown was requested, False if timeout occurred.
        """
        return self._shutdown_event.wait(timeout=timeout)

    def register_initialization_task(self, task: asyncio.Task) -> None:
        """Register the initialization task for cancellation on shutdown.

        Args:
            task: Initialization task to register.
        """
        self._initialization_task = task
        self.logger.debug("Initialization task registered with shutdown coordinator")

    def unregister_initialization_task(self) -> None:
        """Clear the initialization task reference after it completes."""
        self._initialization_task = None
        self.logger.debug("Initialization task unregistered from shutdown coordinator")
