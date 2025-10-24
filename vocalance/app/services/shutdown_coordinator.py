import asyncio
import logging
import threading
from typing import Optional

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import ApplicationShutdownRequestedEvent


class ShutdownCoordinator:
    """
    Production-ready shutdown coordinator following GUI best practices.

    Responsibilities:
    - Provides a single source of truth for shutdown state
    - Coordinates graceful shutdown across all application components
    - Prevents race conditions during concurrent shutdown requests
    - Enables clean cancellation of async initialization
    - Thread-safe shutdown signaling
    - Tracks service initializer for partial shutdown

    Design principles:
    - Single Responsibility: Only manages shutdown coordination
    - Observer Pattern: Uses event bus for loose coupling
    - Fail-Safe: Multiple shutdown requests are idempotent
    - Testable: Clear interface with minimal dependencies
    """

    def __init__(
        self,
        event_bus: EventBus,
        root_window,
        logger: Optional[logging.Logger] = None,
        gui_event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.event_bus = event_bus
        self.root_window = root_window
        self.logger = logger or logging.getLogger(__name__)
        self.gui_event_loop = gui_event_loop

        self._shutdown_requested = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._initialization_task: Optional[asyncio.Task] = None

    def set_gui_event_loop(self, gui_event_loop: asyncio.AbstractEventLoop) -> None:
        """Set the GUI event loop reference after initialization."""
        self.gui_event_loop = gui_event_loop

    def request_shutdown(self, reason: str, source: str) -> bool:
        """
        Request application shutdown.

        Thread-safe, idempotent operation that can be called from any thread.
        Returns True if this is the first shutdown request, False if already shutting down.
        """
        with self._shutdown_lock:
            if self._shutdown_requested:
                self.logger.info(f"Shutdown already in progress. Ignoring duplicate request from {source}")
                return False

            self._shutdown_requested = True
            self._shutdown_event.set()

        self.logger.info(f"Shutdown requested: {reason} (source: {source})")

        # Cancel initialization if it's running
        if self._initialization_task and not self._initialization_task.done():
            self.logger.info("Cancelling initialization task due to shutdown request")
            self._initialization_task.cancel()

        # Publish shutdown event for observers
        if self.gui_event_loop and not self.gui_event_loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.event_bus.publish(ApplicationShutdownRequestedEvent(reason=reason, source=source)), self.gui_event_loop
                )
            except Exception as e:
                self.logger.debug(f"Could not publish shutdown event: {e}")
        else:
            self.logger.warning("GUI event loop not available for shutdown event publication")

        # Trigger Tkinter mainloop exit
        try:
            self.root_window.quit()
        except Exception as e:
            self.logger.error(f"Error calling quit() on root window: {e}")

        return True

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested (thread-safe)."""
        with self._shutdown_lock:
            return self._shutdown_requested

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for shutdown to be requested.
        Returns True if shutdown was requested, False if timeout occurred.
        """
        return self._shutdown_event.wait(timeout=timeout)

    def register_initialization_task(self, task: asyncio.Task) -> None:
        """
        Register the initialization task so it can be cancelled on shutdown.
        This enables clean cancellation of async initialization.
        """
        self._initialization_task = task
        self.logger.debug("Initialization task registered with shutdown coordinator")

    def unregister_initialization_task(self) -> None:
        """Clear the initialization task reference after it completes."""
        self._initialization_task = None
        self.logger.debug("Initialization task unregistered from shutdown coordinator")
