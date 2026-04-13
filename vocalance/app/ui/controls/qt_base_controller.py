import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from vocalance.app.event_bus import EventBus


class QtBaseController(QObject):
    """Base class for Qt controllers providing shared event bus access, signals, and logging.

    All controllers run on the Qt main thread (via PySide6.QtAsyncio). Async handlers
    are scheduled directly on the running asyncio loop; no cross-thread scheduling is needed.
    """

    status_updated = Signal(str, bool)
    error_occurred = Signal(str)

    def __init__(self, event_bus: EventBus, logger: logging.Logger) -> None:
        """Initialize the base controller.

        Args:
            event_bus: Application event bus for pub/sub.
            logger: Logger instance for this controller.
        """
        super().__init__()

        self.event_bus = event_bus
        self.logger = logger
        self._view = None

    def set_view(self, view: Any) -> None:
        """Set the associated view.

        Args:
            view: View instance to associate with this controller.
        """
        self._view = view

    def get_view(self) -> Optional[Any]:
        """Return the associated view, or None if not set."""
        return self._view

    def emit_status(self, message: str, is_error: bool = False) -> None:
        """Emit a status update signal.

        Args:
            message: Status message text.
            is_error: True if this represents an error condition.
        """
        self.status_updated.emit(message, is_error)

    def emit_error(self, error_message: str) -> None:
        """Emit an error signal.

        Args:
            error_message: Error description.
        """
        self.error_occurred.emit(error_message)

    def cleanup(self) -> None:
        """Release view reference and log cleanup completion."""
        self._view = None
        self.logger.debug(f"{self.__class__.__name__} cleanup completed")
