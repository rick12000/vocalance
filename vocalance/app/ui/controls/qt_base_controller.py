"""Base Qt controller with signal-based event handling.

All controllers inherit from this base to get signal/slot support.
"""

import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal


class QtBaseController(QObject):
    """Base class for Qt controllers with signal support.

    Provides:
    - Signal definitions for common events
    - Thread-safe event emission
    - Logging
    - View reference management
    """

    # Common signals all controllers can emit
    status_updated = Signal(str, bool)  # message, is_error
    error_occurred = Signal(str)  # error_message

    def __init__(self, event_bus, event_loop, logger: logging.Logger):
        """Initialize Qt base controller.

        Args:
            event_bus: Event bus for pub/sub.
            event_loop: Asyncio event loop.
            logger: Logger instance.
        """
        super().__init__()

        self.event_bus = event_bus
        self.event_loop = event_loop
        self.logger = logger
        self._view = None

    def set_view(self, view: Any) -> None:
        """Set the associated view.

        Args:
            view: View instance.
        """
        self._view = view

    def get_view(self) -> Optional[Any]:
        """Get the associated view.

        Returns:
            View instance or None.
        """
        return self._view

    def emit_status(self, message: str, is_error: bool = False) -> None:
        """Emit status update signal.

        Args:
            message: Status message.
            is_error: Whether this is an error.
        """
        self.status_updated.emit(message, is_error)

    def emit_error(self, error_message: str) -> None:
        """Emit error signal.

        Args:
            error_message: Error message.
        """
        self.error_occurred.emit(error_message)

    def cleanup(self) -> None:
        """Clean up controller resources."""
        self._view = None
        self.logger.debug(f"{self.__class__.__name__} cleanup completed")
