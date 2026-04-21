import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from vocalance.app.event_bus import EventBus


class QtBaseController(QObject):
    """Minimal controller base: event bus, logger, and optional bound view."""

    status_updated = Signal(str, bool)

    def __init__(self, event_bus: EventBus, logger: logging.Logger) -> None:
        super().__init__()
        self.event_bus = event_bus
        self.logger = logger
        self._attached_view: Any = None

    def set_view(self, view: Any) -> None:
        self._attached_view = view

    def get_view(self) -> Optional[Any]:
        return self._attached_view

    def emit_status(self, message: str, is_error: bool = False) -> None:
        self.status_updated.emit(message, is_error)

    def cleanup(self) -> None:
        self._attached_view = None
