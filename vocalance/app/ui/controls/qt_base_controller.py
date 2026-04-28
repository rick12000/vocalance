from __future__ import annotations

import logging
from typing import Any, Callable, Generic, Optional, Type, TypeVar

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget

from vocalance.app.event_bus import EventBus, SubscriptionTracker
from vocalance.app.events.base_event import BaseEvent

ViewT = TypeVar("ViewT", bound=QWidget)


class QtBaseController(QObject, Generic[ViewT]):
    """Minimal controller base: event bus, logger, optional bound view, tracked subs.

    Subclasses register handlers via ``self.subscribe(EventType, handler)`` in
    ``__init__``. ``shutdown`` (invoked by ``UiRegistry.shutdown``, which is itself
    invoked by ``AppLifecycle`` during teardown) unsubscribes everything that was
    registered, so subclasses never need to write a manual ``event_bus.unsubscribe``.
    """

    status_updated = Signal(str, bool)

    def __init__(self, event_bus: EventBus, logger: logging.Logger) -> None:
        super().__init__()
        self.event_bus = event_bus
        self.logger = logger
        self._attached_view: Optional[ViewT] = None
        self._subs = SubscriptionTracker(event_bus=event_bus)

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable[..., Any]) -> None:
        """Subscribe ``handler`` to ``event_type`` and remember it for teardown."""
        self._subs.subscribe(event_type, handler)

    def set_view(self, view: ViewT) -> None:
        self._attached_view = view

    def get_view(self) -> Optional[ViewT]:
        return self._attached_view

    def emit_status(self, message: str, is_error: bool = False) -> None:
        self.status_updated.emit(message, is_error)

    def shutdown(self) -> None:
        """Unsubscribe every recorded handler and release the bound view.

        Subclasses must call ``super().shutdown()`` after their own cleanup steps.
        """
        self._subs.unsubscribe_all()
        self._attached_view = None
