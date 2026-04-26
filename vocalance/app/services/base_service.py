from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Type

from vocalance.app.event_bus import EventBus
from vocalance.app.events.base_event import BaseEvent
from vocalance.app.lifecycle.concurrency import SubscriptionTracker


class Service(ABC):
    """Contract every application service must satisfy.

    Subclasses register handlers via ``self.subscribe(...)`` in ``__init__``,
    optionally implement ``initialize`` for async setup, and call
    ``await super().shutdown()`` to release every recorded subscription. There
    is no need - and no reason - to write manual ``event_bus.unsubscribe(...)``
    calls anywhere.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._subs = SubscriptionTracker(event_bus=event_bus)

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable[..., Any]) -> None:
        """Subscribe ``handler`` to ``event_type`` and remember it for teardown."""
        self._subs.subscribe(event_type, handler)

    async def initialize(self) -> bool:
        """Optional async startup (e.g. storage reads, heavy imports).

        Returns:
            True when startup succeeded; services may use False to signal failure.
        """
        return True

    async def shutdown(self) -> None:
        """Default teardown: unsubscribe every recorded handler.

        Subclasses that need additional cleanup must call ``await super().shutdown()``
        as the last step.
        """
        self._subs.unsubscribe_all()
