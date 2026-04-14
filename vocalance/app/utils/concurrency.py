import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Type

from vocalance.app.event_bus import EventBus
from vocalance.app.events.base_event import BaseEvent

logger = logging.getLogger(__name__)


def schedule_on_loop(loop: asyncio.AbstractEventLoop, coro) -> None:
    """Schedule a coroutine from any OS thread onto the given asyncio event loop."""
    loop.call_soon_threadsafe(loop.create_task, coro)


@dataclass
class SubscriptionTracker:
    """Tracks event subscriptions for a component and cleans them up in one call."""

    event_bus: EventBus
    _subs: List[Tuple[Type[BaseEvent], Callable]] = field(default_factory=list, init=False)

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable) -> None:
        self.event_bus.subscribe(event_type, handler)
        self._subs.append((event_type, handler))

    def unsubscribe_all(self) -> None:
        for event_type, handler in self._subs:
            self.event_bus.unsubscribe(event_type, handler)
        self._subs.clear()
