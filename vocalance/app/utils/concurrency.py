import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Tuple, Type

from vocalance.app.event_bus import EventBus
from vocalance.app.events.base_event import BaseEvent


def schedule_on_loop(loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, Any]) -> None:
    """Schedule ``coro`` on ``loop`` from any thread (thread-safe).

    Args:
        loop: Target asyncio event loop.
        coro: Coroutine instance to run as a task on that loop.
    """
    loop.call_soon_threadsafe(loop.create_task, coro)


@dataclass
class SubscriptionTracker:
    """Tracks event subscriptions for a component and unsubscribes them in one call."""

    event_bus: EventBus
    subscriptions: List[Tuple[Type[BaseEvent], Callable[..., Any]]] = field(default_factory=list, init=False)

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable[..., Any]) -> None:
        """Subscribe ``handler`` to ``event_type`` and record it for later cleanup.

        Args:
            event_type: Event class to subscribe to.
            handler: Callback invoked when an event of that type is published.
        """
        self.event_bus.subscribe(event_type, handler)
        self.subscriptions.append((event_type, handler))

    def unsubscribe_all(self) -> None:
        """Unsubscribe all handlers previously registered via ``subscribe``."""
        for event_type, handler in self.subscriptions:
            self.event_bus.unsubscribe(event_type, handler)
        self.subscriptions.clear()
