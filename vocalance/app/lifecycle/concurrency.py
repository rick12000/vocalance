from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, List, Tuple, Type

from vocalance.app.event_bus import EventBus
from vocalance.app.events.base_event import BaseEvent


def schedule_on_loop(loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, Any]) -> None:
    """Schedule ``coro`` on ``loop`` as a task from any thread.

    Args:
        loop: Target asyncio event loop.
        coro: Coroutine instance to run as a task on that loop.
    """
    loop.call_soon_threadsafe(loop.create_task, coro)


def schedule_on_loop_callback(
    loop: asyncio.AbstractEventLoop,
    fn: Callable[..., Any],
    *args: Any,
) -> None:
    """Schedule a plain callable to run on ``loop`` from any thread.

    Args:
        loop: Target asyncio event loop.
        fn: Synchronous callable to execute on the loop thread.
        *args: Positional arguments forwarded to ``fn``.
    """
    loop.call_soon_threadsafe(fn, *args)


class SubscriptionTracker:
    """Records event-bus subscriptions for a component and unsubscribes them in one call.

    Owned by the ``Service`` and ``QtBaseController`` bases. Subclasses register
    handlers via ``self.subscribe(EventType, handler)`` in ``__init__``;
    ``super().shutdown()`` releases everything.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._subscriptions: List[Tuple[Type[BaseEvent], Callable[..., Any]]] = []

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable[..., Any]) -> None:
        """Subscribe ``handler`` to ``event_type`` and record it for later cleanup."""
        self.event_bus.subscribe(event_type, handler)
        self._subscriptions.append((event_type, handler))

    def unsubscribe_all(self) -> None:
        """Unsubscribe every handler previously registered via ``subscribe``."""
        for event_type, handler in self._subscriptions:
            self.event_bus.unsubscribe(event_type, handler)
        self._subscriptions.clear()
