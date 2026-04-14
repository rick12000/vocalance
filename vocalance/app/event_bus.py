"""Central event bus.

Design:
- Services subscribe in their ``__init__``.  No deferred ``setup_subscriptions`` step.
- The bus starts in a *paused* state.  Events published before ``start()`` are queued
  and flushed in order once the bus is started.  This eliminates the bootstrapping
  race between "services not yet subscribed" and "events published during init".
- Handlers may be either ``async def`` or plain ``def``.  Async handlers are awaited;
  sync handlers are called directly.  Both are correct; use whichever is honest for
  the implementation (async for I/O-bound service handlers, sync for UI signal emitters).
- Dispatch is direct: the handler set for each event type is pre-computed at
  subscribe-time (exact type stored in a dict), not re-scanned on every publish.
  Inheritance is supported via ``publish`` iterating the MRO.
- Thread-safe subscribe/unsubscribe from any thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from typing import Callable, Dict, List, Type, Union

from vocalance.app.events.base_event import BaseEvent

logger = logging.getLogger(__name__)

Handler = Union[Callable, Callable]


class EventBus:
    """Publish-subscribe bus with deferred start, MRO-based dispatch, and sync/async handlers."""

    def __init__(self) -> None:
        self._subscribers: Dict[Type[BaseEvent], List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._started = False
        self._queue: List[BaseEvent] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Flush queued events and begin live dispatch.

        Call once, after all services have subscribed.
        """
        with self._lock:
            self._started = True
            queued = list(self._queue)
            self._queue.clear()

        if queued:
            loop = asyncio.get_event_loop()
            loop.create_task(self._flush(queued))

    async def _flush(self, events: List[BaseEvent]) -> None:
        for event in events:
            await self._dispatch(event)

    async def shutdown(self) -> None:
        """Clear all subscribers."""
        with self._lock:
            self._subscribers.clear()
            self._queue.clear()

    # ------------------------------------------------------------------
    # Pub/sub
    # ------------------------------------------------------------------

    async def publish(self, event: BaseEvent) -> None:
        """Dispatch event to all handlers registered for its type or any base type."""
        with self._lock:
            if not self._started:
                self._queue.append(event)
                return

        await self._dispatch(event)

    async def _dispatch(self, event: BaseEvent) -> None:
        handlers: List[Callable] = []
        with self._lock:
            for cls in type(event).__mro__:
                if cls in self._subscribers:
                    handlers.extend(self._subscribers[cls])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:
                logger.error("Handler %s failed for %s", handler, type(event).__name__, exc_info=True)

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable) -> None:
        """Register a handler for ``event_type``. May be sync or async."""
        with self._lock:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[BaseEvent], handler: Callable) -> None:
        """Remove a previously registered handler (no-op if not found)."""
        with self._lock:
            subscribers = self._subscribers.get(event_type)
            if subscribers and handler in subscribers:
                subscribers.remove(handler)
