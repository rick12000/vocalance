from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Type

from vocalance.app.events.base_event import BaseEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Publish-subscribe bus with deferred start, MRO-based dispatch, and sync/async handlers.

    Handlers run on the asyncio loop passed to ``start``. One failing handler is logged
    and does not prevent other handlers from running. After ``shutdown``, ``publish`` is
    a no-op and the bus cannot be restarted.
    """

    def __init__(self) -> None:
        self.subscribers: Dict[Type[BaseEvent], List[Callable[..., Any]]] = defaultdict(list)
        self.bus_lock = threading.Lock()
        self.started = False
        self.closed = False
        self.pending_events: List[BaseEvent] = []

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Flush queued events and begin live dispatch on ``loop``.

        Call once, after all services have subscribed. ``loop`` must already be
        running so queued flushes and later ``publish`` calls use one coherent loop.

        Args:
            loop: The running asyncio event loop used for dispatch.

        Raises:
            RuntimeError: If ``loop`` is not running.
        """
        if not loop.is_running():
            raise RuntimeError("EventBus.start requires a running asyncio event loop.")

        with self.bus_lock:
            if self.closed:
                raise RuntimeError("EventBus cannot be started after shutdown.")
            if self.started:
                return
            self.started = True
            queued: List[BaseEvent] = list(self.pending_events)
            self.pending_events.clear()

        if queued:
            loop.create_task(self.flush_pending_events(queued))

    async def flush_pending_events(self, events: List[BaseEvent]) -> None:
        """Dispatch a batch of events that were queued before ``start``."""
        for event in events:
            await self.dispatch_event(event)

    async def shutdown(self) -> None:
        """Stop accepting publishes and clear subscribers and queues."""
        with self.bus_lock:
            self.closed = True
            self.started = False
            self.subscribers.clear()
            self.pending_events.clear()

    async def publish(self, event: BaseEvent) -> None:
        """Publish ``event`` to subscribers, queue if not started, or no-op if shut down."""
        with self.bus_lock:
            if self.closed:
                return
            if not self.started:
                self.pending_events.append(event)
                return

        await self.dispatch_event(event)

    async def dispatch_event(self, event: BaseEvent) -> None:
        handlers: List[Callable[..., Any]] = []
        with self.bus_lock:
            for cls in type(event).__mro__:
                if cls in self.subscribers:
                    handlers.extend(self.subscribers[cls])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:
                logger.error("Handler %s failed for %s", handler, type(event).__name__, exc_info=True)

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable[..., Any]) -> None:
        """Register ``handler`` for exact ``event_type`` (and MRO dispatch on publish)."""
        with self.bus_lock:
            if self.closed:
                return
            self.subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[BaseEvent], handler: Callable[..., Any]) -> None:
        """Remove ``handler`` from ``event_type`` if present."""
        with self.bus_lock:
            if self.closed:
                return
            subs = self.subscribers.get(event_type)
            if subs and handler in subs:
                subs.remove(handler)
