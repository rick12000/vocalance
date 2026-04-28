from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Type

from vocalance.app.events.base_event import BaseEvent

logger = logging.getLogger(__name__)


class SubscriptionTracker:
    """Records event-bus subscriptions for a component and unsubscribes them in one call.

    Owned by the ``Service`` and ``QtBaseController`` bases. Subclasses register
    handlers via ``self.subscribe(EventType, handler)`` in ``__init__``;
    ``super().shutdown()`` releases everything.
    """

    def __init__(self, event_bus: "EventBus") -> None:
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


class EventBus:
    """Publish-subscribe bus with deferred start, MRO-based dispatch, and sync/async handlers.

    This modern implementation decouples publishing from processing:
    - Publishing is non-blocking and immediate, dropping events into a bounded queue.
    - If the system is catastrophically overloaded, backpressure is applied to the publisher.
    - Events are processed strictly sequentially (Event A completes fully before Event B begins)
      to guarantee causal ordering and prevent state corruption or race conditions.
    - Within a single event, synchronous handlers run immediately, and all asynchronous
      handlers run concurrently via `asyncio.gather` for maximum efficiency.

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
        self._queue: asyncio.Queue[BaseEvent] | None = None
        self._worker_task: asyncio.Task[None] | None = None

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

            self._queue = asyncio.Queue(maxsize=500)

            queued: List[BaseEvent] = list(self.pending_events)
            self.pending_events.clear()

        for event in queued:
            self._queue.put_nowait(event)

        self._worker_task = loop.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        """Background worker that processes events sequentially from the queue."""
        if self._queue is None:
            return

        while True:
            event = await self._queue.get()
            try:
                await self._dispatch_event(event)
            except Exception as e:
                logger.error("Catastrophic error in event dispatch: %s", e, exc_info=True)
            finally:
                self._queue.task_done()

    async def shutdown(self) -> None:
        """Stop accepting publishes and clear subscribers and queues."""
        with self.bus_lock:
            self.closed = True
            self.started = False
            self.subscribers.clear()
            self.pending_events.clear()

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def publish(self, event: BaseEvent) -> None:
        """Publish ``event`` to subscribers, queue if not started, or no-op if shut down.

        This method is non-blocking under normal conditions. It will only block if the
        event queue is full (backpressure), preventing memory exhaustion.

        Args:
            event: The event instance to publish.
        """
        with self.bus_lock:
            if self.closed:
                return
            if not self.started:
                self.pending_events.append(event)
                return

        if self._queue is not None:
            await self._queue.put(event)

    async def _dispatch_event(self, event: BaseEvent) -> None:
        """Dispatch a single event to all registered handlers.

        Synchronous handlers are executed sequentially and immediately.
        Asynchronous handlers are executed concurrently. The method waits for all
        async handlers to finish before returning, ensuring strict inter-event ordering.

        Args:
            event: The event instance to dispatch.
        """
        handlers: List[Callable[..., Any]] = []
        with self.bus_lock:
            for cls in type(event).__mro__:
                if cls in self.subscribers:
                    handlers.extend(self.subscribers[cls])

        async_tasks = []

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    async_tasks.append(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error("Sync handler %s failed for %s: %s", handler.__name__, type(event).__name__, e, exc_info=True)

        if async_tasks:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)

            for coro, result in zip(async_tasks, results):
                if isinstance(result, Exception):
                    logger.error(
                        "Async handler failed for %s: %s",
                        type(event).__name__,
                        result,
                        exc_info=(type(result), result, result.__traceback__),
                    )

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
