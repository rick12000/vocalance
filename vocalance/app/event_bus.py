import asyncio
import inspect
import itertools
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Type

from vocalance.app.events.base_event import BaseEvent, EventPriority

logger = logging.getLogger(__name__)


class EventBus:
    """Asynchronous event bus with priority-based event processing.

    Uses a priority queue to process events efficiently with a worker task
    handling event delivery to subscribers. Thread-safe for mixed async/sync contexts.
    """

    def __init__(self, high_priority_sleep: float = 0.001, low_priority_sleep: float = 0.01) -> None:
        self._subscribers: Dict[Type[BaseEvent], List[Callable[[BaseEvent], Any]]] = defaultdict(list)
        self._event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._worker_task: Any = None
        self._is_shutting_down: bool = False
        self._critical_operations: set = set()
        self._high_priority_sleep: float = high_priority_sleep
        self._low_priority_sleep: float = low_priority_sleep
        self._counter: itertools.count = itertools.count()
        self._state_lock: asyncio.Lock = asyncio.Lock()
        self._critical_ops_lock: asyncio.Lock = asyncio.Lock()
        self._subscribers_lock: threading.RLock = threading.RLock()

    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to the bus for processing.

        Args:
            event: Event instance to publish.
        """
        async with self._state_lock:
            is_shutting_down = self._is_shutting_down

        if is_shutting_down:
            logger.debug(f"Rejecting event {type(event).__name__} during shutdown")
            return

        if not isinstance(event, BaseEvent):
            logger.error(f"Event data must be a subclass of BaseEvent, got {type(event)}")
            return

        await self._event_queue.put((event.priority, next(self._counter), event))

        queue_size = self._event_queue.qsize()
        if queue_size > 50:
            logger.warning(f"Event queue size is large: {queue_size} events")
        elif queue_size > 100:
            logger.error(f"Event queue size is critical: {queue_size} events - system may be overloaded")

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable[[BaseEvent], Any]) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: Event class to subscribe to.
            handler: Callable to invoke when event is published.
        """
        if not inspect.isclass(event_type) or not issubclass(event_type, BaseEvent):
            logger.error(f"Can only subscribe to subclasses of BaseEvent, got {event_type}")
            return

        if not callable(handler):
            logger.error(f"Handler must be callable, got {type(handler)}")
            return

        logger.debug(
            f"Subscribing handler {handler.__name__ if hasattr(handler, '__name__') else handler} to event type: {event_type.__name__}"
        )
        with self._subscribers_lock:
            self._subscribers[event_type].append(handler)

    async def _process_events(self) -> None:
        """Process events from the queue and dispatch to subscribers."""
        logger.debug("Event processing worker started")

        async with self._state_lock:
            is_shutting_down = self._is_shutting_down

        while not is_shutting_down:
            try:
                priority, _, event = await self._event_queue.get()

                event_type = type(event)
                handler_found = False

                with self._subscribers_lock:
                    current_subscribers = {event_type: list(handlers) for event_type, handlers in self._subscribers.items()}

                for subscribed_type, handlers in current_subscribers.items():
                    if isinstance(event, subscribed_type):
                        handler_found = True
                        for handler in handlers:
                            try:
                                handler_start = time.monotonic()
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(event)
                                else:
                                    handler(event)

                                handler_time = time.monotonic() - handler_start
                                if handler_time > 0.1:
                                    logger.warning(
                                        f"Slow handler {handler.__name__ if hasattr(handler, '__name__') else handler} for event '{event_type.__name__}': {handler_time:.4f}s"
                                    )

                            except Exception as e:
                                handler_name = handler.__name__ if hasattr(handler, "__name__") else str(handler)
                                logger.error(
                                    f"Error in handler {handler_name} for event '{event_type.__name__}': {e}", exc_info=True
                                )

                if not handler_found:
                    logger.debug(f"No handlers registered for event '{event_type.__name__}'")

                self._event_queue.task_done()

                sleep_duration = self._low_priority_sleep if priority >= EventPriority.NORMAL else self._high_priority_sleep
                await asyncio.sleep(sleep_duration)

                async with self._state_lock:
                    is_shutting_down = self._is_shutting_down

            except asyncio.CancelledError:
                logger.debug("Event processing worker cancelled")
                break
            except Exception as e:
                logger.critical(f"Fatal error in event processing worker: {e}", exc_info=True)
                async with self._state_lock:
                    is_shutting_down = self._is_shutting_down
                if not is_shutting_down:
                    await asyncio.sleep(1)

    async def start_worker(self) -> None:
        """Start the event processing worker task."""
        async with self._state_lock:
            is_shutting_down = self._is_shutting_down

        if is_shutting_down:
            logger.debug("Not starting worker during shutdown")
            return

        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_events())
            logger.debug("Event bus worker started")
        else:
            logger.debug("Event bus worker already running")

    async def stop_worker(self) -> None:
        """Stop the event processing worker and clean up subscribers."""
        if await self.has_critical_operations():
            async with self._critical_ops_lock:
                critical_ops = list(self._critical_operations)
            logger.warning(f"Cannot shutdown event bus - critical operations active: {critical_ops}")
            return

        async with self._state_lock:
            self._is_shutting_down = True

        try:
            await asyncio.wait_for(self._event_queue.join(), timeout=2.0)
            logger.debug("Event queue successfully drained before shutdown")
        except asyncio.TimeoutError:
            remaining = self._event_queue.qsize()
            logger.warning(f"Could not process all events before shutdown. {remaining} events discarded.")

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                logger.debug("Event bus worker successfully stopped")

        with self._subscribers_lock:
            logger.debug(f"Clearing {len(self._subscribers)} subscriber lists")
            for event_type in list(self._subscribers.keys()):
                self._subscribers[event_type].clear()
            self._subscribers.clear()
        logger.debug("All event subscribers cleared")

    async def get_stats(self) -> Dict[str, Any]:
        """Get current event bus statistics.

        Returns:
            Dictionary containing queue size, subscriber counts, worker status, etc.
        """
        worker_status = "running"
        if self._worker_task is None:
            worker_status = "not_created"
        elif self._worker_task.done():
            worker_status = "stopped"
            if self._worker_task.cancelled():
                worker_status = "cancelled"
            elif self._worker_task.exception():
                worker_status = f"error: {self._worker_task.exception()}"

        async with self._state_lock:
            is_shutting_down = self._is_shutting_down

        async with self._critical_ops_lock:
            critical_ops = list(self._critical_operations)

        with self._subscribers_lock:
            subscribers = {event.__name__: len(handlers) for event, handlers in self._subscribers.items()}

        return {
            "queue_size": self._event_queue.qsize(),
            "subscribers": subscribers,
            "worker_status": worker_status,
            "is_shutting_down": is_shutting_down,
            "critical_operations": critical_ops,
        }

    async def register_critical_operation(self, operation_id: str) -> None:
        """Register a critical operation to prevent shutdown.

        Args:
            operation_id: Unique identifier for the operation.
        """
        async with self._critical_ops_lock:
            self._critical_operations.add(operation_id)
        logger.debug(f"Registered critical operation: {operation_id}")

    async def unregister_critical_operation(self, operation_id: str) -> None:
        """Unregister a critical operation.

        Args:
            operation_id: Unique identifier for the operation.
        """
        async with self._critical_ops_lock:
            self._critical_operations.discard(operation_id)
        logger.debug(f"Unregistered critical operation: {operation_id}")

    async def has_critical_operations(self) -> bool:
        """Check if any critical operations are registered.

        Returns:
            True if critical operations exist, False otherwise.
        """
        async with self._critical_ops_lock:
            return len(self._critical_operations) > 0
