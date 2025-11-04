import asyncio
import inspect
import itertools
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Type

from vocalance.app.events.base_event import BaseEvent, EventPriority

logger = logging.getLogger(__name__)


class EventBus:
    """Asynchronous event bus with priority-based event processing and critical operation tracking.

    Central event distribution system using an asyncio priority queue with a dedicated worker
    task for dispatching events to registered subscribers. Supports both async and sync handlers,
    priority-based event ordering, graceful shutdown with queue draining, and critical operation
    registration to prevent premature shutdown during important processes. Thread-safe subscriber
    management enables registration from any thread while processing occurs in the event loop.

    Features thread pool executor for CPU-intensive handlers and backpressure management to
    prevent queue bloating under load.

    Attributes:
        _subscribers: Dictionary mapping event types to lists of handler callables.
        _event_queue: Priority queue ordering events by priority and insertion time.
        _worker_task: Background task processing events from the queue.
        _is_shutting_down: Flag indicating shutdown has been initiated.
        _critical_operations: Set of operation IDs preventing shutdown.
        _high_priority_sleep: Sleep duration between high-priority event processing.
        _low_priority_sleep: Sleep duration between normal/low-priority event processing.
        _thread_pool: ThreadPoolExecutor for running CPU-intensive handlers.
        _max_queue_size: Maximum queue size before backpressure kicks in.
    """

    def __init__(
        self,
        high_priority_sleep: float = 0.001,
        low_priority_sleep: float = 0.01,
        thread_pool_workers: int = 4,
        max_queue_size: int = 200,
    ) -> None:
        """Initialize the event bus with configurable processing delays and thread pool.

        Args:
            high_priority_sleep: Seconds to sleep between high-priority events (default 0.001).
            low_priority_sleep: Seconds to sleep between normal/low-priority events (default 0.01).
            thread_pool_workers: Number of worker threads for CPU-intensive handlers (default 4).
            max_queue_size: Maximum queue size before applying backpressure (default 200).
        """
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

        # Thread pool for CPU-intensive handlers
        self._thread_pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=thread_pool_workers, thread_name_prefix="EventBus-Worker"
        )

        # Backpressure management
        self._max_queue_size: int = max_queue_size
        self._events_dropped: int = 0

    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to the priority queue for asynchronous processing.

        Validates event type, rejects events during shutdown, adds event to priority queue
        with priority value and insertion counter for stable ordering. Implements backpressure
        by dropping lowest-priority events when queue exceeds maximum size.

        Args:
            event: BaseEvent subclass instance to publish to subscribers.
        """
        async with self._state_lock:
            is_shutting_down = self._is_shutting_down

        if is_shutting_down:
            logger.debug(f"Rejecting event {type(event).__name__} during shutdown")
            return

        if not isinstance(event, BaseEvent):
            logger.error(f"Event data must be a subclass of BaseEvent, got {type(event)}")
            return

        queue_size = self._event_queue.qsize()

        # Backpressure: Drop lowest-priority events when queue is full
        if queue_size >= self._max_queue_size:
            # Only drop LOW and NORMAL priority events, never CRITICAL or HIGH
            if event.priority >= EventPriority.NORMAL:
                self._events_dropped += 1
                logger.warning(
                    f"Queue full ({queue_size}/{self._max_queue_size}) - dropping {type(event).__name__} "
                    f"(priority={event.priority}, total_dropped={self._events_dropped})"
                )
                return
            else:
                # Critical/High priority: log error but allow (queue may grow beyond limit)
                logger.error(
                    f"Queue full ({queue_size}/{self._max_queue_size}) but forcing {type(event).__name__} "
                    f"(priority={event.priority}) - system overloaded!"
                )

        await self._event_queue.put((event.priority, next(self._counter), event))

        # Progressive warning levels
        if queue_size > self._max_queue_size * 0.75:
            logger.warning(f"Event queue at 75% capacity: {queue_size}/{self._max_queue_size} events")
        elif queue_size > self._max_queue_size * 0.5:
            logger.info(f"Event queue at 50% capacity: {queue_size}/{self._max_queue_size} events")

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable[[BaseEvent], Any]) -> None:
        """Subscribe a handler to receive events of a specific type.

        Validates that event_type is a BaseEvent subclass and handler is callable,
        then registers the handler to be invoked when matching events are processed.
        Supports both sync and async handlers. Thread-safe for registration from any thread.

        Args:
            event_type: BaseEvent subclass to subscribe to (matches via isinstance).
            handler: Callable accepting a single event parameter, sync or async.
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
        """Process events from the priority queue and dispatch to matching subscribers.

        Main event loop that continuously retrieves events from the priority queue,
        matches them to registered handlers via isinstance checks, invokes handlers
        (supporting both sync and async), measures handler execution time, applies
        priority-based sleep delays, and handles errors gracefully. Runs until
        shutdown is requested or the task is cancelled.
        """
        logger.debug("Event processing worker started")

        async with self._state_lock:
            is_shutting_down = self._is_shutting_down

        while not is_shutting_down:
            try:
                priority, _, event = await self._event_queue.get()

                event_type = type(event)
                handler_found = False

                # Optimize: only copy handlers for matching event types
                # Create shallow copy of handler list to prevent race conditions
                handlers_to_call = []
                with self._subscribers_lock:
                    for subscribed_type, handlers in self._subscribers.items():
                        if isinstance(event, subscribed_type):
                            handler_found = True
                            # Create a copy of the handlers list to avoid modification during iteration
                            handlers_to_call.extend(list(handlers))

                # Execute handlers outside lock for better concurrency
                for handler in handlers_to_call:
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
                        logger.error(f"Error in handler {handler_name} for event '{event_type.__name__}': {e}", exc_info=True)

                if not handler_found:
                    logger.debug(f"No handlers registered for event '{event_type.__name__}'")

                self._event_queue.task_done()

                # Adaptive sleep based on queue depth for optimal throughput vs CPU usage
                # Critical priority: skip sleep entirely for maximum responsiveness
                queue_depth = self._event_queue.qsize()

                if priority == EventPriority.CRITICAL:
                    # Fast path: no sleep for critical events
                    await asyncio.sleep(0)
                elif queue_depth == 0:
                    # Queue empty: sleep to reduce CPU usage
                    sleep_duration = self._low_priority_sleep if priority >= EventPriority.NORMAL else self._high_priority_sleep
                    await asyncio.sleep(sleep_duration)
                elif queue_depth < 10:
                    # Light load: minimal sleep
                    await asyncio.sleep(0)
                elif queue_depth < 50:
                    # Moderate load: very short sleep
                    await asyncio.sleep(0.001)
                elif queue_depth < 100:
                    # Heavy load: short sleep
                    await asyncio.sleep(0.005)
                else:
                    # Critical load: longer sleep + warning
                    await asyncio.sleep(0.01)
                    if queue_depth > self._max_queue_size * 0.9:
                        logger.warning(f"Event queue depth critical: {queue_depth} events")

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
        """Start the event processing worker task if not already running.

        Creates the background asyncio task that processes events from the queue.
        Safe to call multiple times - checks if worker is already running before
        creating a new task. Refuses to start if shutdown has been initiated.
        """
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

    async def run_in_thread_pool(self, func: Callable, *args, **kwargs) -> Any:
        """Run a CPU-intensive function in the thread pool executor.

        Helper method for handlers that need to perform blocking CPU-intensive work
        without blocking the event loop. Automatically handles loop acquisition and
        exception propagation.

        Args:
            func: Callable to execute in thread pool.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result from func execution.

        Example:
            result = await event_bus.run_in_thread_pool(expensive_computation, data)
        """
        if self._thread_pool is None:
            raise RuntimeError("Thread pool has been shutdown")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, lambda: func(*args, **kwargs))

    async def stop_worker(self) -> None:
        """Stop the event processing worker and clean up all subscribers.

        Initiates graceful shutdown by setting the shutdown flag, attempting to drain
        the event queue with a timeout, cancelling the worker task, and clearing all
        registered subscribers. Refuses to shut down if critical operations are active.
        """
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

        # Shutdown thread pool
        if self._thread_pool is not None:
            logger.debug("Shutting down thread pool executor")
            self._thread_pool.shutdown(wait=True, cancel_futures=False)
            self._thread_pool = None
            logger.debug("Thread pool shutdown complete")

    async def get_stats(self) -> Dict[str, Any]:
        """Get current event bus statistics for monitoring and debugging.

        Collects comprehensive metrics including queue size, subscriber counts per event type,
        worker task status, shutdown state, active critical operations, backpressure stats,
        and thread pool status. Useful for diagnosing event processing issues and monitoring
        system health.

        Returns:
            Dictionary with keys: queue_size, max_queue_size, queue_utilization, events_dropped,
            subscribers, worker_status, is_shutting_down, critical_operations, thread_pool_alive.
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

        queue_size = self._event_queue.qsize()

        return {
            "queue_size": queue_size,
            "max_queue_size": self._max_queue_size,
            "queue_utilization": f"{(queue_size / self._max_queue_size * 100):.1f}%",
            "events_dropped": self._events_dropped,
            "subscribers": subscribers,
            "worker_status": worker_status,
            "is_shutting_down": is_shutting_down,
            "critical_operations": critical_ops,
            "thread_pool_alive": self._thread_pool is not None,
        }

    async def register_critical_operation(self, operation_id: str) -> None:
        """Register a critical operation to prevent event bus shutdown.

        Adds an operation ID to the critical operations set, blocking shutdown until
        the operation is unregistered. Used to protect important processes like file
        I/O or model initialization from being interrupted by shutdown.

        Args:
            operation_id: Unique string identifier for the critical operation.
        """
        async with self._critical_ops_lock:
            self._critical_operations.add(operation_id)
        logger.debug(f"Registered critical operation: {operation_id}")

    async def unregister_critical_operation(self, operation_id: str) -> None:
        """Unregister a completed critical operation to allow shutdown.

        Removes the operation ID from the critical operations set. Should be called
        when the critical operation completes successfully or fails, to unblock shutdown.

        Args:
            operation_id: Unique string identifier for the critical operation to remove.
        """
        async with self._critical_ops_lock:
            self._critical_operations.discard(operation_id)
        logger.debug(f"Unregistered critical operation: {operation_id}")

    async def has_critical_operations(self) -> bool:
        """Check if any critical operations are currently registered.

        Queries the critical operations set in a thread-safe manner to determine
        if shutdown should be blocked.

        Returns:
            True if one or more critical operations are active, False otherwise.
        """
        async with self._critical_ops_lock:
            return len(self._critical_operations) > 0
