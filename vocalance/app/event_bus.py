import asyncio
import inspect
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Type

from vocalance.app.events.base_event import BaseEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Synchronous event bus with critical operation tracking.

    Central event distribution system that directly dispatches events to registered subscribers.
    Supports both async and sync handlers, graceful shutdown, and critical operation
    registration to prevent premature shutdown during important processes. Thread-safe subscriber
    management enables registration from any thread.

    Attributes:
        _subscribers: Dictionary mapping event types to lists of handler callables.
        _is_shutting_down: Flag indicating shutdown has been initiated.
        _critical_operations: Set of operation IDs preventing shutdown.
    """

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._subscribers: Dict[Type[BaseEvent], List[Callable[[BaseEvent], Any]]] = defaultdict(list)
        self._is_shutting_down: bool = False
        self._critical_operations: set = set()
        self._lock: threading.RLock = threading.RLock()

    async def publish(self, published_event: BaseEvent) -> None:
        """Publish an event and synchronously dispatch to matching subscribers.

        Validates event type, rejects events during shutdown, and directly executes
        all matching handlers. Awaits async handlers and calls sync handlers directly.

        Args:
            published_event: BaseEvent subclass instance to publish to subscribers.
        """
        with self._lock:
            is_shutting_down = self._is_shutting_down

        if is_shutting_down:
            logger.debug(f"Rejecting event {type(published_event).__name__} during shutdown")
            return

        if not isinstance(published_event, BaseEvent):
            logger.error(f"Event data must be a subclass of BaseEvent, got {type(published_event)}")
            return

        dispatched_type = type(published_event)
        handler_found = False

        # Optimize: only copy handlers for matching event types
        # Create shallow copy of handler list to prevent race conditions
        handlers_to_call = []
        with self._lock:
            for subscribed_type, handlers in self._subscribers.items():
                if isinstance(published_event, subscribed_type):
                    handler_found = True
                    # Create a copy of the handlers list to avoid modification during iteration
                    handlers_to_call.extend(list(handlers))

        # Execute handlers outside lock for better concurrency
        for handler in handlers_to_call:
            try:
                handler_start = time.monotonic()
                if asyncio.iscoroutinefunction(handler):
                    await handler(published_event)
                else:
                    handler(published_event)

                handler_time = time.monotonic() - handler_start
                if handler_time > 0.1:
                    logger.warning(
                        f"Slow handler {handler.__name__ if hasattr(handler, '__name__') else handler} for event '{dispatched_type.__name__}': {handler_time:.4f}s"
                    )

            except Exception as e:
                handler_name = handler.__name__ if hasattr(handler, "__name__") else str(handler)
                logger.error(f"Error in handler {handler_name} for event '{dispatched_type.__name__}': {e}", exc_info=True)

        if not handler_found:
            logger.debug(f"No handlers registered for event '{dispatched_type.__name__}'")

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
        with self._lock:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[BaseEvent], handler: Callable[[BaseEvent], Any]) -> bool:
        """Unsubscribe a handler from receiving events of a specific type.

        Removes the handler from the subscriber list for the given event type.
        Thread-safe for unsubscription from any thread.

        Args:
            event_type: BaseEvent subclass to unsubscribe from.
            handler: The exact handler callable that was previously subscribed.

        Returns:
            True if the handler was found and removed, False otherwise.
        """
        handler_name = handler.__name__ if hasattr(handler, "__name__") else str(handler)

        with self._lock:
            if event_type not in self._subscribers:
                logger.debug(f"No subscribers for event type {event_type.__name__}, cannot unsubscribe {handler_name}")
                return False

            handlers = self._subscribers[event_type]
            if handler in handlers:
                handlers.remove(handler)
                logger.debug(f"Unsubscribed handler {handler_name} from event type: {event_type.__name__}")
                return True
            else:
                logger.debug(f"Handler {handler_name} not found in subscribers for {event_type.__name__}")
                return False

    async def shutdown(self) -> None:
        """Clean up all subscribers and set shutdown flag.

        Forces shutdown after timeout even if critical operations
        are active to prevent shutdown hangs.
        """
        # Check critical operations but force shutdown after 3 seconds
        if self.has_critical_operations():
            with self._lock:
                critical_ops = list(self._critical_operations)
            logger.warning(f"Critical operations still active during shutdown: {critical_ops}")
            logger.warning("Waiting 5 seconds for critical operations to complete...")

            try:
                # Wait up to 5 seconds for critical ops to clear
                async with asyncio.timeout(5.0):
                    while self.has_critical_operations():
                        await asyncio.sleep(0.1)
                    logger.info("All critical operations completed")
            except asyncio.TimeoutError:
                logger.warning("Critical operations did not complete in time, forcing shutdown")
                with self._lock:
                    remaining_ops = list(self._critical_operations)
                    logger.error(f"Force-clearing {len(remaining_ops)} critical operations: {remaining_ops}")
                    self._critical_operations.clear()

        with self._lock:
            self._is_shutting_down = True
            logger.debug(f"Clearing {len(self._subscribers)} subscriber lists")
            for event_type in list(self._subscribers.keys()):
                self._subscribers[event_type].clear()
            self._subscribers.clear()
        logger.debug("All event subscribers cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get current event bus statistics for monitoring and debugging.

        Returns:
            Dictionary with keys: subscribers, is_shutting_down, critical_operations.
        """
        with self._lock:
            is_shutting_down = self._is_shutting_down
            critical_ops = list(self._critical_operations)
            subscribers = {etype.__name__: len(handlers) for etype, handlers in self._subscribers.items()}

        return {
            "subscribers": subscribers,
            "is_shutting_down": is_shutting_down,
            "critical_operations": critical_ops,
        }

    def register_critical_operation(self, operation_id: str) -> None:
        """Register a critical operation to prevent event bus shutdown.

        Adds an operation ID to the critical operations set, blocking shutdown until
        the operation is unregistered. Used to protect important processes like file
        I/O or model initialization from being interrupted by shutdown.

        Args:
            operation_id: Unique string identifier for the critical operation.
        """
        with self._lock:
            self._critical_operations.add(operation_id)
        logger.debug(f"Registered critical operation: {operation_id}")

    def unregister_critical_operation(self, operation_id: str) -> None:
        """Unregister a completed critical operation to allow shutdown.

        Removes the operation ID from the critical operations set. Should be called
        when the critical operation completes successfully or fails, to unblock shutdown.

        Args:
            operation_id: Unique string identifier for the critical operation to remove.
        """
        with self._lock:
            self._critical_operations.discard(operation_id)
        logger.debug(f"Unregistered critical operation: {operation_id}")

    def has_critical_operations(self) -> bool:
        """Check if any critical operations are currently registered.

        Queries the critical operations set in a thread-safe manner to determine
        if shutdown should be blocked.

        Returns:
            True if one or more critical operations are active, False otherwise.
        """
        with self._lock:
            return len(self._critical_operations) > 0
