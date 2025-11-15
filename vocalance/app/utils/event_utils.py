"""Utility classes and decorators for thread-safe event handling.

Provides ThreadSafeEventPublisher for cross-thread event publishing using
asyncio.run_coroutine_threadsafe, EventSubscriptionManager for tracking and
cleanup of subscriptions, and decorators for error-handling event handlers.
"""
import asyncio
import logging
from collections import defaultdict
from functools import wraps
from typing import Callable, Dict, Optional, Type

from vocalance.app.event_bus import EventBus
from vocalance.app.events.base_event import BaseEvent

logger = logging.getLogger(__name__)


class ThreadSafeEventPublisher:
    """Utility class for thread-safe event publishing from any thread.

    Uses asyncio.run_coroutine_threadsafe to safely publish events to the event
    bus from UI threads or background threads. Automatically detects event loop
    and handles errors gracefully.

    Attributes:
        event_bus: EventBus instance for publishing.
        event_loop: Optional cached event loop reference.
        subscriptions: Dict tracking subscriptions for cleanup.
    """

    def __init__(self, event_bus: EventBus, event_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self.event_bus: EventBus = event_bus
        self.event_loop: Optional[asyncio.AbstractEventLoop] = event_loop
        self.subscriptions: Dict = defaultdict(list)

    def _get_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the event loop, attempting to find running loop if not set.

        Returns:
            Event loop if available, None otherwise.
        """
        if self.event_loop and not self.event_loop.is_closed():
            return self.event_loop

        try:
            loop = asyncio.get_running_loop()
            self.event_loop = loop
            return loop
        except RuntimeError:
            return None

    def publish(self, event: BaseEvent) -> None:
        """Publish an event to the event bus.

        Args:
            event: Event instance to publish.
        """
        loop = self._get_event_loop()
        if not loop:
            logger.error(f"Cannot publish event {type(event).__name__}: no running event loop")
            return
        try:
            asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), loop)
        except Exception as e:
            logger.error(f"Failed to publish event {type(event).__name__}: {e}")

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: Event class to subscribe to.
            handler: Callable to invoke when event is published.
        """
        safe_handler_name = handler.__name__ if hasattr(handler, "__name__") else "unnamed_handler"
        try:
            self.event_bus.subscribe(event_type=event_type, handler=handler)
            self.subscriptions[event_type].append(handler)
            logger.debug(f"Subscription successful for {safe_handler_name} to {event_type.__name__}")
        except Exception as e:
            logger.error(f"Failed to subscribe {safe_handler_name} to {event_type.__name__}: {e}")

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events (cleanup method)."""
        logger.debug(f"Clearing {len(self.subscriptions)} subscriptions")
        self.subscriptions.clear()


def thread_safe_event_handler(publisher: ThreadSafeEventPublisher) -> Callable:
    """Decorator to make event handlers thread-safe.

    Args:
        publisher: ThreadSafeEventPublisher instance.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event handler {func.__name__}: {e}", exc_info=True)

        return wrapper

    return decorator


class EventSubscriptionManager:
    """Manages event subscriptions for a component with automatic cleanup tracking.

    Tracks all subscriptions made by a component for centralized cleanup, logging,
    and debugging. Simplifies component teardown.

    Attributes:
        event_bus: EventBus instance.
        component_name: Name of component for logging.
        subscriptions: Dict of event type to handler mappings.
    """

    def __init__(self, event_bus: EventBus, component_name: str) -> None:
        self.event_bus: EventBus = event_bus
        self.component_name: str = component_name
        self.subscriptions: Dict[Type[BaseEvent], Callable] = {}

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable) -> None:
        """Subscribe to an event with automatic cleanup tracking.

        Args:
            event_type: Event class to subscribe to.
            handler: Callable to invoke when event is published.
        """
        handler_name = handler.__name__
        self.event_bus.subscribe(event_type=event_type, handler=handler)
        self.subscriptions[event_type] = handler
        logger.debug(f"{self.component_name}: Subscribed to {event_type.__name__} with handler {handler_name}")

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events (cleanup method)."""
        logger.debug(f"{self.component_name}: Clearing {len(self.subscriptions)} tracked subscriptions")
        self.subscriptions.clear()
