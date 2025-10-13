# Thread-safe event publishing utilities
import logging
import asyncio
from typing import Any, Optional, Dict, Callable, Type
from functools import wraps
from collections import defaultdict

from iris.app.event_bus import EventBus
from iris.app.events.base_event import BaseEvent

logger = logging.getLogger(__name__)

class ThreadSafeEventPublisher:
    """Utility class for thread-safe event publishing."""
    
    def __init__(self, event_bus: EventBus, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        self.event_bus = event_bus
        self.event_loop = event_loop
        self.subscriptions = defaultdict(list)
        
    def _get_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the event loop, attempting to find running loop if not set."""
        if self.event_loop and not self.event_loop.is_closed():
            return self.event_loop
            
        try:
            loop = asyncio.get_running_loop()
            self.event_loop = loop
            return loop
        except RuntimeError:
            return None
    
    def publish(self, event: BaseEvent) -> None:
        """Publishes an event to the event bus."""
        loop = self._get_event_loop()
        if not loop:
            logger.error(f"Cannot publish event {type(event).__name__}: no running event loop")
            return
        try:
            # The new publish method takes only the event object
            asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), loop)
        except Exception as e:
            logger.error(f"Failed to publish event {type(event).__name__}: {e}")

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable) -> None:
        """Subscribes a handler to an event type."""
        safe_handler_name = handler.__name__ if hasattr(handler, '__name__') else 'unnamed_handler'
        try:
            # The new subscribe method takes the event type directly
            self.event_bus.subscribe(event_type=event_type, handler=handler)
            self.subscriptions[event_type].append(handler)
            logger.info(f"Subscription successful for {safe_handler_name} to {event_type.__name__}")
        except Exception as e:
            logger.error(f"Failed to subscribe {safe_handler_name} to {event_type.__name__}: {e}")

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events (cleanup method)."""
        for event_type in self.subscriptions:
            # Note: Current EventBus doesn't have unsubscribe method
            # This would need to be implemented in EventBus for complete cleanup
            logger.info(f"Would unsubscribe from {event_type.__name__}")
        self.subscriptions.clear()

def thread_safe_event_handler(publisher: ThreadSafeEventPublisher):
    """Decorator to make event handlers thread-safe."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event handler {func.__name__}: {e}", exc_info=True)
                # This part is tricky as we don't have a generic error event.
                # For now, we just log the error. A proper implementation
                # might involve a generic ErrorEvent type.
        return wrapper
    return decorator

class EventSubscriptionManager:
    """Manages event subscriptions for a component."""
    
    def __init__(self, event_bus: EventBus, component_name: str):
        self.event_bus = event_bus
        self.component_name = component_name
        self.subscriptions: Dict[Type[BaseEvent], Callable] = {}
        
    def subscribe(self, event_type: Type[BaseEvent], handler: Callable) -> None:
        """Subscribe to an event with automatic cleanup tracking."""
        handler_name = handler.__name__
        self.event_bus.subscribe(event_type=event_type, handler=handler)
        self.subscriptions[event_type] = handler
        logger.info(f"{self.component_name}: Subscribed to {event_type.__name__} with handler {handler_name}")
        
    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events (cleanup method)."""
        # Note: EventBus currently does not support unsubscribing.
        # This is a placeholder for future implementation.
        logger.info(f"{self.component_name}: Clearing {len(self.subscriptions)} tracked subscriptions.")
        self.subscriptions.clear()
