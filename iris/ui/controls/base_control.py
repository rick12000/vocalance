"""Base controller class with common patterns for all UI controls."""

import logging
import asyncio
from typing import Optional, Callable, Any
from iris.event_bus import EventBus
from iris.utils.event_utils import ThreadSafeEventPublisher, EventSubscriptionManager
from iris.ui.utils.ui_thread_utils import schedule_ui_update


class BaseController:
    """Base controller with common functionality for all UI controls."""
    
    def __init__(self, event_bus: EventBus, event_loop: asyncio.AbstractEventLoop, 
                 logger: logging.Logger, controller_name: str):
        self.event_bus = event_bus
        self.event_loop = event_loop
        self.logger = logger
        self.controller_name = controller_name
        
        # Thread-safe event handling
        self.event_publisher = ThreadSafeEventPublisher(event_bus, event_loop)
        self.subscription_manager = EventSubscriptionManager(event_bus, controller_name)
        
        # UI callback reference
        self.view_callback: Optional[Callable] = None
        
    def set_view_callback(self, callback: Callable) -> None:
        """Set the view callback for UI updates."""
        self.view_callback = callback
        self.on_view_ready()
    
    def on_view_ready(self) -> None:
        """Override this method to perform actions when view is ready."""
        pass
    
    def schedule_ui_update(self, callback: Callable, *args) -> None:
        """Schedule a UI update safely."""
        schedule_ui_update(callback, *args)
    
    def publish_event(self, event: Any) -> None:
        """Publish an event safely."""
        try:
            self.event_publisher.publish(event)
        except Exception as e:
            self.logger.error(f"{self.controller_name}: Error publishing event {type(event).__name__}: {e}")
    
    def subscribe_to_events(self, subscriptions: list) -> None:
        """Subscribe to multiple events at once."""
        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)
    
    def notify_status(self, message: str, is_error: bool = False) -> None:
        """Notify view of status updates."""
        if self.view_callback and hasattr(self.view_callback, 'on_status_update'):
            self.schedule_ui_update(self.view_callback.on_status_update, message, is_error)
    
    def show_error(self, title: str, message: str) -> None:
        """Show error message to user."""
        if self.view_callback and hasattr(self.view_callback, 'show_error_message'):
            self.schedule_ui_update(self.view_callback.show_error_message, title, message)
    
    def cleanup(self) -> None:
        """Clean up resources when controller is destroyed."""
        self.subscription_manager.unsubscribe_all()
        self.logger.info(f"{self.controller_name} cleaned up") 