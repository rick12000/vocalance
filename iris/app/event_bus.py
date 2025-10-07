import asyncio
import time
from collections import defaultdict
from typing import Callable, Any, Dict, List, Type
import logging
import inspect
import itertools

from iris.app.events.base_event import BaseEvent, EventPriority

logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self, high_priority_sleep=0.001, low_priority_sleep=0.01):
        self._subscribers: Dict[Type[BaseEvent], List[Callable[[BaseEvent], Any]]] = defaultdict(list)
        self._event_queue = asyncio.PriorityQueue()
        self._worker_task = None
        self._is_shutting_down = False
        self._critical_operations = set()
        self._high_priority_sleep = high_priority_sleep
        self._low_priority_sleep = low_priority_sleep
        self._counter = itertools.count()


    async def publish(self, event: BaseEvent) -> None:
        """Publish an event instance to the bus using the new single-argument API."""

        if self._is_shutting_down:
            logger.debug(f"Rejecting event {type(event).__name__} during shutdown")
            return

        if not isinstance(event, BaseEvent):
            logger.error(f"Event data must be a subclass of BaseEvent, got {type(event)}")
            return

        # Use a strictly incrementing counter for queue tie-breaking
        await self._event_queue.put((event.priority, next(self._counter), event))
        
        queue_size = self._event_queue.qsize()
        if queue_size > 50:
            logger.warning(f"Event queue size is large: {queue_size} events")
        elif queue_size > 100:
            logger.error(f"Event queue size is critical: {queue_size} events - system may be overloaded")

    def subscribe(self, event_type: Type[BaseEvent], handler: Callable[[BaseEvent], Any]) -> None:
        if not inspect.isclass(event_type) or not issubclass(event_type, BaseEvent):
            logger.error(f"Can only subscribe to subclasses of BaseEvent, got {event_type}")
            return
            
        if not callable(handler):
            logger.error(f"Handler must be callable, got {type(handler)}")
            return
            
        logger.info(f"Subscribing handler {handler.__name__ if hasattr(handler, '__name__') else handler} to event type: {event_type.__name__}")
        self._subscribers[event_type].append(handler)

    async def _process_events(self):
        logger.info("Event processing worker started")
        while not self._is_shutting_down:
            try:
                priority, _, event = await self._event_queue.get()
                
                event_type = type(event)
                handler_found = False

                # Iterate over subscribed event types
                for subscribed_type, handlers in self._subscribers.items():
                    # Check if the event is an instance of the subscribed type (supports inheritance)
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
                                if handler_time > 0.1: # 100ms threshold for slow handlers
                                    logger.warning(f"Slow handler {handler.__name__ if hasattr(handler, '__name__') else handler} for event '{event_type.__name__}': {handler_time:.4f}s")
                            
                            except Exception as e:
                                handler_name = handler.__name__ if hasattr(handler, '__name__') else str(handler)
                                logger.error(f"Error in handler {handler_name} for event '{event_type.__name__}': {e}", exc_info=True)
                
                if not handler_found:
                    logger.warning(f"No handlers registered for event '{event_type.__name__}' or its parent types.")

                self._event_queue.task_done()
                
                # Yield control to allow other coroutines to run, sleeping longer for lower priority events
                sleep_duration = self._low_priority_sleep if priority >= EventPriority.NORMAL else self._high_priority_sleep
                await asyncio.sleep(sleep_duration)

            except asyncio.CancelledError:
                logger.info("Event processing worker cancelled")
                break
            except Exception as e:
                logger.critical(f"Fatal error in event processing worker: {e}", exc_info=True)
                if not self._is_shutting_down:
                    await asyncio.sleep(1) # a brief pause before the loop continues

    async def start_worker(self):
        if self._is_shutting_down:
            logger.info("Not starting worker during shutdown")
            return
            
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_events())
            logger.info("Event bus worker started")
        else:
            logger.info("Event bus worker already running")

    async def stop_worker(self):
        if self.has_critical_operations():
            logger.warning(f"Cannot shutdown event bus - critical operations active: {list(self._critical_operations)}")
            return
            
        self._is_shutting_down = True
        
        try:
            await asyncio.wait_for(self._event_queue.join(), timeout=2.0)
            logger.info("Event queue successfully drained before shutdown")
        except asyncio.TimeoutError:
            remaining = self._event_queue.qsize()
            logger.warning(f"Could not process all events before shutdown. {remaining} events discarded.")
        
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                logger.info("Event bus worker successfully stopped")
        
        # This prevents services from being kept alive by event bus subscriptions
        logger.info(f"Clearing {len(self._subscribers)} subscriber lists to prevent memory leaks")
        for event_type in list(self._subscribers.keys()):
            self._subscribers[event_type].clear()
        self._subscribers.clear()
        logger.info("All event subscribers cleared")
        
        logger.info("Event bus shutdown complete.")

    def get_stats(self):
        worker_status = "running"
        if self._worker_task is None:
            worker_status = "not_created"
        elif self._worker_task.done():
            worker_status = "stopped"
            if self._worker_task.cancelled():
                worker_status = "cancelled"
            elif self._worker_task.exception():
                worker_status = f"error: {self._worker_task.exception()}"
        
        return {
            "queue_size": self._event_queue.qsize(),
            "subscribers": {event.__name__: len(handlers) for event, handlers in self._subscribers.items()},
            "worker_status": worker_status,
            "is_shutting_down": self._is_shutting_down,
            "critical_operations": list(self._critical_operations)
        }

    def register_critical_operation(self, operation_id: str) -> None:
        self._critical_operations.add(operation_id)
        logger.info(f"Registered critical operation: {operation_id}")
    
    def unregister_critical_operation(self, operation_id: str) -> None:
        self._critical_operations.discard(operation_id)
        logger.info(f"Unregistered critical operation: {operation_id}")
    
    def has_critical_operations(self) -> bool:
        return len(self._critical_operations) > 0
