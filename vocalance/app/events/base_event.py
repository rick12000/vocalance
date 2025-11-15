from enum import IntEnum

from pydantic import BaseModel


class EventPriority(IntEnum):
    """Defines priority levels for event processing in the event bus.

    Lower numeric values indicate higher priority for queue ordering.
    Used by EventBus to determine processing order and inter-event sleep duration.

    Attributes:
        CRITICAL: Highest priority (10) for time-critical events requiring immediate processing.
        HIGH: High priority (20) for important events that should be processed promptly.
        NORMAL: Default priority (50) for standard events with typical processing needs.
        LOW: Low priority (80) for non-urgent events that can be delayed.
    """

    CRITICAL = 10
    HIGH = 20
    NORMAL = 50
    LOW = 80


class BaseEvent(BaseModel):
    """Base class for all application events with priority-based processing support.

    Pydantic model serving as the root of the event hierarchy. All events published
    through the EventBus must inherit from this class. Provides priority attribute
    for controlling processing order in the event queue.

    Attributes:
        priority: EventPriority level determining processing order (default NORMAL).
    """

    priority: EventPriority = EventPriority.NORMAL
