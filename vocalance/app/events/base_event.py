from enum import IntEnum

from pydantic import BaseModel


class EventPriority(IntEnum):
    """Defines the priority levels for events.

    Attributes:
        CRITICAL: Highest priority for time-critical events.
        HIGH: High priority for important events.
        NORMAL: Default priority for standard events.
        LOW: Low priority for non-urgent events.
    """

    CRITICAL = 10
    HIGH = 20
    NORMAL = 50
    LOW = 80


class BaseEvent(BaseModel):
    """Base class for all events with priority support.

    Attributes:
        priority: Priority level for event processing.
    """

    priority: EventPriority = EventPriority.NORMAL
