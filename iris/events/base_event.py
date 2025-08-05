from pydantic import BaseModel
from enum import IntEnum

class EventPriority(IntEnum):
    """Defines the priority levels for events."""
    CRITICAL = 10
    HIGH = 20
    NORMAL = 50
    LOW = 80

class BaseEvent(BaseModel):
    """Base class for all events, including a priority."""
    priority: EventPriority = EventPriority.NORMAL 