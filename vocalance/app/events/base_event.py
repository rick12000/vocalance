from pydantic import BaseModel


class BaseEvent(BaseModel):
    """Base class for all application events.

    All events published through the EventBus must inherit from this class.
    """
