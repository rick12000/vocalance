from pydantic import BaseModel


class BaseEvent(BaseModel):
    """Root model for event-bus payloads.

    Subclasses define typed fields for each domain event; handlers subscribe
    using the concrete event type.
    """
