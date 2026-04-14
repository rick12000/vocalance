from typing import Any, Dict

from pydantic import BaseModel

from vocalance.app.events.base_event import BaseEvent


class MarksChangedEventData(BaseEvent):
    """Broadcast when the marks collection changes."""

    marks: Dict[str, Dict[str, Any]]


class MarkVisualizationStateChangedEventData(BaseEvent):
    """Broadcast when mark overlay visibility changes."""

    is_visible: bool


class MarkData(BaseModel):
    """Data model for a single mark."""

    name: str
    x: int
    y: int
    description: str = ""
