from typing import Any, Dict, Literal, Optional, Union

from vocalance.app.events.base_event import BaseEvent, EventPriority


class MarkCreateRequestEventData(BaseEvent):
    """Request to create a new mark at specified coordinates."""

    name: Optional[str] = None
    x: int
    y: int
    description: Optional[str] = None
    source: str = "unknown"
    priority: EventPriority = EventPriority.NORMAL


class MarkDeleteByNameRequestEventData(BaseEvent):
    """Request to delete a mark by name."""

    name: str
    priority: EventPriority = EventPriority.NORMAL


class MarkDeleteAllRequestEventData(BaseEvent):
    """Request to delete all marks."""

    priority: EventPriority = EventPriority.NORMAL


class MarkExecuteRequestEventData(BaseEvent):
    """Request to execute/click a mark by name or ID."""

    name_or_id: Union[str, int]
    priority: EventPriority = EventPriority.NORMAL


class MarkGetAllRequestEventData(BaseEvent):
    """Request to retrieve all marks."""

    priority: EventPriority = EventPriority.NORMAL


class MarkVisualizeAllRequestEventData(BaseEvent):
    """Request to visualize all marks on screen."""

    priority: EventPriority = EventPriority.NORMAL


class MarkVisualizeCancelRequestEventData(BaseEvent):
    """Request to cancel mark visualization."""

    priority: EventPriority = EventPriority.NORMAL


class MarkCreatedEventData(BaseEvent):
    """Event indicating a mark has been successfully created."""

    name: str
    x: int
    y: int
    priority: EventPriority = EventPriority.LOW


class MarkDeletedEventData(BaseEvent):
    """Event indicating a mark has been deleted."""

    name: str
    priority: EventPriority = EventPriority.LOW


class MarksChangedEventData(BaseEvent):
    """Event indicating marks collection has been modified."""

    marks: Dict[str, Dict[str, Any]]
    priority: EventPriority = EventPriority.LOW


class AllMarksClearedEventData(BaseEvent):
    """Event indicating all marks have been cleared."""

    count: int
    priority: EventPriority = EventPriority.LOW


class MarkVisualizationStateChangedEventData(BaseEvent):
    """Event indicating mark visualization state has changed."""

    is_visible: bool
    priority: EventPriority = EventPriority.LOW


class MarkOperationFailedEventData(BaseEvent):
    """Event indicating a mark operation has failed."""

    operation: Literal["create", "execute", "delete", "visualize", "reset", "visualize_cancel"]
    name_or_id: Optional[Union[str, int]] = None
    reason: str
    details: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.LOW


class MarkOperationSuccessEventData(BaseEvent):
    """Event indicating a mark operation has succeeded."""

    operation: Literal["create", "execute", "delete", "visualize", "reset", "visualize_cancel"]
    label: Optional[str] = None
    message: Optional[str] = None
    marks_data: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.LOW


class MarkData(BaseEvent):
    """Data model for a single mark."""

    name: str
    x: int
    y: int
    description: str = ""
    priority: EventPriority = EventPriority.LOW
