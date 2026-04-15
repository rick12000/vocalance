from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from vocalance.app.events.base_event import BaseEvent

MarkUiRequestOp = Literal[
    "create",
    "delete",
    "delete_all",
    "execute",
    "set_visualization",
    "refresh_list",
    "prepare_overlay",
]

MarkUiResponseKind = Literal["create_result", "overlay_marks"]


class MarksChangedEventData(BaseEvent):
    """Broadcast when the marks collection changes."""

    marks: Dict[str, Dict[str, Any]]


class MarkVisualizationStateChangedEventData(BaseEvent):
    """Broadcast when mark overlay visibility changes."""

    is_visible: bool
    marks: Optional[Dict[str, Dict[str, Any]]] = None


class MarkUiRequestEvent(BaseEvent):
    """UI-originated mark operations; handled by ``MarkService``."""

    op: MarkUiRequestOp
    name: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    description: Optional[str] = None
    mark_name: Optional[str] = None
    identifier: Optional[str] = None
    visible: Optional[bool] = None


class MarkUiResponseEvent(BaseEvent):
    """Service-originated mark UI outcomes (create validation, overlay payload)."""

    kind: MarkUiResponseKind
    success: bool = True
    message: str = ""
    name: str = ""
    x: int = 0
    y: int = 0
    marks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class MarkData(BaseModel):
    """Data model for a single mark."""

    name: str
    x: int
    y: int
    description: str = ""
