import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from vocalance.app.events.base_event import BaseEvent, EventPriority


class ShowGridRequestEventData(BaseEvent):
    """Request to display the grid overlay."""

    rows: Optional[int] = None
    cols: Optional[int] = None
    priority: EventPriority = EventPriority.NORMAL


class HideGridRequestEventData(BaseEvent):
    """Request to hide the grid overlay."""

    priority: EventPriority = EventPriority.NORMAL


class ClickGridCellRequestEventData(BaseEvent):
    """Request to click a specific grid cell."""

    cell_label: str = Field(description="The label of the grid cell to click (e.g., 'A1', 'C5').")
    priority: EventPriority = EventPriority.NORMAL


class UpdateGridConfigRequestEventData(BaseEvent):
    """Request to update grid configuration parameters."""

    rows: Optional[int] = None
    cols: Optional[int] = None
    cell_width: Optional[int] = None
    cell_height: Optional[int] = None
    line_color: Optional[str] = None
    label_color: Optional[str] = None
    font_size: Optional[int] = None
    font_name: Optional[str] = None
    show_labels: Optional[bool] = None
    priority: EventPriority = EventPriority.NORMAL


class GridVisibilityChangedEventData(BaseEvent):
    """Event indicating grid visibility state has changed."""

    visible: bool
    rows: Optional[int] = None
    cols: Optional[int] = None
    priority: EventPriority = EventPriority.LOW


class GridConfigUpdatedEventData(BaseEvent):
    """Event indicating grid configuration has been updated."""

    rows: int
    cols: int
    cell_width: int
    cell_height: int
    line_color: str
    label_color: str
    font_size: int
    font_name: str
    show_labels: bool
    default_rect_count: int
    message: str = "Grid configuration updated."
    priority: EventPriority = EventPriority.LOW


class GridInteractionSuccessEventData(BaseEvent):
    """Event indicating a grid interaction succeeded."""

    operation: Literal["select_cell"]
    details: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    priority: EventPriority = EventPriority.LOW


class GridInteractionFailedEventData(BaseEvent):
    """Event indicating a grid interaction failed."""

    operation: Literal["select_cell"]
    reason: str
    cell_label: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.LOW


class RequestClickCountsForGridEventData(BaseEvent):
    """Request to calculate click counts for grid rectangles."""

    rect_definitions: List[Dict[str, Any]]
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    priority: EventPriority = EventPriority.NORMAL


class ClickCountsForGridEventData(BaseEvent):
    """Response event providing click counts for grid rectangles."""

    request_id: str
    processed_rects_with_clicks: List[Dict[str, Any]]
    priority: EventPriority = EventPriority.NORMAL
