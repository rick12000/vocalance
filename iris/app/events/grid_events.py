# filepath: src/events/grid_events.py
from iris.app.events.base_event import BaseEvent, EventPriority
from pydantic import Field
from typing import Optional, Dict, Any, List
import uuid

# === GRID REQUEST EVENTS ===

class ShowGridRequestEventData(BaseEvent):
    # Optional: if grid parameters can be overridden at request time
    rows: Optional[int] = None
    cols: Optional[int] = None
    priority: EventPriority = EventPriority.NORMAL

class HideGridRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

class ClickGridCellRequestEventData(BaseEvent):
    cell_label: str = Field(description="The label of the grid cell to click (e.g., 'A1', 'C5').")
    priority: EventPriority = EventPriority.NORMAL

class UpdateGridConfigRequestEventData(BaseEvent):
    # Include fields from GridServiceConfig that can be updated
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

# GridShowRequestEvent, GridSelectRequestEvent, and GridCancelRequestEvent removed - unused in codebase

# === GRID RESPONSE EVENTS ===

class GridVisibilityChangedEventData(BaseEvent):
    visible: bool
    rows: Optional[int] = None # Optionally include current grid dimensions
    cols: Optional[int] = None
    priority: EventPriority = EventPriority.LOW

class GridConfigUpdatedEventData(BaseEvent):
    # Reports the full current configuration after an update
    rows: int
    cols: int
    cell_width: int
    cell_height: int
    line_color: str
    label_color: str
    font_size: int
    font_name: str
    show_labels: bool
    message: str = "Grid configuration updated."
    priority: EventPriority = EventPriority.LOW

class GridInteractionSuccessEventData(BaseEvent):
    operation: str
    details: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    priority: EventPriority = EventPriority.LOW

class GridInteractionFailedEventData(BaseEvent):
    operation: str
    reason: str
    cell_label: Optional[str] = None # If applicable to the operation
    details: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.LOW
class GridOperationFeedbackEvent(BaseEvent):
    success: bool
    message: Optional[str] = None
    priority: EventPriority = EventPriority.LOW

class RequestClickCountsForGridEventData(BaseEvent):
    rect_definitions: List[Dict[str, Any]]
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    priority: EventPriority = EventPriority.NORMAL

class ClickCountsForGridEventData(BaseEvent):
    request_id: str
    processed_rects_with_clicks: List[Dict[str, Any]]
    priority: EventPriority = EventPriority.NORMAL

