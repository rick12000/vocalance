from iris.app.events.base_event import BaseEvent, EventPriority
from typing import Optional, Dict, Tuple, Union, Any

# === MARK REQUEST EVENTS ===

class MarkCreateRequestEventData(BaseEvent):
    name: Optional[str] = None
    x: int
    y: int
    description: Optional[str] = None
    source: str = "unknown"
    priority: EventPriority = EventPriority.NORMAL

class MarkDeleteByNameRequestEventData(BaseEvent):
    name: str
    priority: EventPriority = EventPriority.NORMAL

class MarkDeleteAllRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

class MarkExecuteRequestEventData(BaseEvent):
    name_or_id: Union[str, int]
    priority: EventPriority = EventPriority.NORMAL

class MarkRenameRequestEventData(BaseEvent):
    old_name: str
    new_name: str
    priority: EventPriority = EventPriority.NORMAL

class MarkGetAllRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

class MarkShowNumbersRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

class MarkHideNumbersRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

class MarkVisualizeAllRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

class MarkVisualizeCancelRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

# === MARK RESPONSE EVENTS ===

class MarkCreatedEventData(BaseEvent):
    name: str
    x: int
    y: int
    priority: EventPriority = EventPriority.LOW

class MarkDeletedEventData(BaseEvent):
    name: str
    priority: EventPriority = EventPriority.LOW

class MarksChangedEventData(BaseEvent):
    """Published when the collection of marks has changed (create, delete, reset)."""
    marks: Dict[str, Dict[str, Any]] # As used in MarkService: {name: {"name": name, "x": coords[0], "y": coords[1]}}
    priority: EventPriority = EventPriority.LOW

class AllMarksClearedEventData(BaseEvent):
    """Published when all marks have been successfully cleared."""
    count: int # Number of marks cleared
    priority: EventPriority = EventPriority.LOW

class MarkVisualizationStateChangedEventData(BaseEvent):
    """Published when the mark visualization overlay is shown or hidden."""
    is_visible: bool
    priority: EventPriority = EventPriority.LOW

class MarkOperationSuccessEventData(BaseEvent):
    operation: str # e.g., "create", "delete", "execute", "reset"
    label: Optional[str] = None
    message: Optional[str] = None
    marks_data: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.LOW

class MarkOperationFailedEventData(BaseEvent):
    operation: str
    name_or_id: Optional[Union[str, int]] = None
    reason: str
    details: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.LOW

class MarkNumbersVisibilityEventData(BaseEvent):
    visible: bool
    priority: EventPriority = EventPriority.LOW

# === MARK DATA MODELS ===

class MarkData(BaseEvent):
    name: str
    x: int
    y: int
    description: str = ""
    priority: EventPriority = EventPriority.LOW

