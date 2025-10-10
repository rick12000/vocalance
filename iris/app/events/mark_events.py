from iris.app.events.base_event import BaseEvent, EventPriority
from typing import Optional, Dict, Union, Any

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

class MarkGetAllRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

class MarkVisualizeAllRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

class MarkVisualizeCancelRequestEventData(BaseEvent):
    priority: EventPriority = EventPriority.NORMAL

class MarkCreatedEventData(BaseEvent):
    name: str
    x: int
    y: int
    priority: EventPriority = EventPriority.LOW

class MarkDeletedEventData(BaseEvent):
    name: str
    priority: EventPriority = EventPriority.LOW

class MarksChangedEventData(BaseEvent):
    marks: Dict[str, Dict[str, Any]]
    priority: EventPriority = EventPriority.LOW

class AllMarksClearedEventData(BaseEvent):
    count: int
    priority: EventPriority = EventPriority.LOW

class MarkVisualizationStateChangedEventData(BaseEvent):
    is_visible: bool
    priority: EventPriority = EventPriority.LOW

class MarkOperationFailedEventData(BaseEvent):
    operation: str
    name_or_id: Optional[Union[str, int]] = None
    reason: str
    details: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.LOW

class MarkOperationSuccessEventData(BaseEvent):
    operation: str
    label: Optional[str] = None
    message: Optional[str] = None
    marks_data: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.LOW

class MarkData(BaseEvent):
    name: str
    x: int
    y: int
    description: str = ""
    priority: EventPriority = EventPriority.LOW
