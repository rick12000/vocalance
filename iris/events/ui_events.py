from iris.events.base_event import BaseEvent, EventPriority
from typing import Optional

# --- UI Interaction Events ---

class GetMainWindowHandleRequest(BaseEvent):
    """Event to request the main window handle (e.g., HWND)."""
    priority: EventPriority = EventPriority.CRITICAL

class GetMainWindowHandleResponse(BaseEvent):
    """Event carrying the main window handle or an error if it couldn't be retrieved."""
    hwnd: Optional[int] = None # Using int for HWND, can be Any or a specific type if available
    error_message: Optional[str] = None
    priority: EventPriority = EventPriority.CRITICAL

class ShowToastNotificationEvent(BaseEvent):
    message: str
    duration: int = 3000 # in milliseconds
    priority: EventPriority = EventPriority.LOW
