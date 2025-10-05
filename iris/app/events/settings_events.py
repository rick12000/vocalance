from iris.app.events.base_event import BaseEvent, EventPriority
from typing import Dict, Any

class SettingsResponseEvent(BaseEvent):
    """Event containing current effective settings for UI and services"""
    settings: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
