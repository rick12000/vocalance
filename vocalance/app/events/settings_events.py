"""Settings-related events."""

from typing import Any, Dict

from vocalance.app.events.base_event import BaseEvent, EventPriority


class SettingsLoadedEvent(BaseEvent):
    """Event fired when settings are loaded."""

    settings: Dict[str, Any]
    priority: EventPriority = EventPriority.LOW


class SettingUpdatedEvent(BaseEvent):
    """Event fired when a single setting is updated."""

    key: str
    value: Any
    priority: EventPriority = EventPriority.LOW


class SettingsUpdatedEvent(BaseEvent):
    """Event fired when settings are updated."""

    settings: Dict[str, Any]
    priority: EventPriority = EventPriority.LOW


class SettingChangedEvent(BaseEvent):
    """Event fired when a setting changes."""

    key: str
    old_value: Any
    new_value: Any
    priority: EventPriority = EventPriority.LOW


class SettingsResetEvent(BaseEvent):
    """Event fired when settings are reset."""

    priority: EventPriority = EventPriority.NORMAL


class ResetCompleteEvent(BaseEvent):
    """Event fired when reset is complete."""

    priority: EventPriority = EventPriority.LOW


class RequestSettingsEvent(BaseEvent):
    """Request to retrieve current settings."""

    priority: EventPriority = EventPriority.NORMAL


class UpdateSettingEvent(BaseEvent):
    """Request to update a setting."""

    key: str
    value: Any
    priority: EventPriority = EventPriority.NORMAL
