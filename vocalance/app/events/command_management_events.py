from typing import List, Optional

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.events.base_event import BaseEvent


class CommandMappingsUpdatedEvent(BaseEvent):
    """Broadcast when command mappings change (add/update/delete/reset)."""

    success: bool
    message: str = ""
    updated_count: Optional[int] = None
    updated_mappings: Optional[List[AutomationCommand]] = None


class CommandValidationErrorEvent(BaseEvent):
    """Broadcast when a command mutation fails validation."""

    error_message: str
    command_phrase: str = ""
    action_value: str = ""
