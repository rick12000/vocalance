from typing import List, Literal, Optional

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.events.base_event import BaseEvent

CommandUiOp = Literal["add_hotkey", "update_phrase", "delete_phrase", "reset_defaults", "refresh_mappings"]


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


class CommandUiOperationEvent(BaseEvent):
    """UI-originated command CRUD; handled by ``CommandManagementService``."""

    op: CommandUiOp
    command_phrase: str = ""
    hotkey_value: str = ""
    old_phrase: str = ""
    new_phrase: str = ""
