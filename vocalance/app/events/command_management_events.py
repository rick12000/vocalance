from typing import List, Optional

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.events.base_event import BaseEvent, EventPriority


class AddCustomCommandEvent(BaseEvent):
    """Event to add a new custom automation command"""

    command: AutomationCommand
    priority: EventPriority = EventPriority.NORMAL


class UpdateCommandPhraseEvent(BaseEvent):
    """Event to update an existing command phrase"""

    old_command_phrase: str
    new_command_phrase: str
    priority: EventPriority = EventPriority.NORMAL


class DeleteCustomCommandEvent(BaseEvent):
    """Event to delete a custom command"""

    command: AutomationCommand
    priority: EventPriority = EventPriority.NORMAL


class CommandMappingsUpdatedEvent(BaseEvent):
    """Event fired when command mappings are updated"""

    success: bool
    message: str = ""
    updated_count: Optional[int] = None
    updated_mappings: Optional[List[AutomationCommand]] = None
    priority: EventPriority = EventPriority.LOW


class RequestCommandMappingsEvent(BaseEvent):
    """Event to request current command mappings"""

    priority: EventPriority = EventPriority.NORMAL


class CommandMappingsResponseEvent(BaseEvent):
    """Response event with current command mappings"""

    mappings: List[AutomationCommand]
    priority: EventPriority = EventPriority.LOW


class CommandValidationErrorEvent(BaseEvent):
    """Event fired when command validation fails"""

    error_message: str
    command_phrase: str = ""
    action_value: str = ""
    priority: EventPriority = EventPriority.NORMAL


class ResetCommandsToDefaultsEvent(BaseEvent):
    """Event to reset all commands to their default state"""

    priority: EventPriority = EventPriority.NORMAL
