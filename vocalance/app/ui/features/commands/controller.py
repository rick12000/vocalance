import asyncio
import logging
from typing import List

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_management_events import (
    CommandMappingsUpdatedEvent,
    CommandUiOperationEvent,
    CommandValidationErrorEvent,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtCommandsController(QtBaseController):
    commands_loaded = Signal(list)
    validation_error = Signal(str, str)
    operation_error = Signal(str)

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger("QtCommandsController"))
        self.config = config
        self.available_commands: List[AutomationCommand] = []
        self.event_bus.subscribe(CommandMappingsUpdatedEvent, self.on_command_mappings_updated)
        self.event_bus.subscribe(CommandValidationErrorEvent, self.on_command_validation_error)

    def on_view_ready(self) -> None:
        asyncio.create_task(self.event_bus.publish(CommandUiOperationEvent(op="refresh_mappings")))

    def on_command_mappings_updated(self, mappings_update: CommandMappingsUpdatedEvent) -> None:
        if mappings_update.updated_mappings is not None:
            self.available_commands = mappings_update.updated_mappings
            self.commands_loaded.emit(self.available_commands)
        else:
            self.on_view_ready()

    def on_command_validation_error(self, validation_error: CommandValidationErrorEvent) -> None:
        command_phrase = validation_error.command_phrase or "Unknown"
        self.logger.error("Command validation error for phrase '%s': %s", command_phrase, validation_error.error_message)
        self.validation_error.emit(validation_error.error_message, command_phrase)
        self.operation_error.emit(validation_error.error_message)

    def handle_add_command(self, command_phrase: str, hotkey_value: str) -> None:
        if not command_phrase:
            self.operation_error.emit("Command phrase cannot be empty")
            return
        if not hotkey_value:
            self.operation_error.emit("Hotkey value cannot be empty")
            return
        asyncio.create_task(
            self.event_bus.publish(
                CommandUiOperationEvent(
                    op="add_hotkey",
                    command_phrase=command_phrase,
                    hotkey_value=hotkey_value,
                )
            )
        )

    def handle_change_command_phrase(self, command: AutomationCommand, new_phrase: str) -> None:
        asyncio.create_task(
            self.event_bus.publish(
                CommandUiOperationEvent(op="update_phrase", old_phrase=command.command_key, new_phrase=new_phrase)
            )
        )

    def handle_delete_command(self, command: AutomationCommand) -> None:
        asyncio.create_task(
            self.event_bus.publish(CommandUiOperationEvent(op="delete_phrase", command_phrase=command.command_key))
        )

    def handle_reset_to_defaults(self) -> None:
        asyncio.create_task(self.event_bus.publish(CommandUiOperationEvent(op="reset_defaults")))

    def get_available_commands(self) -> List[AutomationCommand]:
        return self.available_commands

    def cleanup(self) -> None:
        self.event_bus.unsubscribe(CommandMappingsUpdatedEvent, self.on_command_mappings_updated)
        self.event_bus.unsubscribe(CommandValidationErrorEvent, self.on_command_validation_error)
        super().cleanup()
