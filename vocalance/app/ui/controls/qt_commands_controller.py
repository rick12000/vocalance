import asyncio
import logging
from typing import List

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import AutomationCommand, ExactMatchCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_management_events import CommandMappingsUpdatedEvent, CommandValidationErrorEvent
from vocalance.app.services.commands.management import CommandManagementService
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtCommandsController(QtBaseController):
    """Handles event management and business logic for the commands tab."""

    commands_loaded = Signal(list)
    command_created = Signal(str)
    command_updated = Signal(str, str)
    command_deleted = Signal(str)
    validation_error = Signal(str, str)
    operation_error = Signal(str)

    def __init__(
        self,
        event_bus: EventBus,
        command_management_service: CommandManagementService,
        config: GlobalAppConfig,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtCommandsController"),
        )

        self.config = config
        self.available_commands = []
        self._command_management_service = command_management_service

        self.event_bus.subscribe(CommandMappingsUpdatedEvent, self._on_command_mappings_updated)
        self.event_bus.subscribe(CommandValidationErrorEvent, self._on_command_validation_error)

    def on_view_ready(self) -> None:
        asyncio.create_task(self._load_command_mappings())

    async def _load_command_mappings(self) -> None:
        self.available_commands = await self._command_management_service.get_command_mappings()
        self.commands_loaded.emit(self.available_commands)

    def _on_command_mappings_updated(self, mappings_update: CommandMappingsUpdatedEvent) -> None:
        if mappings_update.updated_mappings is not None:
            self.available_commands = mappings_update.updated_mappings
            self.commands_loaded.emit(self.available_commands)
        else:
            asyncio.create_task(self._load_command_mappings())

    def _on_command_validation_error(self, validation_error: CommandValidationErrorEvent) -> None:
        error_message = validation_error.error_message
        command_phrase = validation_error.command_phrase or "Unknown"
        self.logger.error("Command validation error for phrase '%s': %s", command_phrase, error_message)
        self.validation_error.emit(error_message, command_phrase)
        self.operation_error.emit(error_message)

    def handle_add_command(self, command_phrase: str, hotkey_value: str) -> None:
        if not command_phrase:
            self.operation_error.emit("Command phrase cannot be empty")
            return
        if not hotkey_value:
            self.operation_error.emit("Hotkey value cannot be empty")
            return

        command = ExactMatchCommand(
            command_key=command_phrase,
            action_type="hotkey",
            action_value=hotkey_value,
            is_custom=True,
            short_description="Custom Command",
            long_description=f"Custom hotkey command: {hotkey_value}",
            functional_group="Custom",
        )
        asyncio.create_task(self._add_command_async(command))

    async def _add_command_async(self, command: AutomationCommand) -> None:
        success, _ = await self._command_management_service.add_command(command)
        if success:
            self.command_created.emit(command.command_key)

    def handle_change_command_phrase(self, command: AutomationCommand, new_phrase: str) -> None:
        old_phrase = command.command_key
        asyncio.create_task(self._update_phrase_async(old_phrase, new_phrase))

    async def _update_phrase_async(self, old_phrase: str, new_phrase: str) -> None:
        success, _ = await self._command_management_service.update_command_phrase(old_phrase, new_phrase)
        if success:
            self.command_updated.emit(old_phrase, new_phrase)

    def handle_delete_command(self, command: AutomationCommand) -> None:
        asyncio.create_task(self._delete_command_async(command))

    async def _delete_command_async(self, command: AutomationCommand) -> None:
        success, _ = await self._command_management_service.delete_command(command)
        if success:
            self.command_deleted.emit(command.command_key)

    def handle_reset_to_defaults(self) -> None:
        asyncio.create_task(self._command_management_service.reset_to_defaults())

    def get_available_commands(self) -> List[AutomationCommand]:
        return self.available_commands

    def cleanup(self) -> None:
        try:
            self.event_bus.unsubscribe(CommandMappingsUpdatedEvent, self._on_command_mappings_updated)
            self.event_bus.unsubscribe(CommandValidationErrorEvent, self._on_command_validation_error)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
