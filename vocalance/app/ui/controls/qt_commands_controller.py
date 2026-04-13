import asyncio
import logging
from typing import List

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import AutomationCommand, ExactMatchCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_management_events import (
    AddCustomCommandEvent,
    CommandMappingsResponseEvent,
    CommandMappingsUpdatedEvent,
    CommandValidationErrorEvent,
    DeleteCustomCommandEvent,
    RequestCommandMappingsEvent,
    ResetCommandsToDefaultsEvent,
    UpdateCommandPhraseEvent,
)
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
        """Initialize commands controller.

        Args:
            event_bus: Event bus for pub/sub.
            command_management_service: Command management service instance.
            config: Global app configuration.
        """
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtCommandsController"),
        )

        self.config = config
        self.available_commands = []

        self._subscribe_to_events()
        self.logger.debug("QtCommandsController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to command management events."""
        try:
            self.event_bus.subscribe(CommandMappingsUpdatedEvent, self._on_command_mappings_updated)
            self.event_bus.subscribe(CommandMappingsResponseEvent, self._on_command_mappings_response)
            self.event_bus.subscribe(CommandValidationErrorEvent, self._on_command_validation_error)
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def on_view_ready(self) -> None:
        """Request initial command mappings when view is ready."""
        self._request_command_mappings()

    def _request_command_mappings(self) -> None:
        """Publish a request for current command mappings."""
        asyncio.ensure_future(self.event_bus.publish(RequestCommandMappingsEvent()))

    def _on_command_mappings_updated(self, mappings_update: CommandMappingsUpdatedEvent) -> None:
        """Handle command mappings updated event."""
        if mappings_update.updated_mappings is not None:
            self.available_commands = mappings_update.updated_mappings
            self.commands_loaded.emit(self.available_commands)
        else:
            self._request_command_mappings()

    def _on_command_mappings_response(self, response: CommandMappingsResponseEvent) -> None:
        """Handle command mappings response event."""
        self.available_commands = response.mappings
        self.commands_loaded.emit(self.available_commands)

    def _on_command_validation_error(self, validation_error: CommandValidationErrorEvent) -> None:
        """Handle command validation error event."""
        error_message = validation_error.error_message
        command_phrase = validation_error.command_phrase or "Unknown"
        self.logger.error(f"Command validation error for phrase '{command_phrase}': {error_message}")
        self.validation_error.emit(error_message, command_phrase)
        self.operation_error.emit(error_message)

    def handle_add_command(self, command_phrase: str, hotkey_value: str):
        """Publish an add-command event from the view.

        Args:
            command_phrase: Voice phrase to trigger the command.
            hotkey_value: Hotkey action value for the command.
        """
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
        asyncio.ensure_future(self.event_bus.publish(AddCustomCommandEvent(command=command)))
        self.command_created.emit(command_phrase)

    def handle_change_command_phrase(self, command: AutomationCommand, new_phrase: str):
        """Publish an update-phrase event from the view.

        Args:
            command: Existing command to update.
            new_phrase: New voice phrase for the command.
        """
        old_phrase = command.command_key
        asyncio.ensure_future(
            self.event_bus.publish(UpdateCommandPhraseEvent(old_command_phrase=old_phrase, new_command_phrase=new_phrase))
        )
        self.command_updated.emit(old_phrase, new_phrase)

    def handle_delete_command(self, command: AutomationCommand):
        """Publish a delete-command event from the view.

        Args:
            command: Command to delete.
        """
        asyncio.ensure_future(self.event_bus.publish(DeleteCustomCommandEvent(command=command)))
        self.command_deleted.emit(command.command_key)

    def handle_reset_to_defaults(self):
        """Publish a reset-to-defaults event from the view."""
        asyncio.ensure_future(self.event_bus.publish(ResetCommandsToDefaultsEvent()))

    def get_available_commands(self) -> List[AutomationCommand]:
        """Return the cached list of available commands."""
        return self.available_commands

    def cleanup(self) -> None:
        """Unsubscribe from all events and release resources."""
        try:
            self.event_bus.unsubscribe(CommandMappingsUpdatedEvent, self._on_command_mappings_updated)
            self.event_bus.unsubscribe(CommandMappingsResponseEvent, self._on_command_mappings_response)
            self.event_bus.unsubscribe(CommandValidationErrorEvent, self._on_command_validation_error)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
