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
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtCommandsController(QtBaseController):
    """Handles event management and business logic for the commands tab."""

    # Signals for command operations
    commands_loaded = Signal(list)  # List of AutomationCommand objects
    command_created = Signal(str)  # command_phrase
    command_updated = Signal(str, str)  # old_phrase, new_phrase
    command_deleted = Signal(str)  # command_phrase
    validation_error = Signal(str, str)  # error_message, command_phrase
    operation_error = Signal(str)

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        command_management_service,
        config: GlobalAppConfig,
        main_window,
    ):
        """Initialize commands controller.

        Args:
            event_bus: Event bus for pub/sub.
            event_loop: Asyncio event loop.
            command_management_service: Command management service instance.
            config: Global app configuration.
            main_window: Main window reference.
        """
        super().__init__(
            event_bus=event_bus,
            event_loop=event_loop,
            logger=logging.getLogger("QtCommandsController"),
        )

        self.command_service = command_management_service
        self.config = config
        self.main_window = main_window

        # Cache of available commands for display
        self.available_commands = []

        # Subscribe to command management events
        self._subscribe_to_events()

        self.logger.debug("QtCommandsController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to command-related events using exact legacy event types."""
        try:
            self.event_bus.subscribe(CommandMappingsUpdatedEvent, self._on_command_mappings_updated)
            self.event_bus.subscribe(CommandMappingsResponseEvent, self._on_command_mappings_response)
            self.event_bus.subscribe(CommandValidationErrorEvent, self._on_command_validation_error)
            self.logger.debug("Subscribed to command events (legacy types)")
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def on_view_ready(self):
        """Request initial command mappings when view is ready."""
        self._request_command_mappings()

    # --- Private Methods ---

    def _request_command_mappings(self):
        """Request current command mappings from the service."""
        event = RequestCommandMappingsEvent()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    # --- Event Handlers ---

    async def _on_command_mappings_updated(self, event):
        """Handle command mappings updated event."""
        if hasattr(event, "updated_mappings") and event.updated_mappings is not None:
            self.available_commands = event.updated_mappings
            self.commands_loaded.emit(self.available_commands)
        else:
            self._request_command_mappings()

    async def _on_command_mappings_response(self, event):
        """Handle command mappings response event."""
        if hasattr(event, "mappings"):
            self.available_commands = event.mappings
            self.commands_loaded.emit(self.available_commands)

    async def _on_command_validation_error(self, event):
        """Handle command validation error event."""
        error_message = event.error_message
        command_phrase = getattr(event, "command_phrase", "Unknown")

        self.logger.error(f"Command validation error for phrase '{command_phrase}': {error_message}")
        self.validation_error.emit(error_message, command_phrase)
        self.operation_error.emit(error_message)

    # --- Public Methods (Publish Events) ---

    def handle_add_command(self, command_phrase: str, hotkey_value: str):
        """Handle add command request from the view."""
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

        event = AddCustomCommandEvent(command=command)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)
        self.command_created.emit(command_phrase)

    def handle_change_command_phrase(self, command: AutomationCommand, new_phrase: str):
        """Handle change command phrase request from the view."""
        old_phrase = command.command_key
        event = UpdateCommandPhraseEvent(old_command_phrase=old_phrase, new_command_phrase=new_phrase)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)
        self.command_updated.emit(old_phrase, new_phrase)

    def handle_delete_command(self, command: AutomationCommand):
        """Handle delete command request from the view."""
        event = DeleteCustomCommandEvent(command=command)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)
        self.command_deleted.emit(command.command_key)

    def handle_reset_to_defaults(self):
        """Handle reset to defaults request from view."""
        event = ResetCommandsToDefaultsEvent()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    # --- Getters for View ---

    def get_available_commands(self) -> List[AutomationCommand]:
        """Get the list of available commands."""
        return self.available_commands

    def cleanup(self) -> None:
        """Clean up controller resources."""
        try:
            self.event_bus.unsubscribe(CommandMappingsUpdatedEvent, self._on_command_mappings_updated)
            self.event_bus.unsubscribe(CommandMappingsResponseEvent, self._on_command_mappings_response)
            self.event_bus.unsubscribe(CommandValidationErrorEvent, self._on_command_validation_error)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
