import asyncio
import logging
from typing import List
from iris.app.ui.controls.base_control import BaseController
from iris.app.events.command_management_events import (
    AddCustomCommandEvent,
    UpdateCommandPhraseEvent,
    DeleteCustomCommandEvent,
    RequestCommandMappingsEvent,
    CommandMappingsUpdatedEvent,
    CommandMappingsResponseEvent,
    CommandValidationErrorEvent,
    ResetCommandsToDefaultsEvent
)
from iris.app.config.command_types import ExactMatchCommand, AutomationCommand


class CommandsController(BaseController):
    """Handles event management and business logic for the commands tab."""
    
    def __init__(self, event_bus, event_loop, app_logger):
        super().__init__(event_bus, event_loop, app_logger, "CommandsController")
        
        # Cache of available commands for display
        self.available_commands = []
        
        self.subscribe_to_events([
            (CommandMappingsUpdatedEvent, self._on_command_mappings_updated),
            (CommandMappingsResponseEvent, self._on_command_mappings_response),
            (CommandValidationErrorEvent, self._on_command_validation_error),
        ])

    def on_view_ready(self):
        """Request initial command mappings when view is ready."""
        self._request_command_mappings()

    def _request_command_mappings(self):
        """Request current command mappings from the service."""
        event = RequestCommandMappingsEvent()
        self.publish_event(event)

    def handle_add_command(self, command_phrase: str, hotkey_value: str):
        """Handle add command request from the view."""
        if not command_phrase:
            self.show_error("Error", "Command phrase cannot be empty")
            return

        if not hotkey_value:
            self.show_error("Error", "Hotkey value cannot be empty")
            return

        # Create the automation command object
        command = ExactMatchCommand(
            command_key=command_phrase,
            action_type="hotkey",
            action_value=hotkey_value,
            is_custom=True,
            short_description="Custom Command",
            long_description=f"Custom hotkey command: {hotkey_value}"
        )
        
        event = AddCustomCommandEvent(command=command)
        self.publish_event(event)

    def handle_change_command_phrase(self, command: AutomationCommand, new_phrase: str):
        """Handle change command phrase request from the view."""
        old_phrase = command.command_key
        event = UpdateCommandPhraseEvent(
            old_command_phrase=old_phrase,
            new_command_phrase=new_phrase
        )
        self.publish_event(event)

    def handle_delete_command(self, command: AutomationCommand):
        """Handle delete command request from the view."""
        event = DeleteCustomCommandEvent(command=command)
        self.publish_event(event)

    def handle_reset_to_defaults(self):
        """Handle reset to defaults request from view."""
        event = ResetCommandsToDefaultsEvent()
        self.publish_event(event)

    async def _on_command_mappings_updated(self, event):
        """Handle command mappings updated event."""
        if hasattr(event, 'updated_mappings') and event.updated_mappings is not None:
            # Use the mappings provided in the event
            self.available_commands = event.updated_mappings
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.display_commands, self.available_commands)
        else:
            # Request fresh mappings if not provided in the event
            self._request_command_mappings()

    async def _on_command_mappings_response(self, event):
        """Handle command mappings response event."""
        if hasattr(event, 'mappings'):
            self.available_commands = event.mappings
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.display_commands, self.available_commands)

    async def _on_command_validation_error(self, event):
        """Handle command validation error event."""
        error_message = event.error_message
        command_phrase = getattr(event, 'command_phrase', 'Unknown')
        
        self.logger.error(f"Command validation error for phrase '{command_phrase}': {error_message}")
        
        if self.view_callback:
            self.schedule_ui_update(
                self.view_callback.show_error_message, 
                "Validation Error", 
                error_message
            ) 