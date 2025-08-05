"""
Command Management Service

Event-driven service that handles command management operations by delegating to CommandStorageAdapter.
Replaces the event handling functionality that was previously in CommandStorageService.
"""

import logging
from typing import Any, Dict, List, Optional

from iris.event_bus import EventBus
from iris.config.app_config import GlobalAppConfig
from iris.events.command_management_events import (
    AddCustomCommandEvent, UpdateCommandPhraseEvent, DeleteCustomCommandEvent,
    RequestCommandMappingsEvent, ResetCommandsToDefaultsEvent,
    CommandMappingsUpdatedEvent, CommandValidationErrorEvent, CommandMappingsResponseEvent
)
from iris.services.storage.storage_adapters import StorageAdapterFactory, CommandStorageAdapter
from iris.services.protected_terms_service import ProtectedTermsService

logger = logging.getLogger(__name__)


class CommandManagementService:
    """
    Service that handles command management events and delegates to CommandStorageAdapter.
    Provides the event-driven interface that was previously in CommandStorageService.
    """

    def __init__(self, event_bus: EventBus, app_config: GlobalAppConfig, storage_factory: StorageAdapterFactory):
        self._event_bus = event_bus
        self._app_config = app_config
        self._command_adapter = storage_factory.get_command_adapter()
        self._protected_terms_service = ProtectedTermsService(app_config, storage_factory)
        
        logger.info("CommandManagementService initialized")

    def setup_subscriptions(self) -> None:
        """Set up event subscriptions for command management."""
        logger.info("Setting up CommandManagementService event subscriptions...")
        
        # Command management events
        self._event_bus.subscribe(AddCustomCommandEvent, self._handle_add_custom_command)
        self._event_bus.subscribe(UpdateCommandPhraseEvent, self._handle_update_command_phrase)
        self._event_bus.subscribe(DeleteCustomCommandEvent, self._handle_delete_custom_command)
        self._event_bus.subscribe(RequestCommandMappingsEvent, self._handle_request_command_mappings)
        self._event_bus.subscribe(ResetCommandsToDefaultsEvent, self._handle_reset_to_defaults)
        
        logger.info("CommandManagementService subscriptions set up")

    async def _validate_command_phrase(self, command_phrase: str, exclude_phrase: str = "") -> Optional[str]:
        """Validate a command phrase for conflicts using centralized protected terms service."""
        # Use centralized validation service
        validation_error = await self._protected_terms_service.validate_command_name(command_phrase, exclude_phrase)
        if validation_error:
            return validation_error
        
        # Check existing commands
        action_map = await self._command_adapter.get_action_map()
        if command_phrase.lower().strip() in action_map and command_phrase != exclude_phrase:
            return f"Command phrase '{command_phrase}' already exists"
        
        return None

    async def get_command_mappings(self) -> List[Any]:
        """Get all command mappings for UI display."""
        try:
            from iris.config.automation_command_registry import AutomationCommandRegistry
            
            mappings = []
            
            # Add custom commands
            custom_commands = await self._command_adapter.get_custom_commands()
            mappings.extend(custom_commands.values())
            
            # Add default commands
            default_commands = AutomationCommandRegistry.get_default_commands()
            
            for command_data in default_commands:
                phrase_overrides = await self._command_adapter.get_phrase_overrides()
                effective_phrase = phrase_overrides.get(command_data.command_key, command_data.command_key)
                
                if effective_phrase != command_data.command_key:
                    # Create a copy with the effective phrase
                    from iris.config.command_types import AutomationCommand
                    command_data = AutomationCommand(
                        command_key=effective_phrase,
                        action_type=command_data.action_type,
                        action_value=command_data.action_value,
                        short_description=command_data.short_description,
                        long_description=command_data.long_description,
                        is_custom=command_data.is_custom
                    )
                
                mappings.append(command_data)
            
            return mappings
        except Exception as e:
            logger.error(f"Failed to get command mappings: {e}")
            return []

    # Event handlers
    async def _handle_add_custom_command(self, event_data: AddCustomCommandEvent) -> None:
        """Handle adding a new custom command"""
        try:
            command = event_data.command
            command_phrase = command.command_key.lower().strip()
            
            if not command_phrase:
                await self._publish_validation_error("Command phrase cannot be empty", command_phrase)
                return
            
            # Check if phrase is protected using centralized service
            validation_error = await self._protected_terms_service.validate_command_name(command_phrase)
            if validation_error:
                await self._publish_validation_error(validation_error, command_phrase)
                return
            
            # Mark as custom command
            command.is_custom = True
            
            # Store the command
            success = await self._command_adapter.store_custom_command(command_phrase, command)
            
            if success:
                await self._publish_mappings_updated(True, f"Added custom command: {command_phrase}")
                logger.info(f"Successfully added custom command: {command_phrase}")
            else:
                await self._publish_validation_error("Failed to store custom command", command_phrase)
                
        except Exception as e:
            logger.error(f"Error adding custom command: {e}", exc_info=True)
            await self._publish_validation_error(f"Error adding custom command: {str(e)}", getattr(event_data.command, 'command_key', ''))

    async def _handle_update_command_phrase(self, event: UpdateCommandPhraseEvent) -> None:
        """Handle update command phrase request."""
        try:
            old_phrase = event.old_command_phrase
            new_phrase = event.new_command_phrase
            
            # Validate the new phrase
            validation_error = await self._validate_command_phrase(new_phrase, exclude_phrase=old_phrase)
            if validation_error:
                logger.warning(f"Validation failed for command phrase update '{old_phrase}' -> '{new_phrase}': {validation_error}")
                await self._publish_validation_error(validation_error, new_phrase)
                return
            
            # Check if this is a custom command or default command
            custom_commands = await self._command_adapter.get_custom_commands()
            is_custom_command = old_phrase.lower().strip() in custom_commands
            
            success = False
            if is_custom_command:
                # Update custom command
                success = await self._command_adapter.update_command_phrase(old_phrase, new_phrase, new_phrase)
                logger.debug(f"Attempted to update custom command '{old_phrase}' -> '{new_phrase}': {success}")
            else:
                # Update default command using phrase override
                from iris.config.automation_command_registry import AutomationCommandRegistry
                default_commands = AutomationCommandRegistry.get_default_commands()
                default_phrases = {cmd.command_key for cmd in default_commands}
                
                # Check if the old phrase matches a default command or is already an override
                phrase_overrides = await self._command_adapter.get_phrase_overrides()
                original_phrase = None
                
                # Find the original default phrase this override refers to
                for orig_phrase, override_phrase in phrase_overrides.items():
                    if override_phrase == old_phrase:
                        original_phrase = orig_phrase
                        break
                
                # If not found in overrides, check if it's a direct default phrase
                if original_phrase is None and old_phrase in default_phrases:
                    original_phrase = old_phrase
                
                if original_phrase:
                    success = await self._command_adapter.set_phrase_override(original_phrase, new_phrase)
                    logger.debug(f"Attempted to set phrase override for default command '{original_phrase}': '{old_phrase}' -> '{new_phrase}': {success}")
                else:
                    logger.error(f"Could not find original command for phrase '{old_phrase}'")
                    await self._publish_validation_error(f"Could not find command '{old_phrase}' to update", old_phrase)
                    return
            
            if success:
                await self._publish_mappings_updated(True, f"Updated command phrase: '{old_phrase}' -> '{new_phrase}'")
                logger.info(f"Successfully updated command phrase: '{old_phrase}' -> '{new_phrase}'")
            else:
                logger.error(f"Failed to update command phrase: '{old_phrase}' -> '{new_phrase}'")
                await self._publish_validation_error("Failed to update command phrase", new_phrase)
                
        except Exception as e:
            logger.error(f"Error updating command phrase '{event.old_command_phrase}' -> '{event.new_command_phrase}': {e}", exc_info=True)
            await self._publish_validation_error(f"Error updating command phrase: {str(e)}", event.new_command_phrase)

    async def _handle_delete_custom_command(self, event: DeleteCustomCommandEvent) -> None:
        """Handle delete custom command request."""
        success = await self._command_adapter.delete_custom_command(event.command_phrase)
        
        if success:
            await self._event_bus.publish(CommandMappingsUpdatedEvent(success=True, 
                message=f"Deleted custom command: {event.command_phrase}"))
            logger.info(f"Deleted custom command: {event.command_phrase}")
        else:
            await self._event_bus.publish(CommandValidationErrorEvent(error_message="Failed to delete custom command"))

    async def _handle_request_command_mappings(self, event: RequestCommandMappingsEvent) -> None:
        """Handle request for command mappings."""
        mappings = await self.get_command_mappings()
        
        # Publish CommandMappingsResponseEvent for UI updates
        response_event = CommandMappingsResponseEvent(mappings=mappings)
        await self._event_bus.publish(response_event)
        
        logger.debug(f"Handled command mappings request - {len(mappings)} mappings")

    async def _handle_reset_to_defaults(self, event: ResetCommandsToDefaultsEvent) -> None:
        """Handle reset to defaults request."""
        success = await self._command_adapter.reset_to_defaults()
        
        if success:
            await self._event_bus.publish(CommandMappingsUpdatedEvent(success=True, message="Reset commands to defaults"))
            logger.info("Reset commands to defaults")
        else:
            await self._event_bus.publish(CommandValidationErrorEvent(error_message="Failed to reset commands to defaults"))

    async def _publish_validation_error(self, error_message: str, command_phrase: str = "") -> None:
        """Publish validation error event."""
        await self._event_bus.publish(CommandValidationErrorEvent(error_message=error_message, command_phrase=command_phrase))

    async def _publish_mappings_updated(self, success: bool, message: str) -> None:
        """Publish mappings updated event with current command mappings."""
        try:
            # Get current mappings to include in the event
            current_mappings = await self.get_command_mappings()
            await self._event_bus.publish(CommandMappingsUpdatedEvent(
                success=success, 
                message=message,
                updated_mappings=current_mappings,
                updated_count=len(current_mappings)
            ))
        except Exception as e:
            logger.error(f"Error publishing mappings updated event: {e}")
            # Fallback to basic event without mappings
            await self._event_bus.publish(CommandMappingsUpdatedEvent(success=success, message=message))

    async def shutdown(self) -> None:
        """Shutdown service."""
        logger.info("CommandManagementService shutting down") 