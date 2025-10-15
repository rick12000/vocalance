"""
Command Management Service

Event-driven service that handles command management operations with integrated protected terms validation.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from iris.app.event_bus import EventBus
from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.automation_command_registry import AutomationCommandRegistry
from iris.app.config.command_types import AutomationCommand
from iris.app.events.command_management_events import (
    AddCustomCommandEvent, UpdateCommandPhraseEvent, DeleteCustomCommandEvent,
    RequestCommandMappingsEvent, ResetCommandsToDefaultsEvent,
    CommandMappingsUpdatedEvent, CommandValidationErrorEvent, CommandMappingsResponseEvent
)
from iris.app.services.storage.storage_service import StorageService
from iris.app.services.storage.storage_models import CommandsData, MarksData, SoundMappingsData

logger = logging.getLogger(__name__)


class CommandManagementService:
    """
    Service that handles command management events with integrated protected terms validation.
    """

    def __init__(self, event_bus: EventBus, app_config: GlobalAppConfig, storage: StorageService):
        self._event_bus = event_bus
        self._app_config = app_config
        self._storage = storage
        
        logger.info("CommandManagementService initialized")

    def setup_subscriptions(self) -> None:
        """Set up event subscriptions for command management."""
        logger.info("Setting up CommandManagementService event subscriptions...")
        
        self._event_bus.subscribe(event_type=AddCustomCommandEvent, handler=self._handle_add_custom_command)
        self._event_bus.subscribe(event_type=UpdateCommandPhraseEvent, handler=self._handle_update_command_phrase)
        self._event_bus.subscribe(event_type=DeleteCustomCommandEvent, handler=self._handle_delete_custom_command)
        self._event_bus.subscribe(event_type=RequestCommandMappingsEvent, handler=self._handle_request_command_mappings)
        self._event_bus.subscribe(event_type=ResetCommandsToDefaultsEvent, handler=self._handle_reset_to_defaults)
        
        logger.info("CommandManagementService subscriptions set up")

    async def _get_protected_terms(self) -> Set[str]:
        """
        Get all protected terms that cannot be used as custom command names.
        Includes automation commands, system triggers, live mark names, and live sound names.
        """
        protected = set()
        
        protected.update(phrase.lower().strip() for phrase in AutomationCommandRegistry.get_protected_phrases())
        protected.add(self._app_config.grid.show_grid_phrase.lower().strip())
        
        mark_triggers = self._app_config.mark.triggers
        protected.add(mark_triggers.create_mark.lower().strip())
        protected.add(mark_triggers.delete_mark.lower().strip())
        protected.update(phrase.lower().strip() for phrase in mark_triggers.visualize_marks)
        protected.update(phrase.lower().strip() for phrase in mark_triggers.reset_marks)
        
        dictation = self._app_config.dictation
        protected.add(dictation.start_trigger.lower().strip())
        protected.add(dictation.stop_trigger.lower().strip())
        protected.add(dictation.type_trigger.lower().strip())
        protected.add(dictation.smart_start_trigger.lower().strip())
        
        try:
            marks_data = await self._storage.read(model_type=MarksData)
            protected.update(name.lower().strip() for name in marks_data.marks.keys())
        except Exception as e:
            logger.warning(f"Could not fetch mark names for protection: {e}")
        
        try:
            sound_data = await self._storage.read(model_type=SoundMappingsData)
            protected.update(sound.lower().strip() for sound in sound_data.mappings.keys())
        except Exception as e:
            logger.warning(f"Could not fetch sound names for protection: {e}")
        
        return protected

    async def _get_action_map(self) -> Dict[str, Any]:
        """Get action map for command lookup."""
        action_map = {}
        
        # Load custom commands (already AutomationCommand objects)
        commands_data = await self._storage.read(model_type=CommandsData)
        for normalized_phrase, command_obj in commands_data.custom_commands.items():
            action_map[normalized_phrase] = command_obj
        
        # Load default commands
        default_commands = AutomationCommandRegistry.get_default_commands()
        
        for command_data in default_commands:
            normalized_phrase = command_data.command_key.lower().strip()
            
            if normalized_phrase not in action_map:
                # Apply any phrase overrides
                effective_phrase = commands_data.phrase_overrides.get(command_data.command_key, command_data.command_key)
                
                if effective_phrase != command_data.command_key:
                    command_data = AutomationCommand(
                        command_key=effective_phrase,
                        action_type=command_data.action_type,
                        action_value=command_data.action_value,
                        short_description=command_data.short_description,
                        long_description=command_data.long_description,
                        is_custom=command_data.is_custom
                    )
                    normalized_phrase = effective_phrase.lower().strip()
                
                action_map[normalized_phrase] = command_data
        
        return action_map
    
    async def _validate_command_phrase(self, command_phrase: str, exclude_phrase: str = "") -> Optional[str]:
        """Validate a command phrase for conflicts."""
        if not command_phrase or not command_phrase.strip():
            return "Command phrase cannot be empty"
        
        normalized_phrase = command_phrase.lower().strip()
        normalized_exclude = exclude_phrase.lower().strip() if exclude_phrase else ""
        
        if normalized_phrase == normalized_exclude:
            return None
        
        protected_terms = await self._get_protected_terms()
        if normalized_phrase in protected_terms:
            return f"'{command_phrase}' is a protected term and cannot be used"
        
        action_map = await self._get_action_map()
        if normalized_phrase in action_map:
            return f"Command phrase '{command_phrase}' already exists"
        
        return None

    async def get_command_mappings(self) -> List[Any]:
        """Get all command mappings for UI display."""
        try:
            mappings = []
            commands_data = await self._storage.read(model_type=CommandsData)
            
            # Add custom commands (already AutomationCommand objects)
            mappings.extend(commands_data.custom_commands.values())
            
            # Add default commands with overrides
            default_commands = AutomationCommandRegistry.get_default_commands()
            
            for command_data in default_commands:
                effective_phrase = commands_data.phrase_overrides.get(command_data.command_key, command_data.command_key)
                
                if effective_phrase != command_data.command_key:
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

    async def _handle_add_custom_command(self, event_data: AddCustomCommandEvent) -> None:
        """Handle adding a new custom command"""
        try:
            command = event_data.command
            command_phrase = command.command_key.lower().strip()
            
            validation_error = await self._validate_command_phrase(command_phrase)
            if validation_error:
                await self._publish_validation_error(validation_error, command_phrase)
                return
            
            command.is_custom = True
            
            commands_data = await self._storage.read(model_type=CommandsData)
            commands_data.custom_commands[command_phrase] = command
            
            success = await self._storage.write(data=commands_data)
            
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
            
            validation_error = await self._validate_command_phrase(new_phrase, exclude_phrase=old_phrase)
            if validation_error:
                logger.warning(f"Validation failed for command phrase update '{old_phrase}' -> '{new_phrase}': {validation_error}")
                await self._publish_validation_error(validation_error, new_phrase)
                return
            
            commands_data = await self._storage.read(model_type=CommandsData)
            is_custom_command = old_phrase.lower().strip() in commands_data.custom_commands
            
            success = False
            if is_custom_command:
                command_obj = commands_data.custom_commands[old_phrase]
                command_obj.command_key = new_phrase
                del commands_data.custom_commands[old_phrase]
                commands_data.custom_commands[new_phrase] = command_obj
                success = await self._storage.write(data=commands_data)
            else:
                default_commands = AutomationCommandRegistry.get_default_commands()
                default_phrases = {cmd.command_key for cmd in default_commands}
                
                original_phrase = None
                
                for orig_phrase, override_phrase in commands_data.phrase_overrides.items():
                    if override_phrase == old_phrase:
                        original_phrase = orig_phrase
                        break
                
                if original_phrase is None and old_phrase in default_phrases:
                    original_phrase = old_phrase
                
                if original_phrase:
                    commands_data.phrase_overrides[original_phrase] = new_phrase
                    success = await self._storage.write(data=commands_data)
                else:
                    logger.error(f"Could not find original command for phrase '{old_phrase}'")
                    await self._publish_validation_error(f"Could not find command '{old_phrase}' to update", old_phrase)
                    return
            
            if success:
                await self._publish_mappings_updated(True, f"Updated command phrase: '{old_phrase}' -> '{new_phrase}'")
                logger.info(f"Successfully updated command phrase: '{old_phrase}' -> '{new_phrase}'")
            else:
                await self._publish_validation_error("Failed to update command phrase", new_phrase)
                
        except Exception as e:
            logger.error(f"Error updating command phrase '{event.old_command_phrase}' -> '{event.new_command_phrase}': {e}", exc_info=True)
            await self._publish_validation_error(f"Error updating command phrase: {str(e)}", event.new_command_phrase)

    async def _handle_delete_custom_command(self, event: DeleteCustomCommandEvent) -> None:
        """Handle delete custom command request."""
        command_phrase = event.command.command_key.lower().strip()
        commands_data = await self._storage.read(model_type=CommandsData)
        if command_phrase in commands_data.custom_commands:
            del commands_data.custom_commands[command_phrase]
            success = await self._storage.write(data=commands_data)
        else:
            success = True

        if success:
            await self._publish_mappings_updated(True, f"Deleted custom command: {command_phrase}")
            logger.info(f"Deleted custom command: {command_phrase}")
        else:
            await self._publish_validation_error("Failed to delete custom command")

    async def _handle_request_command_mappings(self, event: RequestCommandMappingsEvent) -> None:
        """Handle request for command mappings."""
        mappings = await self.get_command_mappings()
        await self._event_bus.publish(CommandMappingsResponseEvent(mappings=mappings))
        logger.debug(f"Handled command mappings request - {len(mappings)} mappings")

    async def _handle_reset_to_defaults(self, event: ResetCommandsToDefaultsEvent) -> None:
        """Handle reset to defaults request."""
        commands_data = CommandsData()
        success = await self._storage.write(data=commands_data)
        
        if success:
            await self._publish_mappings_updated(True, "Reset commands to defaults")
            logger.info("Reset commands to defaults")
        else:
            await self._publish_validation_error("Failed to reset commands to defaults")

    async def _publish_validation_error(self, error_message: str, command_phrase: str = "") -> None:
        """Publish validation error event."""
        await self._event_bus.publish(CommandValidationErrorEvent(
            error_message=error_message, 
            command_phrase=command_phrase
        ))

    async def _publish_mappings_updated(self, success: bool, message: str) -> None:
        """Publish mappings updated event with current command mappings."""
        try:
            current_mappings = await self.get_command_mappings()
            await self._event_bus.publish(CommandMappingsUpdatedEvent(
                success=success, 
                message=message,
                updated_mappings=current_mappings,
                updated_count=len(current_mappings)
            ))
        except Exception as e:
            logger.error(f"Error publishing mappings updated event: {e}")
            await self._event_bus.publish(CommandMappingsUpdatedEvent(success=success, message=message)) 