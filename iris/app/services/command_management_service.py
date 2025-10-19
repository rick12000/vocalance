import logging
from typing import List, Optional

from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.automation_command_registry import AutomationCommandRegistry
from iris.app.config.command_types import AutomationCommand
from iris.app.event_bus import EventBus
from iris.app.events.command_management_events import (
    AddCustomCommandEvent,
    CommandMappingsResponseEvent,
    CommandMappingsUpdatedEvent,
    CommandValidationErrorEvent,
    DeleteCustomCommandEvent,
    RequestCommandMappingsEvent,
    ResetCommandsToDefaultsEvent,
    UpdateCommandPhraseEvent,
)
from iris.app.services.command_action_map_provider import CommandActionMapProvider
from iris.app.services.protected_terms_validator import ProtectedTermsValidator
from iris.app.services.storage.storage_models import CommandsData
from iris.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class CommandManagementService:
    """Service for managing custom automation commands with protected term validation.

    Handles adding, updating, and deleting custom commands while validating against
    protected terms (system commands, marks, sounds) to prevent conflicts.
    """

    def __init__(
        self,
        event_bus: EventBus,
        app_config: GlobalAppConfig,
        storage: StorageService,
        protected_terms_validator: ProtectedTermsValidator,
        action_map_provider: CommandActionMapProvider,
    ) -> None:
        self._event_bus = event_bus
        self._app_config = app_config
        self._storage = storage
        self._protected_terms_validator = protected_terms_validator
        self._action_map_provider = action_map_provider

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

    async def _validate_command_phrase(self, command_phrase: str, exclude_phrase: str = "") -> Optional[str]:
        """Validate command phrase against protected terms and existing commands.

        Args:
            command_phrase: The command phrase to validate
            exclude_phrase: Phrase to exclude from conflict check (for updates)

        Returns:
            Error message string if invalid, None if valid
        """
        is_valid, error_msg = await self._protected_terms_validator.validate_term(term=command_phrase, exclude_term=exclude_phrase)

        if not is_valid:
            return error_msg

        normalized_phrase = command_phrase.lower().strip()
        normalized_exclude = exclude_phrase.lower().strip() if exclude_phrase else ""

        if normalized_phrase == normalized_exclude:
            return None

        action_map = await self._action_map_provider.get_action_map()
        if normalized_phrase in action_map:
            return f"Command phrase '{command_phrase}' already exists"

        return None

    async def get_command_mappings(self) -> List[AutomationCommand]:
        """Get all command mappings (custom and default with overrides) for UI.

        Returns:
            List of AutomationCommand objects including custom commands and defaults
        """
        try:
            mappings = []
            commands_data = await self._storage.read(model_type=CommandsData)

            # Add custom commands (already AutomationCommand objects)
            for custom_command in commands_data.custom_commands.values():
                if custom_command.is_custom and custom_command.functional_group == "Other":
                    custom_command.functional_group = "Custom"
                mappings.append(custom_command)

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
                        is_custom=command_data.is_custom,
                        functional_group=command_data.functional_group,
                    )

                mappings.append(command_data)

            return mappings
        except Exception as e:
            logger.error(f"Failed to get command mappings: {e}")
            return []

    async def _handle_add_custom_command(self, event_data: AddCustomCommandEvent) -> None:
        """Handle adding a new custom command"""
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

    async def _handle_update_command_phrase(self, event: UpdateCommandPhraseEvent) -> None:
        """Handle update command phrase request."""
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
        await self._event_bus.publish(CommandValidationErrorEvent(error_message=error_message, command_phrase=command_phrase))

    async def _publish_mappings_updated(self, success: bool, message: str) -> None:
        """Publish mappings updated event with current command mappings."""
        current_mappings = await self.get_command_mappings()
        await self._event_bus.publish(
            CommandMappingsUpdatedEvent(
                success=success, message=message, updated_mappings=current_mappings, updated_count=len(current_mappings)
            )
        )
