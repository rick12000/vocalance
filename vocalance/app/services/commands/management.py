"""Persisted custom commands and phrase overrides; validates against protected terms."""

from __future__ import annotations

import logging
from typing import List, Optional

from vocalance.app.config.automation_command_registry import AutomationCommandRegistry
from vocalance.app.config.command_types import AutomationCommand
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
from vocalance.app.services.commands.action_map_provider import CommandActionMapProvider
from vocalance.app.services.commands.projection import build_command_projection
from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.storage.storage_models import CommandsData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


def _registry_phrase_for_current_phrase(commands_data: CommandsData, current_norm: str) -> Optional[str]:
    for original, override in commands_data.phrase_overrides.items():
        if override.lower().strip() == current_norm:
            return original
    for cmd in AutomationCommandRegistry.get_default_commands():
        if cmd.command_key.lower().strip() == current_norm:
            return cmd.command_key
    return None


class CommandManagementService:
    """CRUD for stored automation commands; publishes mapping updates on the event bus."""

    def __init__(
        self,
        event_bus: EventBus,
        storage: StorageService,
        protected_terms_validator: ProtectedTermsValidator,
        action_map_provider: CommandActionMapProvider,
    ) -> None:
        self._event_bus = event_bus
        self._storage = storage
        self._protected_terms_validator = protected_terms_validator
        self._action_map_provider = action_map_provider

    def setup_subscriptions(self) -> None:
        self._event_bus.subscribe(event_type=AddCustomCommandEvent, handler=self._handle_add_custom_command)
        self._event_bus.subscribe(event_type=UpdateCommandPhraseEvent, handler=self._handle_update_command_phrase)
        self._event_bus.subscribe(event_type=DeleteCustomCommandEvent, handler=self._handle_delete_custom_command)
        self._event_bus.subscribe(event_type=RequestCommandMappingsEvent, handler=self._handle_request_command_mappings)
        self._event_bus.subscribe(event_type=ResetCommandsToDefaultsEvent, handler=self._handle_reset_to_defaults)

    async def _validate_command_phrase(self, command_phrase: str, exclude_phrase: str = "") -> Optional[str]:
        is_valid, error_msg = await self._protected_terms_validator.validate_term(
            term=command_phrase, exclude_term=exclude_phrase or None
        )
        if not is_valid:
            return error_msg

        norm = command_phrase.lower().strip()
        ex = exclude_phrase.lower().strip() if exclude_phrase else ""
        if norm == ex:
            return None

        action_map = await self._action_map_provider.get_action_map()
        if norm in action_map:
            return f"Command phrase '{command_phrase}' already exists"
        return None

    async def get_command_mappings(self) -> List[AutomationCommand]:
        commands_data = await self._storage.read(model_type=CommandsData)
        return build_command_projection(commands_data)[1]

    async def _handle_add_custom_command(self, add_command: AddCustomCommandEvent) -> None:
        command = add_command.command
        phrase = command.command_key.lower().strip()
        err = await self._validate_command_phrase(phrase)
        if err:
            await self._publish_validation_error(err, phrase)
            return

        command.is_custom = True
        commands_data = await self._storage.read(model_type=CommandsData)
        commands_data.custom_commands[phrase] = command
        if await self._storage.write(data=commands_data):
            await self._publish_mappings_updated(True, f"Added custom command: {phrase}")
        else:
            await self._publish_validation_error("Failed to store custom command", phrase)

    async def _handle_update_command_phrase(self, phrase_update: UpdateCommandPhraseEvent) -> None:
        old_phrase = phrase_update.old_command_phrase
        new_phrase = phrase_update.new_command_phrase
        old_norm = old_phrase.lower().strip()
        new_norm = new_phrase.lower().strip()

        err = await self._validate_command_phrase(new_norm, exclude_phrase=old_phrase)
        if err:
            await self._publish_validation_error(err, new_phrase)
            return

        commands_data = await self._storage.read(model_type=CommandsData)
        success = False

        if old_norm in commands_data.custom_commands:
            obj = commands_data.custom_commands[old_norm]
            obj.command_key = new_norm
            del commands_data.custom_commands[old_norm]
            commands_data.custom_commands[new_norm] = obj
            success = await self._storage.write(data=commands_data)
        else:
            original = _registry_phrase_for_current_phrase(commands_data, old_norm)
            if original:
                commands_data.phrase_overrides[original] = new_norm
                success = await self._storage.write(data=commands_data)
            else:
                logger.error("Could not find original command for phrase %r", old_phrase)
                await self._publish_validation_error(f"Could not find command '{old_phrase}' to update", old_phrase)
                return

        if success:
            await self._publish_mappings_updated(True, f"Updated command phrase: '{old_phrase}' -> '{new_phrase}'")
        else:
            await self._publish_validation_error("Failed to update command phrase", new_phrase)

    async def _handle_delete_custom_command(self, delete_command: DeleteCustomCommandEvent) -> None:
        phrase = delete_command.command.command_key.lower().strip()
        commands_data = await self._storage.read(model_type=CommandsData)
        if phrase in commands_data.custom_commands:
            del commands_data.custom_commands[phrase]
            success = await self._storage.write(data=commands_data)
        else:
            success = True

        if success:
            await self._publish_mappings_updated(True, f"Deleted custom command: {phrase}")
        else:
            await self._publish_validation_error("Failed to delete custom command")

    async def _handle_request_command_mappings(self, _request: RequestCommandMappingsEvent) -> None:
        mappings = await self.get_command_mappings()
        await self._event_bus.publish(CommandMappingsResponseEvent(mappings=mappings))

    async def _handle_reset_to_defaults(self, _reset: ResetCommandsToDefaultsEvent) -> None:
        if await self._storage.write(data=CommandsData()):
            await self._publish_mappings_updated(True, "Reset commands to defaults")
        else:
            await self._publish_validation_error("Failed to reset commands to defaults")

    async def _publish_validation_error(self, error_message: str, command_phrase: str = "") -> None:
        await self._event_bus.publish(CommandValidationErrorEvent(error_message=error_message, command_phrase=command_phrase))

    async def _publish_mappings_updated(self, success: bool, message: str) -> None:
        current = await self.get_command_mappings()
        await self._event_bus.publish(
            CommandMappingsUpdatedEvent(
                success=success,
                message=message,
                updated_mappings=current,
                updated_count=len(current),
            )
        )
