from __future__ import annotations

from typing import List, Optional, Tuple

from vocalance.app.config.automation_command_registry import AutomationCommandRegistry
from vocalance.app.config.command_types import AutomationCommand, ExactMatchCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_management_events import (
    CommandMappingsUpdatedEvent,
    CommandUiOperationEvent,
    CommandValidationErrorEvent,
)
from vocalance.app.services.base_service import Service
from vocalance.app.services.commands.utilities.command_projection import build_command_projection, load_action_map
from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.storage.storage_models import CommandsData
from vocalance.app.services.storage.storage_service import StorageService


def registry_phrase_for_normalized_phrase(commands_data: CommandsData, normalized_phrase: str) -> Optional[str]:
    """Resolve the registry or override original phrase for a normalized key, if any."""
    for original, override in commands_data.phrase_overrides.items():
        if override.lower().strip() == normalized_phrase:
            return original
    for cmd in AutomationCommandRegistry.get_default_commands():
        if cmd.command_key.lower().strip() == normalized_phrase:
            return cmd.command_key
    return None


class CommandManagementService(Service):
    """CRUD for stored automation commands; publishes mapping updates on the event bus."""

    def __init__(
        self,
        event_bus: EventBus,
        storage: StorageService,
        protected_terms_validator: ProtectedTermsValidator,
    ) -> None:
        super().__init__(event_bus)
        self.storage = storage
        self.protected_terms_validator = protected_terms_validator
        self.subscribe(CommandUiOperationEvent, self._handle_command_ui_operation)

    async def _handle_command_ui_operation(self, event: CommandUiOperationEvent) -> None:
        op: str = event.op
        if op == "add_hotkey":
            command = ExactMatchCommand(
                command_key=event.command_phrase,
                action_type="hotkey",
                action_value=event.hotkey_value,
                is_custom=True,
                short_description="Custom Command",
                long_description=f"Custom hotkey command: {event.hotkey_value}",
                functional_group="Custom",
            )
            await self.add_command(command)
        elif op == "update_phrase":
            await self.update_command_phrase(event.old_phrase, event.new_phrase)
        elif op == "delete_phrase":
            phrase = event.command_phrase.lower().strip()
            commands = await self.get_command_mappings()
            for cmd in commands:
                if cmd.command_key.lower().strip() == phrase:
                    await self.delete_command(cmd)
                    return
        elif op == "reset_defaults":
            await self.reset_to_defaults()
        elif op == "refresh_mappings":
            await self.publish_mappings_updated(True, "Command mappings refreshed")

    async def validate_command_phrase(self, command_phrase: str, exclude_phrase: str = "") -> Optional[str]:
        """Return an error message if ``command_phrase`` is invalid or collides; otherwise None."""
        is_valid, error_msg = await self.protected_terms_validator.validate_term(
            term=command_phrase, exclude_term=exclude_phrase or None
        )
        if not is_valid:
            return error_msg

        norm = command_phrase.lower().strip()
        ex = exclude_phrase.lower().strip() if exclude_phrase else ""
        if norm == ex:
            return None

        action_map = await load_action_map(self.storage)
        if norm in action_map:
            return f"Command phrase '{command_phrase}' already exists"
        return None

    async def get_command_mappings(self) -> List[AutomationCommand]:
        """Return merged registry and custom commands with phrase overrides applied."""
        commands_data = await self.storage.read(model_type=CommandsData)
        return build_command_projection(commands_data)[1]

    async def add_command(self, command: AutomationCommand) -> Tuple[bool, str]:
        """Persist a custom command after validation; emits validation errors on the bus when blocked."""
        phrase: str = command.command_key.lower().strip()
        err = await self.validate_command_phrase(phrase)
        if err:
            await self.event_bus.publish(CommandValidationErrorEvent(error_message=err, command_phrase=phrase))
            return False, err

        command.is_custom = True
        commands_data = await self.storage.read(model_type=CommandsData)
        commands_data.custom_commands[phrase] = command
        if await self.storage.write(data=commands_data):
            await self.publish_mappings_updated(True, f"Added custom command: {phrase}")
            return True, ""
        err: str = "Failed to store custom command"
        await self.event_bus.publish(CommandValidationErrorEvent(error_message=err, command_phrase=phrase))
        return False, err

    async def update_command_phrase(self, old_phrase: str, new_phrase: str) -> Tuple[bool, str]:
        """Rename a custom phrase or set a registry phrase override."""
        old_norm: str = old_phrase.lower().strip()
        new_norm: str = new_phrase.lower().strip()

        err = await self.validate_command_phrase(new_norm, exclude_phrase=old_phrase)
        if err:
            await self.event_bus.publish(CommandValidationErrorEvent(error_message=err, command_phrase=new_phrase))
            return False, err

        commands_data = await self.storage.read(model_type=CommandsData)

        success: bool
        if old_norm in commands_data.custom_commands:
            obj = commands_data.custom_commands[old_norm]
            obj.command_key = new_norm
            del commands_data.custom_commands[old_norm]
            commands_data.custom_commands[new_norm] = obj
            success = await self.storage.write(data=commands_data)
        else:
            original = registry_phrase_for_normalized_phrase(commands_data, old_norm)
            if original:
                commands_data.phrase_overrides[original] = new_norm
                success = await self.storage.write(data=commands_data)
            else:
                err = f"Could not find command '{old_phrase}' to update"
                await self.event_bus.publish(CommandValidationErrorEvent(error_message=err, command_phrase=old_phrase))
                return False, err

        if success:
            await self.publish_mappings_updated(True, f"Updated command phrase: '{old_phrase}' -> '{new_phrase}'")
            return True, ""
        err = "Failed to update command phrase"
        await self.event_bus.publish(CommandValidationErrorEvent(error_message=err, command_phrase=new_phrase))
        return False, err

    async def delete_command(self, command: AutomationCommand) -> Tuple[bool, str]:
        """Remove a custom command phrase from storage when present."""
        phrase: str = command.command_key.lower().strip()
        commands_data = await self.storage.read(model_type=CommandsData)
        if phrase in commands_data.custom_commands:
            del commands_data.custom_commands[phrase]
            success = await self.storage.write(data=commands_data)
        else:
            success = True

        if success:
            await self.publish_mappings_updated(True, f"Deleted custom command: {phrase}")
            return True, ""
        err = "Failed to delete custom command"
        await self.event_bus.publish(CommandValidationErrorEvent(error_message=err, command_phrase=phrase))
        return False, err

    async def reset_to_defaults(self) -> Tuple[bool, str]:
        """Replace stored commands with defaults and broadcast."""
        if await self.storage.write(data=CommandsData()):
            await self.publish_mappings_updated(True, "Reset commands to defaults")
            return True, ""
        err = "Failed to reset commands to defaults"
        await self.event_bus.publish(CommandValidationErrorEvent(error_message=err))
        return False, err

    async def publish_mappings_updated(self, success: bool, message: str) -> None:
        """Publish the current command list for UI and other subscribers."""
        current = await self.get_command_mappings()
        await self.event_bus.publish(
            CommandMappingsUpdatedEvent(
                success=success,
                message=message,
                updated_mappings=current,
                updated_count=len(current),
            )
        )
