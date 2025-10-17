import logging
from typing import Dict

from iris.app.config.automation_command_registry import AutomationCommandRegistry
from iris.app.config.command_types import AutomationCommand
from iris.app.services.storage.storage_models import CommandsData
from iris.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class CommandActionMapProvider:
    """Centralized provider for automation command action maps.

    Single source of truth for building the complete command map from custom
    commands and default commands with phrase overrides.
    """

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage
        logger.info("CommandActionMapProvider initialized")

    async def get_action_map(self) -> Dict[str, AutomationCommand]:
        """Build complete action map from custom and default commands.

        Loads custom commands from storage, merges with default commands, applies
        phrase overrides, and returns normalized map of phrase to command object.

        Returns:
            Dictionary mapping normalized command phrases to AutomationCommand objects
        """
        action_map: Dict[str, AutomationCommand] = {}

        commands_data = await self._storage.read(model_type=CommandsData)

        for normalized_phrase, command_obj in commands_data.custom_commands.items():
            action_map[normalized_phrase] = command_obj

        default_commands = AutomationCommandRegistry.get_default_commands()

        for command_data in default_commands:
            normalized_phrase = command_data.command_key.lower().strip()

            if normalized_phrase not in action_map:
                effective_phrase = commands_data.phrase_overrides.get(command_data.command_key, command_data.command_key)

                if effective_phrase != command_data.command_key:
                    command_data = AutomationCommand(
                        command_key=effective_phrase,
                        action_type=command_data.action_type,
                        action_value=command_data.action_value,
                        short_description=command_data.short_description,
                        long_description=command_data.long_description,
                        is_custom=command_data.is_custom,
                    )
                    normalized_phrase = effective_phrase.lower().strip()

                action_map[normalized_phrase] = command_data

        return action_map
