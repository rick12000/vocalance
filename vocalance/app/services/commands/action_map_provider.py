"""Load automation commands from storage and expose the parser action map."""

from __future__ import annotations

from typing import Dict

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.services.commands.projection import build_command_projection
from vocalance.app.services.storage.storage_models import CommandsData
from vocalance.app.services.storage.storage_service import StorageService


class CommandActionMapProvider:
    """Reads `CommandsData` from storage and builds the normalized phrase → command map."""

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    async def get_action_map(self) -> Dict[str, AutomationCommand]:
        commands_data = await self._storage.read(model_type=CommandsData)
        action_map, _ = build_command_projection(commands_data)
        return action_map
