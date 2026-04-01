"""Build parser action map and UI command list from persisted `CommandsData`."""

from __future__ import annotations

from typing import Dict, List, Tuple

from vocalance.app.config.automation_command_registry import AutomationCommandRegistry
from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.services.storage.storage_models import CommandsData


def _with_effective_key(commands_data: CommandsData, template: AutomationCommand) -> AutomationCommand:
    key = commands_data.phrase_overrides.get(template.command_key, template.command_key)
    if key == template.command_key:
        return template
    return AutomationCommand(
        command_key=key,
        action_type=template.action_type,
        action_value=template.action_value,
        short_description=template.short_description,
        long_description=template.long_description,
        is_custom=template.is_custom,
        functional_group=template.functional_group,
    )


def build_command_projection(commands_data: CommandsData) -> Tuple[Dict[str, AutomationCommand], List[AutomationCommand]]:
    """Return `(phrase → command)` for parsing and an ordered list for the settings UI."""
    action_map: Dict[str, AutomationCommand] = dict(commands_data.custom_commands)

    registry_defaults = AutomationCommandRegistry.get_default_commands()
    for template in registry_defaults:
        resolved = _with_effective_key(commands_data, template)
        norm = resolved.command_key.lower().strip()
        if norm not in action_map:
            action_map[norm] = resolved

    ui_list: List[AutomationCommand] = []
    for cmd in commands_data.custom_commands.values():
        if cmd.is_custom and cmd.functional_group == "Other":
            ui_list.append(cmd.model_copy(update={"functional_group": "Custom"}))
        else:
            ui_list.append(cmd.model_copy())

    for template in registry_defaults:
        resolved = _with_effective_key(commands_data, template)
        ui_list.append(resolved.model_copy())

    return action_map, ui_list
