from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from vocalance.app.config.alias_validation import is_valid_alias_text
from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.config.hotkey_validation import is_valid_custom_hotkey

MAX_PROMPT_TEXT_LENGTH = 4000
MAX_ALIAS_KEY_LENGTH = 100
MAX_ALIAS_VALUE_LENGTH = 2000
MAX_SOUND_MAPPING_PHRASE_LENGTH = 200


class StorageData(BaseModel):
    """Base class for all storage models with versioning support."""

    version: int = Field(default=1, description="Schema version for migrations")


class Coordinate(BaseModel):
    """2D coordinate model."""

    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")


class GridClickEvent(BaseModel):
    """Grid click event record."""

    x: int
    y: int
    timestamp: float
    cell_id: Optional[str] = None


class AgenticPrompt(BaseModel):
    """Agentic prompt configuration."""

    id: str
    text: str = Field(..., max_length=MAX_PROMPT_TEXT_LENGTH)
    name: str = Field(..., max_length=200)
    created_at: str
    is_default: bool = False


class MarksData(StorageData):
    """Storage model for mark coordinates."""

    marks: Dict[str, Coordinate] = Field(default_factory=dict, description="Map of mark name to coordinate")


class AppUserConfigDocument(StorageData):
    """Canonical on-disk user configuration (UI-only overrides)."""

    overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Sparse overrides merged onto GlobalAppConfig")


class CommandsData(StorageData):
    """Storage model for custom commands and phrase overrides."""

    custom_commands: Dict[str, AutomationCommand] = Field(
        default_factory=dict, description="User-defined custom commands mapped by phrase"
    )
    phrase_overrides: Dict[str, str] = Field(default_factory=dict, description="Phrase overrides for default commands")

    @model_validator(mode="after")
    def filter_invalid_custom_commands(self) -> "CommandsData":
        self.custom_commands = {
            phrase: cmd
            for phrase, cmd in self.custom_commands.items()
            if cmd.action_type == "hotkey" and is_valid_custom_hotkey(cmd.action_value)
        }
        return self


class GridClicksData(StorageData):
    """Storage model for grid click history."""

    clicks: List[GridClickEvent] = Field(default_factory=list, description="History of grid click events")


class AgenticPromptsData(StorageData):
    """Storage model for agentic prompts."""

    prompts: List[AgenticPrompt] = Field(default_factory=list, description="List of agentic prompt configurations")
    current_prompt_id: Optional[str] = Field(default=None, description="ID of currently active prompt")


class SoundMappingsData(StorageData):
    """Storage model for sound recognition mappings."""

    mappings: Dict[str, str] = Field(default_factory=dict, description="Map of sound name to action/command")

    @field_validator("mappings")
    @classmethod
    def validate_mapping_lengths(cls, v: Dict[str, str]) -> Dict[str, str]:
        for key, phrase in v.items():
            if len(phrase) > MAX_SOUND_MAPPING_PHRASE_LENGTH:
                raise ValueError(f"Sound mapping phrase for '{key}' exceeds {MAX_SOUND_MAPPING_PHRASE_LENGTH} characters")
        return v


class DictationAliasData(StorageData):
    """Storage model for dictation alias substitutions.

    Maps activation phrases to substitution text. During dictation,
    'insert {key}' patterns are replaced with the corresponding value.
    """

    aliases: Dict[str, str] = Field(default_factory=dict, description="Map of activation phrase to substitution text")

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, v: Dict[str, str]) -> Dict[str, str]:
        for key, value in v.items():
            if len(key) > MAX_ALIAS_KEY_LENGTH:
                raise ValueError(f"Alias key '{key[:40]}...' exceeds {MAX_ALIAS_KEY_LENGTH} characters")
            if len(value) > MAX_ALIAS_VALUE_LENGTH:
                raise ValueError(f"Alias value for '{key}' exceeds {MAX_ALIAS_VALUE_LENGTH} characters")
            if not is_valid_alias_text(key) or not is_valid_alias_text(value):
                raise ValueError(f"Alias '{key}' contains characters that are not permitted")
        return v
