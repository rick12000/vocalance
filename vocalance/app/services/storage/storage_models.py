from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from vocalance.app.config.command_types import AutomationCommand


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
    text: str
    name: str
    created_at: str
    is_default: bool = False


class CommandHistoryEntry(BaseModel):
    """Command execution history entry."""

    command: str
    timestamp: float
    success: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarksData(StorageData):
    """Storage model for mark coordinates."""

    marks: Dict[str, Coordinate] = Field(default_factory=dict, description="Map of mark name to coordinate")


class SettingsData(StorageData):
    """Storage model for user settings overrides."""

    user_overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="User setting overrides organized by category"
    )


class CommandsData(StorageData):
    """Storage model for custom commands and phrase overrides."""

    custom_commands: Dict[str, AutomationCommand] = Field(
        default_factory=dict, description="User-defined custom commands mapped by phrase"
    )
    phrase_overrides: Dict[str, str] = Field(default_factory=dict, description="Phrase overrides for default commands")


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


class CommandHistoryData(StorageData):
    """Storage model for command execution history."""

    history: List[CommandHistoryEntry] = Field(default_factory=list, description="Historical command execution records")

    @field_validator("history")
    @classmethod
    def validate_history_limit(cls, v: List[CommandHistoryEntry]) -> List[CommandHistoryEntry]:
        max_entries = 10000
        if len(v) > max_entries:
            return v[-max_entries:]
        return v
