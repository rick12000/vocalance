from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from vocalance.app.config.app_config import GlobalAppConfig


class UserConfigurableField(BaseModel):
    """Metadata for one user-editable setting."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="Dot path, e.g. llm.max_tokens")
    label: str = Field(description="Short UI label")
    section: str = Field(description="UI section heading")
    value_type: type = Field(default=str, description="Python type used for coercion (int, float, str)")
    llm_session_stale: bool = Field(default=False)


USER_CONFIGURABLE_FIELDS: Tuple[UserConfigurableField, ...] = (
    UserConfigurableField(
        path="llm.context_length", label="Max Context Tokens", section="LLM model", value_type=int, llm_session_stale=True
    ),
    UserConfigurableField(
        path="llm.max_tokens", label="Max Output Tokens", section="LLM model", value_type=int, llm_session_stale=True
    ),
    UserConfigurableField(
        path="llm.selected_model_id", label="Model", section="LLM model", value_type=str, llm_session_stale=True
    ),
    UserConfigurableField(path="grid.default_rect_count", label="Default Cell Count", section="Grid Settings", value_type=int),
    UserConfigurableField(
        path="sound_recognizer.confidence_threshold",
        label="Confidence Threshold",
        section="Sound Recognizer Settings",
        value_type=float,
    ),
    UserConfigurableField(
        path="sound_recognizer.vote_threshold", label="Vote Threshold", section="Sound Recognizer Settings", value_type=float
    ),
    UserConfigurableField(
        path="vad.command_silent_chunks_for_end", label="Max Silent Command Chunks", section="Voice Settings", value_type=int
    ),
)

FIELD_BY_PATH: Dict[str, UserConfigurableField] = {f.path: f for f in USER_CONFIGURABLE_FIELDS}

ALLOWED_USER_SETTING_PATHS: FrozenSet[str] = frozenset(FIELD_BY_PATH)

LLM_SESSION_SETTING_PATHS: FrozenSet[str] = frozenset(f.path for f in USER_CONFIGURABLE_FIELDS if f.llm_session_stale)


def get_config_field_bounds(path: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract ge/le bounds from the GlobalAppConfig Pydantic field for ``path``."""
    category, key = path.split(".", 1)
    sub_model_class = GlobalAppConfig.model_fields[category].annotation
    field_info = sub_model_class.model_fields.get(key)
    if field_info is None:
        return None, None
    ge_val = next((m.ge for m in field_info.metadata if hasattr(m, "ge")), None)
    le_val = next((m.le for m in field_info.metadata if hasattr(m, "le")), None)
    return ge_val, le_val


def read_nested_config_value(config: GlobalAppConfig, path: str) -> object:
    category, key = path.split(".", 1)
    sub = getattr(config, category)
    return getattr(sub, key)


def effective_settings_projection(config: GlobalAppConfig) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for field in USER_CONFIGURABLE_FIELDS:
        category, key = field.path.split(".", 1)
        out.setdefault(category, {})[key] = read_nested_config_value(config, field.path)
    return out


def apply_dot_path_to_config(config: GlobalAppConfig, path: str, value: object) -> None:
    category, key = path.split(".", 1)
    sub = getattr(config, category)
    merged = type(sub).model_validate({**sub.model_dump(mode="python"), key: value})
    setattr(config, category, merged)


def sanitize_user_overrides(raw: object) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for category, items in raw.items():
        if not isinstance(items, dict):
            continue
        for key, value in items.items():
            path = f"{category}.{key}"
            if path in ALLOWED_USER_SETTING_PATHS:
                out.setdefault(category, {})[key] = value
    return out
