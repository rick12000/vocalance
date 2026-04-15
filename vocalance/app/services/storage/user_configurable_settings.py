from typing import Any, Dict, FrozenSet, Tuple

from pydantic import BaseModel, ConfigDict, Field

from vocalance.app.config.app_config import GlobalAppConfig


class UserConfigurableField(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str = Field(description="Dot path, e.g. llm.max_tokens")
    label: str = Field(description="Short UI label")
    llm_session_stale: bool = Field(default=False)


USER_CONFIGURABLE_FIELDS: Tuple[UserConfigurableField, ...] = (
    UserConfigurableField(path="llm.context_length", label="Context length", llm_session_stale=True),
    UserConfigurableField(path="llm.max_tokens", label="Max tokens", llm_session_stale=True),
    UserConfigurableField(path="llm.selected_model_id", label="Model", llm_session_stale=True),
    UserConfigurableField(path="grid.default_rect_count", label="Default cell count"),
    UserConfigurableField(path="sound_recognizer.confidence_threshold", label="Confidence threshold"),
    UserConfigurableField(path="sound_recognizer.vote_threshold", label="Vote threshold"),
    UserConfigurableField(path="vad.command_silent_chunks_for_end", label="Max silent command chunks"),
)

ALLOWED_USER_SETTING_PATHS: FrozenSet[str] = frozenset(f.path for f in USER_CONFIGURABLE_FIELDS)

LLM_SESSION_SETTING_PATHS: FrozenSet[str] = frozenset(f.path for f in USER_CONFIGURABLE_FIELDS if f.llm_session_stale)


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
