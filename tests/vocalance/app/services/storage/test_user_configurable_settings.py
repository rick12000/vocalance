import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.storage.user_configurable_settings import (
    ALLOWED_USER_SETTING_PATHS,
    apply_dot_path_to_config,
    effective_settings_projection,
    sanitize_user_overrides,
)


@pytest.mark.parametrize(
    "path, value",
    [
        ("grid.default_rect_count", 42),
        ("llm.max_tokens", 256),
        ("llm.context_length", 1024),
        ("vad.command_silent_chunks_for_end", 9),
    ],
)
def test_apply_dot_path_updates_nested_config(path, value):
    config = GlobalAppConfig()
    apply_dot_path_to_config(config, path, value)

    category, key = path.split(".", 1)
    assert getattr(getattr(config, category), key) == value


def test_effective_settings_projection_covers_all_allowed_paths():
    projection = effective_settings_projection(GlobalAppConfig())

    flattened = {f"{category}.{key}" for category, items in projection.items() for key in items}
    assert flattened == set(ALLOWED_USER_SETTING_PATHS)


def test_sanitize_keeps_allowed_drops_unknown():
    raw = {
        "grid": {"default_rect_count": 10, "rows": 5},
        "llm": {"max_tokens": 100},
        "bogus_category": {"whatever": 1},
    }
    result = sanitize_user_overrides(raw)

    assert result == {"grid": {"default_rect_count": 10}, "llm": {"max_tokens": 100}}


@pytest.mark.parametrize("raw", [None, [], "string", 42, {"grid": "not-a-dict"}])
def test_sanitize_ignores_non_dict_shapes(raw):
    assert sanitize_user_overrides(raw) == {}
