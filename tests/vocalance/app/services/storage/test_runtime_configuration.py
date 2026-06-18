import pytest

from vocalance.app.services.storage.storage_models import AppUserConfigDocument


@pytest.mark.parametrize(
    "path, expected",
    [
        ("grid.default_rect_count", 500),
        ("llm.max_tokens", 1500),
    ],
)
async def test_get_setting_returns_live_value_for_allowed_path(runtime_config_store, path, expected):
    assert runtime_config_store.get_setting(path) == expected


def test_get_setting_returns_default_for_disallowed_path(runtime_config_store):
    sentinel = object()
    assert runtime_config_store.get_setting("grid.rows", default=sentinel) is sentinel


async def test_update_rejects_disallowed_path(runtime_config_store):
    ok = await runtime_config_store.update_multiple_settings({"grid.rows": 9})
    assert ok is False


async def test_update_applies_and_persists_allowed_setting(runtime_config_store):
    ok = await runtime_config_store.update_multiple_settings({"grid.default_rect_count": 12})

    assert ok is True
    assert runtime_config_store.current().grid.default_rect_count == 12

    persisted = await runtime_config_store.storage.read(model_type=AppUserConfigDocument)
    assert persisted.overrides["grid"]["default_rect_count"] == 12


async def test_reset_to_defaults_restores_code_defaults(runtime_config_store):
    await runtime_config_store.update_multiple_settings({"grid.default_rect_count": 12})

    ok, _ = await runtime_config_store.reset_to_defaults()

    assert ok is True
    assert runtime_config_store.current().grid.default_rect_count == 500
    assert runtime_config_store.overrides == {}
