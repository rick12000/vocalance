import json
import time

import pytest

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.services.storage.storage_models import AppUserConfigDocument, CommandsData, MarksData, StorageData
from vocalance.app.services.storage.storage_service import CacheEntry


@pytest.mark.parametrize(
    "age, ttl, expected",
    [(0.0, 1000.0, False), (2.0, 5.0, False), (10.0, 5.0, True)],
)
def test_cache_entry_expiration(age, ttl, expected):
    entry = CacheEntry(data="payload", timestamp=time.time() - age)
    assert entry.is_expired(ttl=ttl) is expected


async def test_write_then_read_roundtrip(storage_service):
    marks = MarksData(marks={"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}})
    assert await storage_service.write(data=marks) is True

    storage_service.clear_cache()
    result = await storage_service.read(model_type=MarksData)

    assert len(result.marks) == 2
    assert result.marks["a"].x == 1
    assert result.marks["b"].y == 4


async def test_read_missing_file_returns_default(storage_service):
    result = await storage_service.read(model_type=AppUserConfigDocument)
    assert result.overrides == {}


async def test_read_corrupted_json_returns_default(storage_service):
    path = storage_service.get_path(MarksData)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    result = await storage_service.read(model_type=MarksData)
    assert len(result.marks) == 0


async def test_valid_cache_is_served_without_reading_disk(storage_service):
    await storage_service.write(data=MarksData(marks={"a": {"x": 1, "y": 2}}))

    path = storage_service.get_path(MarksData)
    path.write_text(json.dumps({"version": 1, "marks": {"z": {"x": 9, "y": 9}}}), encoding="utf-8")

    result = await storage_service.read(model_type=MarksData)
    assert "a" in result.marks
    assert "z" not in result.marks


async def test_expired_cache_forces_disk_reload(storage_service):
    await storage_service.write(data=MarksData(marks={"a": {"x": 1, "y": 2}}))

    path = storage_service.get_path(MarksData)
    path.write_text(json.dumps({"version": 1, "marks": {"z": {"x": 9, "y": 9}}}), encoding="utf-8")
    storage_service.cache[storage_service.get_cache_key(MarksData)].timestamp = 0.0

    result = await storage_service.read(model_type=MarksData)
    assert "z" in result.marks
    assert "a" not in result.marks


async def test_clear_cache_targets_single_model(storage_service):
    await storage_service.write(data=MarksData(marks={}))
    await storage_service.write(data=CommandsData())

    storage_service.clear_cache(model_type=MarksData)

    assert storage_service.get_cache_key(MarksData) not in storage_service.cache
    assert storage_service.get_cache_key(CommandsData) in storage_service.cache


async def test_clear_cache_clears_everything(storage_service):
    await storage_service.write(data=MarksData(marks={}))
    await storage_service.write(data=CommandsData())

    storage_service.clear_cache()

    assert len(storage_service.cache) == 0


def test_get_path_rejects_unmapped_model(storage_service):
    with pytest.raises(ValueError):
        storage_service.get_path(StorageData)


async def test_nested_pydantic_models_survive_roundtrip(storage_service):
    command = AutomationCommand(
        command_key="save",
        action_type="hotkey",
        action_value="ctrl+s",
        is_custom=True,
        short_description="Save",
    )
    await storage_service.write(data=CommandsData(custom_commands={"save": command}))

    storage_service.clear_cache()
    result = await storage_service.read(model_type=CommandsData)

    assert result.custom_commands["save"].action_value == "ctrl+s"
    assert result.custom_commands["save"].is_custom is True


async def test_shutdown_clears_cache(storage_service):
    storage_service.cache["x"] = CacheEntry(data=MarksData(marks={}), timestamp=0.0)
    await storage_service.shutdown()
    assert len(storage_service.cache) == 0
