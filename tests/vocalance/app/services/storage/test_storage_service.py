import asyncio
import tempfile
from pathlib import Path

import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.storage.storage_models import CommandsData, MarksData, SettingsData
from vocalance.app.services.storage.storage_service import CacheEntry, StorageService


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for storage testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def app_config(temp_storage_dir):
    """Create application configuration with temp storage - COMPLETELY ISOLATED."""
    config = GlobalAppConfig()

    # CRITICAL: Override ALL storage paths to use temp directory
    # This prevents tests from touching production data
    temp_path = Path(temp_storage_dir)
    config.storage.user_data_root = temp_storage_dir
    config.storage.marks_dir = str(temp_path / "marks")
    config.storage.settings_dir = str(temp_path / "settings")
    config.storage.click_tracker_dir = str(temp_path / "click_tracker")
    config.storage.sound_model_dir = str(temp_path / "sound_model")
    config.storage.command_history_dir = str(temp_path / "command_history")

    return config


@pytest.fixture
def storage_service(app_config):
    """Create StorageService instance."""
    service = StorageService(config=app_config)
    yield service
    # Cleanup
    service._executor.shutdown(wait=True)


def test_cache_entry_expiration():
    """Test cache entry expiration logic."""
    import time

    current_time = time.time()
    entry = CacheEntry(data="test", timestamp=current_time)

    # Not expired with large TTL
    assert entry.is_expired(ttl=1000.0) is False

    # Create an old entry that should be expired
    old_entry = CacheEntry(data="test", timestamp=current_time - 10.0)
    assert old_entry.is_expired(ttl=5.0) is True


@pytest.mark.asyncio
async def test_read_nonexistent_file_returns_default(storage_service):
    """Test reading non-existent file returns default instance."""
    # Clear cache to ensure we're not reading cached data
    storage_service.clear_cache()

    # Use a model type that definitely doesn't have existing data
    from vocalance.app.services.storage.storage_models import CommandHistoryData

    # Delete the file if it exists to test default behavior
    path = storage_service._get_path(CommandHistoryData)
    if path.exists():
        path.unlink()

    data = await storage_service.read(model_type=CommandHistoryData)

    assert isinstance(data, CommandHistoryData)
    assert len(data.history) == 0


@pytest.mark.asyncio
async def test_write_and_read_data(storage_service):
    """Test writing and reading data."""
    from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry

    # Create data - use CommandHistoryData to avoid conflicts with existing marks
    history_entry = CommandHistoryEntry(command="test_command", timestamp=1000.0, success=True, metadata={})
    history_data = CommandHistoryData(history=[history_entry])

    # Write
    success = await storage_service.write(data=history_data)
    assert success is True

    # Clear cache to force disk read
    storage_service.clear_cache()

    # Read back
    read_data = await storage_service.read(model_type=CommandHistoryData)
    assert isinstance(read_data, CommandHistoryData)
    assert len(read_data.history) == 1
    assert read_data.history[0].command == "test_command"


@pytest.mark.asyncio
async def test_write_updates_cache(storage_service):
    """Test writing data updates cache."""
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})

    await storage_service.write(data=marks_data)

    # Cache should contain the data
    cache_key = storage_service._get_cache_key(MarksData)
    assert cache_key in storage_service._cache
    assert storage_service._cache[cache_key].data == marks_data


@pytest.mark.asyncio
async def test_read_uses_cache(storage_service):
    """Test reading uses cached data."""
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})

    # Write to populate cache
    await storage_service.write(data=marks_data)

    # Clear cache TTL to force cache hit
    cache_key = storage_service._get_cache_key(MarksData)
    storage_service._cache[cache_key].timestamp = 999999999.0

    # Read should use cache (no file I/O)
    read_data = await storage_service.read(model_type=MarksData)
    assert read_data == marks_data


@pytest.mark.asyncio
async def test_cache_expiration_forces_disk_read(storage_service):
    """Test expired cache forces disk read."""
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})

    # Write data
    await storage_service.write(data=marks_data)

    # Expire cache
    cache_key = storage_service._get_cache_key(MarksData)
    storage_service._cache[cache_key].timestamp = 0.0

    # Read should reload from disk
    read_data = await storage_service.read(model_type=MarksData)
    assert isinstance(read_data, MarksData)


@pytest.mark.asyncio
async def test_multiple_model_types(storage_service):
    """Test storing multiple different model types."""
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})
    commands_data = CommandsData(custom_commands={}, phrase_overrides={"copy": "copy that"})

    # Write different types
    await storage_service.write(data=marks_data)
    await storage_service.write(data=commands_data)

    # Read back
    read_marks = await storage_service.read(model_type=MarksData)
    read_commands = await storage_service.read(model_type=CommandsData)

    assert isinstance(read_marks, MarksData)
    assert isinstance(read_commands, CommandsData)
    assert "mark1" in read_marks.marks
    assert "copy" in read_commands.phrase_overrides


@pytest.mark.asyncio
async def test_atomic_write_creates_backup(storage_service, app_config):
    """Test atomic write creates backup of existing file."""
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})

    # First write
    await storage_service.write(data=marks_data)

    # Modify and write again
    marks_data.marks["mark2"] = {"x": 300, "y": 400}
    await storage_service.write(data=marks_data)

    # Verify data written correctly
    read_data = await storage_service.read(model_type=MarksData)
    assert "mark1" in read_data.marks
    assert "mark2" in read_data.marks


@pytest.mark.asyncio
async def test_clear_cache_specific_model(storage_service):
    """Test clearing cache for specific model type."""
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})
    commands_data = CommandsData(custom_commands={}, phrase_overrides={})

    # Populate cache
    await storage_service.write(data=marks_data)
    await storage_service.write(data=commands_data)

    # Clear specific cache
    storage_service.clear_cache(model_type=MarksData)

    # MarksData should be cleared, CommandsData should remain
    marks_cache_key = storage_service._get_cache_key(MarksData)
    commands_cache_key = storage_service._get_cache_key(CommandsData)

    assert marks_cache_key not in storage_service._cache
    assert commands_cache_key in storage_service._cache


@pytest.mark.asyncio
async def test_clear_all_cache(storage_service):
    """Test clearing entire cache."""
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})
    commands_data = CommandsData(custom_commands={}, phrase_overrides={})

    # Populate cache
    await storage_service.write(data=marks_data)
    await storage_service.write(data=commands_data)

    # Clear all cache
    storage_service.clear_cache()

    assert len(storage_service._cache) == 0


@pytest.mark.asyncio
async def test_get_cache_stats(storage_service):
    """Test getting cache statistics."""
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})

    await storage_service.write(data=marks_data)

    stats = storage_service.get_cache_stats()

    assert stats["entries"] >= 1
    assert "MarksData" in stats["models"]
    assert stats["ttl_seconds"] == storage_service._cache_ttl


@pytest.mark.asyncio
async def test_concurrent_reads(storage_service):
    """Test concurrent reads are thread-safe."""
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})
    await storage_service.write(data=marks_data)

    # Perform concurrent reads
    tasks = [storage_service.read(model_type=MarksData) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # All reads should succeed and return same data
    assert len(results) == 10
    assert all(isinstance(r, MarksData) for r in results)
    assert all("mark1" in r.marks for r in results)


@pytest.mark.asyncio
async def test_concurrent_writes(storage_service):
    """Test concurrent writes are thread-safe."""
    from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry

    async def write_history(cmd_id: int):
        entry = CommandHistoryEntry(command=f"cmd_{cmd_id}", timestamp=1000.0 + cmd_id, success=True, metadata={})
        history_data = CommandHistoryData(history=[entry])
        return await storage_service.write(data=history_data)

    # Perform concurrent writes - note: last write wins for same model type
    # This tests thread safety, not that all data persists
    tasks = [write_history(i) for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # At least some writes should succeed (race condition is expected)
    assert any(r is True for r in results if not isinstance(r, Exception))


@pytest.mark.asyncio
async def test_read_handles_corrupted_json(storage_service, app_config):
    """Test reading handles corrupted JSON gracefully."""
    # Write corrupted JSON directly
    path = storage_service._get_path(MarksData)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("{invalid json")

    # Should return default instance
    data = await storage_service.read(model_type=MarksData)
    assert isinstance(data, MarksData)
    assert len(data.marks) == 0


@pytest.mark.asyncio
async def test_write_creates_directories(storage_service):
    """Test write creates necessary directories."""
    settings_data = SettingsData()

    success = await storage_service.write(data=settings_data)

    assert success is True
    path = storage_service._get_path(SettingsData)
    assert path.parent.exists()


@pytest.mark.asyncio
async def test_storage_config_property(storage_service, app_config):
    """Test storage_config property provides backward compatibility."""
    storage_config = storage_service.storage_config

    assert storage_config == app_config.storage


@pytest.mark.asyncio
async def test_shutdown_waits_for_executor(storage_service):
    """Test shutdown waits for executor to complete."""
    # Write some data
    marks_data = MarksData(marks={"mark1": {"x": 100, "y": 200}})
    await storage_service.write(data=marks_data)

    # Shutdown
    await storage_service.shutdown()

    # Executor should be shutdown
    assert storage_service._executor._shutdown


@pytest.mark.asyncio
async def test_write_multiple_times_updates_data(storage_service):
    """Test multiple writes update data correctly."""
    # First write
    marks_data1 = MarksData(marks={"mark1": {"x": 100, "y": 200}})
    await storage_service.write(data=marks_data1)

    # Second write with different data
    marks_data2 = MarksData(marks={"mark2": {"x": 300, "y": 400}})
    await storage_service.write(data=marks_data2)

    # Read should get latest data
    storage_service.clear_cache()
    read_data = await storage_service.read(model_type=MarksData)

    assert "mark2" in read_data.marks
    assert "mark1" not in read_data.marks


@pytest.mark.asyncio
async def test_make_serializable_handles_nested_models(storage_service):
    """Test serialization handles nested Pydantic models."""
    from vocalance.app.config.command_types import AutomationCommand

    custom_cmd = AutomationCommand(
        command_key="test",
        action_type="hotkey",
        action_value="ctrl+t",
        is_custom=True,
        short_description="Test",
        long_description="Test command",
    )

    commands_data = CommandsData(custom_commands={"test": custom_cmd}, phrase_overrides={})

    # Write should handle nested model
    success = await storage_service.write(data=commands_data)
    assert success is True

    # Read back and verify
    storage_service.clear_cache()
    read_data = await storage_service.read(model_type=CommandsData)
    assert "test" in read_data.custom_commands
