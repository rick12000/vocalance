import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.services.command_history_manager import CommandHistoryManager
from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry


@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    storage = Mock()
    storage.read = AsyncMock()
    storage.write = AsyncMock()
    return storage


@pytest.fixture
def history_manager(mock_storage):
    """Create CommandHistoryManager instance."""
    return CommandHistoryManager(storage=mock_storage)


@pytest.mark.asyncio
async def test_initialize_empty_history(history_manager, mock_storage):
    """Test initialization with empty history."""
    mock_storage.read.return_value = CommandHistoryData(history=[])

    success = await history_manager.initialize()

    assert success is True
    mock_storage.read.assert_called_once()

    history = await history_manager.get_full_history()
    assert len(history) == 0


@pytest.mark.asyncio
async def test_initialize_with_existing_history(history_manager, mock_storage):
    """Test initialization loads existing history."""
    existing_entries = [
        CommandHistoryEntry(command="copy", timestamp=1000.0, success=None, metadata={"source": "stt"}),
        CommandHistoryEntry(command="paste", timestamp=2000.0, success=None, metadata={"source": "stt"}),
    ]
    mock_storage.read.return_value = CommandHistoryData(history=existing_entries)

    success = await history_manager.initialize()

    assert success is True
    history = await history_manager.get_full_history()
    assert len(history) == 2
    assert history[0].command == "copy"
    assert history[1].command == "paste"


@pytest.mark.asyncio
async def test_initialize_handles_storage_error(history_manager, mock_storage):
    """Test initialization handles storage errors gracefully."""
    mock_storage.read.side_effect = Exception("Storage error")

    success = await history_manager.initialize()

    assert success is False
    history = await history_manager.get_full_history()
    assert len(history) == 0


@pytest.mark.asyncio
async def test_record_command(history_manager, mock_storage):
    """Test recording commands to in-memory history."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await history_manager.initialize()

    await history_manager.record_command(command="copy", source="stt")
    await history_manager.record_command(command="paste", source="sound")

    history = await history_manager.get_full_history()
    assert len(history) == 2
    assert history[0].command == "copy"
    assert history[0].metadata["source"] == "stt"
    assert history[1].command == "paste"
    assert history[1].metadata["source"] == "sound"


@pytest.mark.asyncio
async def test_get_recent_history(history_manager, mock_storage):
    """Test retrieving recent commands."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await history_manager.initialize()

    for i in range(10):
        await history_manager.record_command(command=f"command_{i}", source="stt")

    recent = await history_manager.get_recent_history(count=3)
    assert len(recent) == 3
    assert recent[0].command == "command_7"
    assert recent[1].command == "command_8"
    assert recent[2].command == "command_9"


@pytest.mark.asyncio
async def test_get_recent_history_more_than_available(history_manager, mock_storage):
    """Test retrieving more recent commands than available."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await history_manager.initialize()

    await history_manager.record_command(command="copy", source="stt")

    recent = await history_manager.get_recent_history(count=10)
    assert len(recent) == 1


@pytest.mark.asyncio
async def test_shutdown_writes_history(history_manager, mock_storage):
    """Test shutdown persists history to storage."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    mock_storage.write.return_value = True
    await history_manager.initialize()

    await history_manager.record_command(command="copy", source="stt")
    await history_manager.record_command(command="paste", source="stt")

    success = await history_manager.shutdown()

    assert success is True
    mock_storage.write.assert_called_once()

    # Verify written data
    written_data = mock_storage.write.call_args[1]["data"]
    assert isinstance(written_data, CommandHistoryData)
    assert len(written_data.history) == 2


@pytest.mark.asyncio
async def test_shutdown_empty_history(history_manager, mock_storage):
    """Test shutdown with empty history doesn't write."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await history_manager.initialize()

    success = await history_manager.shutdown()

    assert success is True
    mock_storage.write.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_handles_write_failure(history_manager, mock_storage):
    """Test shutdown handles storage write failure."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    mock_storage.write.return_value = False
    await history_manager.initialize()

    await history_manager.record_command(command="copy", source="stt")

    success = await history_manager.shutdown()

    assert success is False


@pytest.mark.asyncio
async def test_concurrent_recording(history_manager, mock_storage):
    """Test thread-safe concurrent command recording."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await history_manager.initialize()

    # Record commands concurrently
    tasks = [history_manager.record_command(command=f"cmd_{i}", source="stt") for i in range(50)]
    await asyncio.gather(*tasks)

    history = await history_manager.get_full_history()
    assert len(history) == 50
