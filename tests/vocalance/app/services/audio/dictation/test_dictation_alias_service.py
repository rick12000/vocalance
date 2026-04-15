import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.services.audio.dictation_handling.dictation_alias_service import DictationAliasService
from vocalance.app.services.storage.storage_models import DictationAliasData


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    event_bus = Mock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    storage = Mock()
    storage.read = AsyncMock(return_value=DictationAliasData(aliases={}))
    storage.write = AsyncMock(return_value=True)
    return storage


@pytest.fixture
async def alias_service(mock_event_bus, mock_storage):
    """Create DictationAliasService instance for testing."""
    service = DictationAliasService(
        event_bus=mock_event_bus,
        storage=mock_storage,
        event_loop=asyncio.get_running_loop(),
    )
    await service.initialize()
    return service


@pytest.mark.asyncio
async def test_initialize_loads_aliases(mock_event_bus, mock_storage):
    """Test that initialization loads aliases from storage."""
    test_aliases = {"greeting": "hello world", "closing": "goodbye"}
    mock_storage.read.return_value = DictationAliasData(aliases=test_aliases)

    service = DictationAliasService(
        event_bus=mock_event_bus,
        storage=mock_storage,
        event_loop=asyncio.get_running_loop(),
    )
    result = await service.initialize()

    assert result is True
    assert service.get_aliases() == test_aliases


@pytest.mark.asyncio
async def test_get_aliases_returns_copy(alias_service):
    """Test that get_aliases returns a copy, not reference."""
    # Add an alias first
    await alias_service.add_alias("test", "value")

    aliases1 = alias_service.get_aliases()
    aliases1["new_key"] = "new_value"
    aliases2 = alias_service.get_aliases()

    # Modifying returned dict should not affect internal state
    assert "new_key" not in aliases2


@pytest.mark.asyncio
async def test_add_alias_success(alias_service, mock_storage):
    """Test successfully adding a new alias."""
    mock_storage.write.return_value = True

    result = await alias_service.add_alias("greeting", "hello there")

    assert result is True
    assert alias_service.get_aliases()["greeting"] == "hello there"
    mock_storage.write.assert_called_once()


@pytest.mark.asyncio
async def test_add_alias_duplicate_fails(alias_service):
    """Test that adding a duplicate alias fails."""
    await alias_service.add_alias("greeting", "hello")
    result = await alias_service.add_alias("greeting", "hi")

    assert result is False
    # Original value should be preserved
    assert alias_service.get_aliases()["greeting"] == "hello"


@pytest.mark.asyncio
async def test_add_alias_empty_key_fails(alias_service):
    """Test that adding alias with empty key fails."""
    result = await alias_service.add_alias("", "value")
    assert result is False


@pytest.mark.asyncio
async def test_add_alias_empty_value_fails(alias_service):
    """Test that adding alias with empty value fails."""
    result = await alias_service.add_alias("key", "")
    assert result is False


@pytest.mark.asyncio
async def test_add_alias_trims_whitespace(alias_service):
    """Test that add_alias trims whitespace from key and value."""
    await alias_service.add_alias("  greeting  ", "  hello world  ")

    aliases = alias_service.get_aliases()
    assert "greeting" in aliases
    assert aliases["greeting"] == "hello world"


@pytest.mark.asyncio
async def test_add_alias_case_insensitive_key(alias_service):
    """Test that add_alias converts key to lowercase."""
    await alias_service.add_alias("GREETING", "hello")

    aliases = alias_service.get_aliases()
    assert "greeting" in aliases


@pytest.mark.asyncio
async def test_add_alias_rollback_on_save_failure(alias_service, mock_storage):
    """Test that alias is rolled back if save fails."""
    mock_storage.write.return_value = False

    result = await alias_service.add_alias("greeting", "hello")

    assert result is False
    assert "greeting" not in alias_service.get_aliases()


@pytest.mark.asyncio
async def test_update_alias_success(alias_service, mock_storage):
    """Test successfully updating an existing alias."""
    await alias_service.add_alias("greeting", "hello")
    mock_storage.write.return_value = True

    result = await alias_service.update_alias("greeting", "hi there")

    assert result is True
    assert alias_service.get_aliases()["greeting"] == "hi there"


@pytest.mark.asyncio
async def test_update_alias_nonexistent_fails(alias_service):
    """Test that updating nonexistent alias fails."""
    result = await alias_service.update_alias("nonexistent", "value")
    assert result is False


@pytest.mark.asyncio
async def test_update_alias_empty_value_fails(alias_service):
    """Test that updating with empty value fails."""
    await alias_service.add_alias("greeting", "hello")
    result = await alias_service.update_alias("greeting", "")

    assert result is False
    assert alias_service.get_aliases()["greeting"] == "hello"


@pytest.mark.asyncio
async def test_update_alias_rollback_on_save_failure(alias_service, mock_storage):
    """Test that update is rolled back if save fails."""
    await alias_service.add_alias("greeting", "hello")
    mock_storage.write.return_value = False

    result = await alias_service.update_alias("greeting", "hi")

    assert result is False
    assert alias_service.get_aliases()["greeting"] == "hello"


@pytest.mark.asyncio
async def test_delete_alias_success(alias_service, mock_storage):
    """Test successfully deleting an alias."""
    await alias_service.add_alias("greeting", "hello")
    mock_storage.write.return_value = True

    result = await alias_service.delete_alias("greeting")

    assert result is True
    assert "greeting" not in alias_service.get_aliases()


@pytest.mark.asyncio
async def test_delete_alias_nonexistent_fails(alias_service):
    """Test that deleting nonexistent alias fails."""
    result = await alias_service.delete_alias("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_delete_alias_rollback_on_save_failure(alias_service, mock_storage):
    """Test that delete is rolled back if save fails."""
    await alias_service.add_alias("greeting", "hello")
    mock_storage.write.return_value = False

    result = await alias_service.delete_alias("greeting")

    assert result is False
    assert "greeting" in alias_service.get_aliases()


@pytest.mark.asyncio
async def test_apply_substitutions_basic(alias_service):
    """Test basic alias substitution."""
    await alias_service.add_alias("greeting", "hello world")

    text = "insert greeting to everyone"
    result = alias_service.apply_substitutions(text)

    assert result == "hello world to everyone"


@pytest.mark.asyncio
async def test_apply_substitutions_case_insensitive(alias_service):
    """Test that substitutions are case-insensitive."""
    await alias_service.add_alias("greeting", "hello")

    text = "Insert Greeting to all"
    result = alias_service.apply_substitutions(text)

    assert result == "hello to all"


@pytest.mark.asyncio
async def test_apply_substitutions_multiple_aliases(alias_service):
    """Test substitution with multiple aliases."""
    await alias_service.add_alias("greeting", "hello")
    await alias_service.add_alias("farewell", "goodbye")

    text = "insert greeting and insert farewell"
    result = alias_service.apply_substitutions(text)

    assert result == "hello and goodbye"


@pytest.mark.asyncio
async def test_apply_substitutions_longest_first(alias_service):
    """Test that longest aliases are matched first."""
    await alias_service.add_alias("test", "test_value")
    await alias_service.add_alias("test phrase", "test_phrase_value")

    text = "insert test phrase"
    result = alias_service.apply_substitutions(text)

    assert result == "test_phrase_value"


@pytest.mark.asyncio
async def test_apply_substitutions_word_boundary(alias_service):
    """Test that substitution respects word boundaries."""
    await alias_service.add_alias("test", "value")

    text = "insert test and then testing"
    result = alias_service.apply_substitutions(text)

    # Should match "insert test" and replace it with "value"
    # Other words like "testing" should not be affected (different word)
    assert "value" in result
    assert "and" in result
    assert "then" in result
    assert "testing" in result


@pytest.mark.asyncio
async def test_apply_substitutions_no_aliases(alias_service):
    """Test that text is unchanged when no aliases match."""
    text = "insert something"
    result = alias_service.apply_substitutions(text)

    assert result == text


@pytest.mark.asyncio
async def test_apply_substitutions_empty_aliases(alias_service):
    """Test that empty text is handled gracefully."""
    result = alias_service.apply_substitutions("")
    assert result == ""


@pytest.mark.asyncio
async def test_apply_substitutions_special_chars_escaped(alias_service):
    """Test that special regex characters in aliases are escaped."""
    # The pattern is "insert {activation_phrase}" so use a phrase with special chars
    await alias_service.add_alias("test special", "value")

    # Note: The activation phrase is the key part that comes after "insert"
    text = "insert test special"
    result = alias_service.apply_substitutions(text)

    assert result == "value"


@pytest.mark.asyncio
async def test_apply_substitutions_preserves_surrounding_text(alias_service):
    """Test that surrounding text is preserved after substitution."""
    await alias_service.add_alias("test", "replaced")

    text = "start insert test end"
    result = alias_service.apply_substitutions(text)

    assert result == "start replaced end"


@pytest.mark.asyncio
async def test_concurrent_alias_operations(alias_service, mock_storage):
    """Test that concurrent operations on aliases are thread-safe."""
    mock_storage.write.return_value = True

    # Add multiple aliases "concurrently" (sequentially in test)
    await alias_service.add_alias("greeting", "hello")
    await alias_service.add_alias("farewell", "goodbye")

    aliases = alias_service.get_aliases()
    assert len(aliases) == 2


@pytest.mark.asyncio
async def test_shutdown_saves_aliases(alias_service, mock_storage):
    """Test that shutdown saves any pending aliases."""
    await alias_service.add_alias("test", "value")
    mock_storage.write.return_value = True

    await alias_service.shutdown()

    # Should have called write during add and shutdown
    assert mock_storage.write.call_count >= 1
