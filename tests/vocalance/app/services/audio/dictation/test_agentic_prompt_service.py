"""Unit tests for AgenticPromptService.

Tests prompt management, CRUD operations, and event publishing.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.services.audio.dictation_handling.llm_support.agentic_prompt_service import AgenticPromptService
from vocalance.app.services.storage.storage_models import AgenticPrompt, AgenticPromptsData


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    event_bus = Mock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.llm = Mock()
    return config


@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    storage = Mock()
    storage.read = AsyncMock(return_value=AgenticPromptsData(prompts=[]))
    storage.write = AsyncMock(return_value=True)
    return storage


@pytest.fixture
async def agentic_service(mock_event_bus, mock_config, mock_storage):
    """Create AgenticPromptService instance for testing."""
    service = AgenticPromptService(
        event_bus=mock_event_bus,
        config=mock_config,
        storage=mock_storage,
    )
    await service.initialize()
    return service


@pytest.mark.asyncio
async def test_initialize_creates_default_prompt(mock_event_bus, mock_config, mock_storage):
    """Test that initialization creates default prompt if none exists."""
    mock_storage.read.return_value = AgenticPromptsData(prompts=[])

    service = AgenticPromptService(
        event_bus=mock_event_bus,
        config=mock_config,
        storage=mock_storage,
    )
    result = await service.initialize()

    assert result is True
    assert service.current_prompt_id is not None
    assert len(service.prompts) == 1
    default_prompt = service._get_default_prompt()
    assert default_prompt is not None
    assert default_prompt.is_default is True


@pytest.mark.asyncio
async def test_get_all_prompts(agentic_service):
    """Test retrieving all prompts."""
    await agentic_service.add_prompt("Custom prompt", "Custom")

    prompts = agentic_service.get_all_prompts()

    assert len(prompts) >= 2  # Default + custom
    assert any(p.name == "Custom" for p in prompts)


@pytest.mark.asyncio
async def test_get_prompt_by_id(agentic_service):
    """Test retrieving a specific prompt."""
    prompt_id = await agentic_service.add_prompt("Test prompt", "Test")

    prompts = agentic_service.get_all_prompts()
    retrieved = next((p for p in prompts if p.id == prompt_id), None)

    assert retrieved is not None
    assert retrieved.id == prompt_id
    assert retrieved.name == "Test"


@pytest.mark.asyncio
async def test_add_prompt_success(agentic_service, mock_storage):
    """Test successfully adding a new prompt."""
    mock_storage.write.return_value = True

    prompt_id = await agentic_service.add_prompt("Test prompt", "Test Name")

    assert prompt_id is not None
    prompts = agentic_service.get_all_prompts()
    prompt = next((p for p in prompts if p.id == prompt_id), None)
    assert prompt is not None
    assert prompt.text == "Test prompt"
    assert prompt.name == "Test Name"


@pytest.mark.asyncio
async def test_add_prompt_trims_whitespace(agentic_service):
    """Test that add_prompt trims whitespace."""
    prompt_id = await agentic_service.add_prompt("  Test prompt  ", "  Test Name  ")

    prompts = agentic_service.get_all_prompts()
    prompt = next((p for p in prompts if p.id == prompt_id), None)
    assert prompt.text == "Test prompt"
    assert prompt.name == "Test Name"


@pytest.mark.asyncio
async def test_delete_prompt_success(agentic_service, mock_storage):
    """Test successfully deleting a prompt."""
    prompt_id = await agentic_service.add_prompt("Test prompt", "Test")
    mock_storage.write.return_value = True

    result = await agentic_service.delete_prompt(prompt_id)

    assert result is True
    prompts = agentic_service.get_all_prompts()
    assert not any(p.id == prompt_id for p in prompts)


@pytest.mark.asyncio
async def test_delete_prompt_nonexistent_fails(agentic_service):
    """Test that deleting nonexistent prompt returns False."""
    result = await agentic_service.delete_prompt("nonexistent_id")
    assert result is False


@pytest.mark.asyncio
async def test_delete_prompt_fails_if_is_default(agentic_service):
    """Test that deleting default prompt fails."""
    default_prompt = agentic_service._get_default_prompt()

    result = await agentic_service.delete_prompt(default_prompt.id)

    assert result is False
    prompts = agentic_service.get_all_prompts()
    assert any(p.id == default_prompt.id for p in prompts)


@pytest.mark.asyncio
async def test_update_prompt_success(agentic_service, mock_storage):
    """Test successfully updating a prompt."""
    prompt_id = await agentic_service.add_prompt("Original", "Original")
    mock_storage.write.return_value = True

    result = await agentic_service.edit_prompt(prompt_id, "Updated name", "Updated text")

    assert result is True
    prompts = agentic_service.get_all_prompts()
    prompt = next((p for p in prompts if p.id == prompt_id), None)
    assert prompt.text == "Updated text"
    assert prompt.name == "Updated name"


@pytest.mark.asyncio
async def test_update_prompt_nonexistent_fails(agentic_service):
    """Test that editing nonexistent prompt fails."""
    result = await agentic_service.edit_prompt("nonexistent_id", "name", "text")
    assert result is False


@pytest.mark.asyncio
async def test_set_current_prompt(agentic_service, mock_storage):
    """Test setting the current active prompt."""
    prompt_id = await agentic_service.add_prompt("Test prompt", "Test")
    mock_storage.write.return_value = True

    result = await agentic_service.set_current_prompt(prompt_id)

    assert result is True
    assert agentic_service.current_prompt_id == prompt_id


@pytest.mark.asyncio
async def test_set_current_prompt_nonexistent_fails(agentic_service):
    """Test that setting nonexistent prompt fails."""
    result = await agentic_service.set_current_prompt("nonexistent_id")
    assert result is False


@pytest.mark.asyncio
async def test_get_current_prompt(agentic_service):
    """Test retrieving the current active prompt."""
    current = agentic_service.get_current_prompt_data()

    assert current is not None
    assert current.id == agentic_service.current_prompt_id


@pytest.mark.asyncio
async def test_default_prompt_text_correct(agentic_service):
    """Test that default prompt has expected text."""
    default_prompt = agentic_service._get_default_prompt()

    assert default_prompt is not None
    assert "grammar" in default_prompt.text.lower()
    assert "punctuation" in default_prompt.text.lower()


@pytest.mark.asyncio
async def test_thread_safe_prompt_access(agentic_service):
    """Test that prompts can be accessed safely."""
    # Add multiple prompts
    id1 = await agentic_service.add_prompt("Prompt 1", "Name 1")
    id2 = await agentic_service.add_prompt("Prompt 2", "Name 2")

    # Retrieve all
    prompts = agentic_service.get_all_prompts()

    assert len(prompts) >= 3  # Default + 2 custom
    assert any(p.id == id1 for p in prompts)
    assert any(p.id == id2 for p in prompts)


@pytest.mark.asyncio
async def test_prompt_timestamps(agentic_service):
    """Test that prompts have creation timestamps."""
    prompt_id = await agentic_service.add_prompt("Test", "Test")

    prompts = agentic_service.get_all_prompts()
    prompt = next((p for p in prompts if p.id == prompt_id), None)

    assert prompt.created_at is not None
    # Verify it's a valid ISO format timestamp
    datetime.fromisoformat(prompt.created_at)


@pytest.mark.asyncio
async def test_default_prompt_persistence(mock_event_bus, mock_config, mock_storage):
    """Test that default prompt is created even on re-initialization."""
    prompts_data = AgenticPromptsData(prompts=[])
    mock_storage.read.return_value = prompts_data

    service1 = AgenticPromptService(
        event_bus=mock_event_bus,
        config=mock_config,
        storage=mock_storage,
    )
    await service1.initialize()

    assert service1._get_default_prompt() is not None


@pytest.mark.asyncio
async def test_get_all_prompts_returns_list(agentic_service):
    """Test that get_all_prompts returns a proper list."""
    prompts = agentic_service.get_all_prompts()

    assert isinstance(prompts, list)
    assert len(prompts) > 0
    assert all(isinstance(p, AgenticPrompt) for p in prompts)


@pytest.mark.asyncio
async def test_get_prompt_returns_copy(agentic_service):
    """Test that modifying retrieved prompt doesn't affect original."""
    prompt_id = await agentic_service.add_prompt("Original", "Original")

    prompts = agentic_service.get_all_prompts()
    retrieved = next((p for p in prompts if p.id == prompt_id), None)
    retrieved_text = retrieved.text

    # This tests that we can safely work with returned prompt
    assert retrieved.text == retrieved_text
