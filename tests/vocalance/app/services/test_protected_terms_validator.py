from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.storage.storage_models import MarksData, SoundMappingsData


@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    storage = Mock()
    storage.read = AsyncMock()
    return storage


@pytest.fixture
def app_config():
    """Create application configuration for testing."""
    return GlobalAppConfig()


@pytest.fixture
def validator(app_config, mock_storage):
    """Create ProtectedTermsValidator instance."""
    return ProtectedTermsValidator(config=app_config, storage=mock_storage)


@pytest.mark.asyncio
async def test_get_protected_terms_includes_default_commands(validator, mock_storage):
    """Test protected terms include default automation commands."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    protected = await validator.get_all_protected_terms()

    # Should include common default commands (using commands that actually exist)
    assert "copy" in protected
    assert "paste" in protected
    assert "back" in protected


@pytest.mark.asyncio
async def test_get_protected_terms_includes_grid_phrase(validator, mock_storage, app_config):
    """Test protected terms include grid show phrase."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    protected = await validator.get_all_protected_terms()

    assert app_config.grid.show_grid_phrase.lower() in protected


@pytest.mark.asyncio
async def test_get_protected_terms_includes_mark_triggers(validator, mock_storage, app_config):
    """Test protected terms include mark system triggers."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    protected = await validator.get_all_protected_terms()

    # Mark triggers should be protected
    assert app_config.mark.triggers.create_mark.lower() in protected
    assert app_config.mark.triggers.delete_mark.lower() in protected


@pytest.mark.asyncio
async def test_get_protected_terms_includes_dictation_triggers(validator, mock_storage, app_config):
    """Test protected terms include dictation triggers."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    protected = await validator.get_all_protected_terms()

    # Dictation triggers should be protected
    assert app_config.dictation.start_trigger.lower() in protected
    assert app_config.dictation.stop_trigger.lower() in protected
    assert app_config.dictation.type_trigger.lower() in protected


@pytest.mark.asyncio
async def test_get_protected_terms_includes_existing_marks(validator, mock_storage):
    """Test protected terms include existing mark names."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={"mark1": {"x": 100, "y": 200}, "mark2": {"x": 300, "y": 400}})
        if model_type == MarksData
        else SoundMappingsData(mappings={})
    )

    protected = await validator.get_all_protected_terms()

    assert "mark1" in protected
    assert "mark2" in protected


@pytest.mark.asyncio
async def test_get_protected_terms_includes_sound_mappings(validator, mock_storage):
    """Test protected terms include sound mapping names."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={})
        if model_type == MarksData
        else SoundMappingsData(mappings={"lip pop": "copy", "tongue click": "paste"})
    )

    protected = await validator.get_all_protected_terms()

    assert "lip pop" in protected
    assert "tongue click" in protected


@pytest.mark.asyncio
async def test_get_protected_terms_handles_storage_error(validator, mock_storage):
    """Test protected terms handles storage read errors gracefully."""
    mock_storage.read.side_effect = Exception("Storage error")

    protected = await validator.get_all_protected_terms()

    # Should still return system protected terms
    assert "copy" in protected
    assert len(protected) > 0


@pytest.mark.asyncio
async def test_is_term_protected_true(validator, mock_storage):
    """Test checking if term is protected returns True."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    is_protected = await validator.is_term_protected("copy")

    assert is_protected is True


@pytest.mark.asyncio
async def test_is_term_protected_false(validator, mock_storage):
    """Test checking if term is protected returns False."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    is_protected = await validator.is_term_protected("my custom command")

    assert is_protected is False


@pytest.mark.asyncio
async def test_is_term_protected_case_insensitive(validator, mock_storage):
    """Test term protection check is case insensitive."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    assert await validator.is_term_protected("COPY") is True
    assert await validator.is_term_protected("Copy") is True
    assert await validator.is_term_protected("copy") is True


@pytest.mark.asyncio
async def test_validate_term_valid(validator, mock_storage):
    """Test validating a valid term."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    is_valid, error = await validator.validate_term("my custom command")

    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_validate_term_empty_string(validator, mock_storage):
    """Test validating empty string returns error."""
    is_valid, error = await validator.validate_term("")

    assert is_valid is False
    assert error == "Term cannot be empty"


@pytest.mark.asyncio
async def test_validate_term_whitespace_only(validator, mock_storage):
    """Test validating whitespace-only string returns error."""
    is_valid, error = await validator.validate_term("   ")

    assert is_valid is False
    assert error == "Term cannot be empty"


@pytest.mark.asyncio
async def test_validate_term_protected(validator, mock_storage):
    """Test validating protected term returns error."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    is_valid, error = await validator.validate_term("copy")

    assert is_valid is False
    assert "protected term" in error.lower()


@pytest.mark.asyncio
async def test_validate_term_with_exclude(validator, mock_storage):
    """Test validating term with exclude allows self-reference."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={"my mark": {"x": 100, "y": 200}}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    # Should be invalid without exclude
    is_valid, error = await validator.validate_term("my mark")
    assert is_valid is False

    # Should be valid with exclude
    is_valid, error = await validator.validate_term("my mark", exclude_term="my mark")
    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_cache_behavior(validator, mock_storage):
    """Test protected terms are cached and reused."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    # First call should read from storage
    protected1 = await validator.get_all_protected_terms()
    call_count_first = mock_storage.read.call_count

    # Second call should use cache
    protected2 = await validator.get_all_protected_terms()
    call_count_second = mock_storage.read.call_count

    assert protected1 == protected2
    assert call_count_second == call_count_first


@pytest.mark.asyncio
async def test_invalidate_cache(validator, mock_storage):
    """Test cache invalidation forces reload."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    # First call
    await validator.get_all_protected_terms()
    call_count_first = mock_storage.read.call_count

    # Invalidate cache
    validator.invalidate_cache()

    # Next call should read from storage again
    await validator.get_all_protected_terms()
    call_count_after_invalidate = mock_storage.read.call_count

    assert call_count_after_invalidate > call_count_first


@pytest.mark.asyncio
async def test_protected_terms_normalized(validator, mock_storage):
    """Test all protected terms are normalized to lowercase."""
    mock_storage.read.side_effect = lambda model_type: (
        MarksData(marks={"My Mark": {"x": 100, "y": 200}}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    protected = await validator.get_all_protected_terms()

    # All terms should be lowercase
    assert all(term == term.lower() for term in protected)
    assert "my mark" in protected
