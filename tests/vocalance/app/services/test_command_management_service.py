from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.events.command_management_events import (
    AddCustomCommandEvent,
    DeleteCustomCommandEvent,
    RequestCommandMappingsEvent,
    ResetCommandsToDefaultsEvent,
    UpdateCommandPhraseEvent,
)
from vocalance.app.services.command_management_service import CommandManagementService
from vocalance.app.services.storage.storage_models import CommandsData


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    event_bus = Mock()
    event_bus.subscribe = Mock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    storage = Mock()
    storage.read = AsyncMock()
    storage.write = AsyncMock()
    return storage


@pytest.fixture
def mock_protected_terms_validator():
    """Mock protected terms validator."""
    validator = Mock()
    validator.validate_term = AsyncMock()
    return validator


@pytest.fixture
def mock_action_map_provider():
    """Mock action map provider."""
    provider = Mock()
    provider.get_action_map = AsyncMock()
    return provider


@pytest.fixture
def app_config():
    """Create application configuration."""
    return GlobalAppConfig()


@pytest.fixture
def command_management_service(mock_event_bus, app_config, mock_storage, mock_protected_terms_validator, mock_action_map_provider):
    """Create CommandManagementService instance."""
    return CommandManagementService(
        event_bus=mock_event_bus,
        app_config=app_config,
        storage=mock_storage,
        protected_terms_validator=mock_protected_terms_validator,
        action_map_provider=mock_action_map_provider,
    )


@pytest.mark.asyncio
async def test_setup_subscriptions(command_management_service, mock_event_bus):
    """Test event subscriptions are configured."""
    command_management_service.setup_subscriptions()

    # Should subscribe to multiple event types
    assert mock_event_bus.subscribe.call_count >= 4


@pytest.mark.asyncio
async def test_add_custom_command_success(
    command_management_service, mock_storage, mock_protected_terms_validator, mock_event_bus
):
    """Test successfully adding custom command."""
    mock_protected_terms_validator.validate_term.return_value = (True, None)
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})
    mock_storage.write.return_value = True

    custom_cmd = AutomationCommand(
        command_key="my command",
        action_type="hotkey",
        action_value="ctrl+m",
        is_custom=True,
        short_description="Test",
        long_description="Test command",
    )

    event = AddCustomCommandEvent(command=custom_cmd)
    await command_management_service._handle_add_custom_command(event)

    # Should write to storage
    mock_storage.write.assert_called_once()
    # Should publish success event
    assert mock_event_bus.publish.call_count >= 1


@pytest.mark.asyncio
async def test_add_custom_command_validation_error(
    command_management_service, mock_protected_terms_validator, mock_storage, mock_event_bus
):
    """Test adding custom command with validation error."""
    mock_protected_terms_validator.validate_term.return_value = (False, "Protected term")
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    custom_cmd = AutomationCommand(
        command_key="copy",
        action_type="hotkey",
        action_value="ctrl+c",
        is_custom=True,
        short_description="Test",
        long_description="Test",
    )

    event = AddCustomCommandEvent(command=custom_cmd)
    await command_management_service._handle_add_custom_command(event)

    # Should not write to storage
    mock_storage.write.assert_not_called()
    # Should publish error event
    mock_event_bus.publish.assert_called()


@pytest.mark.asyncio
async def test_update_command_phrase_custom_command(
    command_management_service, mock_storage, mock_protected_terms_validator, mock_event_bus
):
    """Test updating phrase for custom command."""
    custom_cmd = AutomationCommand(
        command_key="old phrase",
        action_type="hotkey",
        action_value="ctrl+o",
        is_custom=True,
        short_description="Test",
        long_description="Test",
    )
    mock_storage.read.return_value = CommandsData(custom_commands={"old phrase": custom_cmd}, phrase_overrides={})
    mock_protected_terms_validator.validate_term.return_value = (True, None)
    mock_storage.write.return_value = True

    event = UpdateCommandPhraseEvent(old_command_phrase="old phrase", new_command_phrase="new phrase")
    await command_management_service._handle_update_command_phrase(event)

    # Should write updated data
    mock_storage.write.assert_called_once()
    written_data = mock_storage.write.call_args[1]["data"]
    assert "new phrase" in written_data.custom_commands
    assert "old phrase" not in written_data.custom_commands


@pytest.mark.asyncio
async def test_update_command_phrase_validation_error(
    command_management_service, mock_storage, mock_protected_terms_validator, mock_event_bus
):
    """Test updating phrase with validation error."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})
    mock_protected_terms_validator.validate_term.return_value = (False, "Protected term")

    event = UpdateCommandPhraseEvent(old_command_phrase="old", new_command_phrase="copy")
    await command_management_service._handle_update_command_phrase(event)

    # Should not write
    mock_storage.write.assert_not_called()
    # Should publish error
    mock_event_bus.publish.assert_called()


@pytest.mark.asyncio
async def test_delete_custom_command(command_management_service, mock_storage, mock_event_bus):
    """Test deleting custom command."""
    custom_cmd = AutomationCommand(
        command_key="delete me",
        action_type="hotkey",
        action_value="ctrl+d",
        is_custom=True,
        short_description="Test",
        long_description="Test",
    )
    mock_storage.read.return_value = CommandsData(custom_commands={"delete me": custom_cmd}, phrase_overrides={})
    mock_storage.write.return_value = True

    event = DeleteCustomCommandEvent(command=custom_cmd)
    await command_management_service._handle_delete_custom_command(event)

    # Should write updated data
    mock_storage.write.assert_called_once()
    written_data = mock_storage.write.call_args[1]["data"]
    assert "delete me" not in written_data.custom_commands


@pytest.mark.asyncio
async def test_delete_nonexistent_command(command_management_service, mock_storage, mock_event_bus):
    """Test deleting command that doesn't exist."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    custom_cmd = AutomationCommand(
        command_key="nonexistent",
        action_type="hotkey",
        action_value="ctrl+n",
        is_custom=True,
        short_description="Test",
        long_description="Test",
    )

    event = DeleteCustomCommandEvent(command=custom_cmd)
    await command_management_service._handle_delete_custom_command(event)

    # Should not call write (command doesn't exist)
    mock_storage.write.assert_not_called()


@pytest.mark.asyncio
async def test_request_command_mappings(command_management_service, mock_storage, mock_event_bus):
    """Test requesting command mappings."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    event = RequestCommandMappingsEvent()
    await command_management_service._handle_request_command_mappings(event)

    # Should publish mappings response
    mock_event_bus.publish.assert_called_once()


@pytest.mark.asyncio
async def test_reset_to_defaults(command_management_service, mock_storage, mock_event_bus):
    """Test resetting commands to defaults."""
    mock_storage.write.return_value = True

    event = ResetCommandsToDefaultsEvent()
    await command_management_service._handle_reset_to_defaults(event)

    # Should write empty CommandsData
    mock_storage.write.assert_called_once()
    written_data = mock_storage.write.call_args[1]["data"]
    assert isinstance(written_data, CommandsData)
    assert len(written_data.custom_commands) == 0
    assert len(written_data.phrase_overrides) == 0


@pytest.mark.asyncio
async def test_get_command_mappings_includes_custom_and_defaults(command_management_service, mock_storage):
    """Test getting all command mappings."""
    custom_cmd = AutomationCommand(
        command_key="custom",
        action_type="hotkey",
        action_value="ctrl+custom",
        is_custom=True,
        short_description="Custom",
        long_description="Custom command",
    )
    mock_storage.read.return_value = CommandsData(custom_commands={"custom": custom_cmd}, phrase_overrides={})

    mappings = await command_management_service.get_command_mappings()

    # Should include custom commands
    custom_commands = [m for m in mappings if m.is_custom]
    assert len(custom_commands) >= 1

    # Should include default commands
    default_commands = [m for m in mappings if not m.is_custom]
    assert len(default_commands) > 0


@pytest.mark.asyncio
async def test_get_command_mappings_applies_overrides(command_management_service, mock_storage):
    """Test command mappings apply phrase overrides."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={"copy": "copy that"})

    mappings = await command_management_service.get_command_mappings()

    # Should have override phrase, not original
    mapping_phrases = [m.command_key for m in mappings]
    assert "copy that" in mapping_phrases
    assert "copy" not in mapping_phrases


@pytest.mark.asyncio
async def test_validate_command_phrase_empty(command_management_service, mock_protected_terms_validator, mock_action_map_provider):
    """Test validation rejects empty phrases."""
    # Empty phrase should be caught before calling validator, but set up mock anyway
    mock_protected_terms_validator.validate_term.return_value = (False, "Term cannot be empty")
    mock_action_map_provider.get_action_map.return_value = {}

    error = await command_management_service._validate_command_phrase("")
    assert error is not None
    assert "empty" in error.lower()


@pytest.mark.asyncio
async def test_validate_command_phrase_protected(
    command_management_service, mock_protected_terms_validator, mock_action_map_provider
):
    """Test validation rejects protected terms."""
    mock_protected_terms_validator.validate_term.return_value = (False, "Protected term")
    mock_action_map_provider.get_action_map.return_value = {}

    error = await command_management_service._validate_command_phrase("copy")
    assert error is not None


@pytest.mark.asyncio
async def test_validate_command_phrase_already_exists(
    command_management_service, mock_protected_terms_validator, mock_action_map_provider
):
    """Test validation rejects phrases that already exist."""
    mock_protected_terms_validator.validate_term.return_value = (True, None)
    mock_action_map_provider.get_action_map.return_value = {
        "existing command": AutomationCommand(
            command_key="existing command",
            action_type="hotkey",
            action_value="ctrl+e",
            is_custom=False,
            short_description="Existing",
            long_description="Existing",
        )
    }

    error = await command_management_service._validate_command_phrase("existing command")
    assert error is not None
    assert "already exists" in error.lower()


@pytest.mark.asyncio
async def test_validate_command_phrase_with_exclude(
    command_management_service, mock_protected_terms_validator, mock_action_map_provider
):
    """Test validation allows phrase when it matches exclude."""
    mock_protected_terms_validator.validate_term.return_value = (True, None)
    mock_action_map_provider.get_action_map.return_value = {}

    error = await command_management_service._validate_command_phrase("same phrase", exclude_phrase="same phrase")
    assert error is None


@pytest.mark.asyncio
async def test_update_default_command_phrase(command_management_service, mock_storage, mock_protected_terms_validator):
    """Test updating phrase for default command creates override."""
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})
    mock_protected_terms_validator.validate_term.return_value = (True, None)
    mock_storage.write.return_value = True

    event = UpdateCommandPhraseEvent(old_command_phrase="copy", new_command_phrase="copy that")
    await command_management_service._handle_update_command_phrase(event)

    # Should create phrase override
    mock_storage.write.assert_called_once()
    written_data = mock_storage.write.call_args[1]["data"]
    assert "copy" in written_data.phrase_overrides
    assert written_data.phrase_overrides["copy"] == "copy that"


@pytest.mark.asyncio
async def test_custom_command_functional_group_renamed(command_management_service, mock_storage):
    """Test custom commands with 'Other' group are renamed to 'Custom'."""
    custom_cmd = AutomationCommand(
        command_key="test",
        action_type="hotkey",
        action_value="ctrl+t",
        is_custom=True,
        short_description="Test",
        long_description="Test",
        functional_group="Other",
    )
    mock_storage.read.return_value = CommandsData(custom_commands={"test": custom_cmd}, phrase_overrides={})

    mappings = await command_management_service.get_command_mappings()

    # Custom command with 'Other' should be renamed to 'Custom'
    test_cmd = next((m for m in mappings if m.command_key == "test"), None)
    assert test_cmd is not None
    assert test_cmd.functional_group == "Custom"
