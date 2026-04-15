from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.services.commands.management import CommandManagementService
from vocalance.app.services.storage.storage_models import CommandsData


@pytest.fixture
def mock_event_bus():
    event_bus = Mock()
    event_bus.subscribe = Mock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_storage():
    storage = Mock()
    storage.read = AsyncMock()
    storage.write = AsyncMock()
    return storage


@pytest.fixture
def mock_protected_terms_validator():
    validator = Mock()
    validator.validate_term = AsyncMock()
    return validator


@pytest.fixture
def command_management_service(mock_event_bus, mock_storage, mock_protected_terms_validator):
    return CommandManagementService(
        event_bus=mock_event_bus,
        storage=mock_storage,
        protected_terms_validator=mock_protected_terms_validator,
    )


@pytest.mark.asyncio
async def test_registers_ui_request_subscriptions(command_management_service, mock_event_bus):
    """CommandManagementService subscribes to UI request events on the bus."""
    assert mock_event_bus.subscribe.call_count == 1


@pytest.mark.asyncio
async def test_add_command_success(command_management_service, mock_storage, mock_protected_terms_validator, mock_event_bus):
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

    success, err = await command_management_service.add_command(custom_cmd)

    assert success is True
    assert err == ""
    mock_storage.write.assert_called_once()
    assert mock_event_bus.publish.call_count >= 1


@pytest.mark.asyncio
async def test_add_command_validation_error(
    command_management_service, mock_protected_terms_validator, mock_storage, mock_event_bus
):
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

    success, err = await command_management_service.add_command(custom_cmd)

    assert success is False
    assert err != ""
    mock_storage.write.assert_not_called()
    mock_event_bus.publish.assert_called()


@pytest.mark.asyncio
async def test_update_command_phrase_custom_command(
    command_management_service, mock_storage, mock_protected_terms_validator, mock_event_bus
):
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

    success, err = await command_management_service.update_command_phrase("old phrase", "new phrase")

    assert success is True
    mock_storage.write.assert_called_once()
    written_data = mock_storage.write.call_args[1]["data"]
    assert "new phrase" in written_data.custom_commands
    assert "old phrase" not in written_data.custom_commands


@pytest.mark.asyncio
async def test_update_command_phrase_validation_error(
    command_management_service, mock_storage, mock_protected_terms_validator, mock_event_bus
):
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})
    mock_protected_terms_validator.validate_term.return_value = (False, "Protected term")

    success, err = await command_management_service.update_command_phrase("old", "copy")

    assert success is False
    mock_storage.write.assert_not_called()
    mock_event_bus.publish.assert_called()


@pytest.mark.asyncio
async def test_delete_command(command_management_service, mock_storage, mock_event_bus):
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

    success, err = await command_management_service.delete_command(custom_cmd)

    assert success is True
    mock_storage.write.assert_called_once()
    written_data = mock_storage.write.call_args[1]["data"]
    assert "delete me" not in written_data.custom_commands


@pytest.mark.asyncio
async def test_delete_nonexistent_command(command_management_service, mock_storage, mock_event_bus):
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    custom_cmd = AutomationCommand(
        command_key="nonexistent",
        action_type="hotkey",
        action_value="ctrl+n",
        is_custom=True,
        short_description="Test",
        long_description="Test",
    )

    success, err = await command_management_service.delete_command(custom_cmd)

    assert success is True
    mock_storage.write.assert_not_called()


@pytest.mark.asyncio
async def test_reset_to_defaults(command_management_service, mock_storage, mock_event_bus):
    mock_storage.read.return_value = CommandsData()
    mock_storage.write.return_value = True

    success, err = await command_management_service.reset_to_defaults()

    assert success is True
    mock_storage.write.assert_called_once()
    written_data = mock_storage.write.call_args[1]["data"]
    assert isinstance(written_data, CommandsData)
    assert len(written_data.custom_commands) == 0
    assert len(written_data.phrase_overrides) == 0


@pytest.mark.asyncio
async def test_get_command_mappings_includes_custom_and_defaults(command_management_service, mock_storage):
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

    custom_commands = [m for m in mappings if m.is_custom]
    assert len(custom_commands) >= 1
    default_commands = [m for m in mappings if not m.is_custom]
    assert len(default_commands) > 0


@pytest.mark.asyncio
async def test_get_command_mappings_applies_overrides(command_management_service, mock_storage):
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={"copy": "copy that"})

    mappings = await command_management_service.get_command_mappings()

    mapping_phrases = [m.command_key for m in mappings]
    assert "copy that" in mapping_phrases
    assert "copy" not in mapping_phrases


@pytest.mark.asyncio
async def test_validate_command_phrase_empty(command_management_service, mock_protected_terms_validator):
    mock_protected_terms_validator.validate_term.return_value = (False, "Term cannot be empty")

    error = await command_management_service.validate_command_phrase("")
    assert error is not None
    assert "empty" in error.lower()


@pytest.mark.asyncio
async def test_validate_command_phrase_protected(command_management_service, mock_protected_terms_validator):
    mock_protected_terms_validator.validate_term.return_value = (False, "Protected term")

    error = await command_management_service.validate_command_phrase("copy")
    assert error is not None


@pytest.mark.asyncio
async def test_validate_command_phrase_already_exists(command_management_service, mock_protected_terms_validator, mock_storage):
    mock_protected_terms_validator.validate_term.return_value = (True, None)
    existing = AutomationCommand(
        command_key="existing command",
        action_type="hotkey",
        action_value="ctrl+e",
        is_custom=True,
        short_description="Existing",
        long_description="Existing",
    )
    mock_storage.read.return_value = CommandsData(
        custom_commands={"existing command": existing},
        phrase_overrides={},
    )

    error = await command_management_service.validate_command_phrase("existing command")
    assert error is not None
    assert "already exists" in error.lower()


@pytest.mark.asyncio
async def test_validate_command_phrase_with_exclude(command_management_service, mock_protected_terms_validator):
    mock_protected_terms_validator.validate_term.return_value = (True, None)

    error = await command_management_service.validate_command_phrase("same phrase", exclude_phrase="same phrase")
    assert error is None


@pytest.mark.asyncio
async def test_update_default_command_phrase(command_management_service, mock_storage, mock_protected_terms_validator):
    mock_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})
    mock_protected_terms_validator.validate_term.return_value = (True, None)
    mock_storage.write.return_value = True

    success, err = await command_management_service.update_command_phrase("copy", "copy that")

    assert success is True
    mock_storage.write.assert_called_once()
    written_data = mock_storage.write.call_args[1]["data"]
    assert "copy" in written_data.phrase_overrides
    assert written_data.phrase_overrides["copy"] == "copy that"


@pytest.mark.asyncio
async def test_custom_command_functional_group_renamed(command_management_service, mock_storage):
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

    test_cmd = next((m for m in mappings if m.command_key == "test"), None)
    assert test_cmd is not None
    assert test_cmd.functional_group == "Custom"
