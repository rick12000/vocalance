import pytest

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.events.command_management_events import CommandMappingsUpdatedEvent, CommandValidationErrorEvent
from vocalance.app.services.storage.storage_models import CommandsData


def _custom_command(command_key: str, action_value: str = "ctrl+m") -> AutomationCommand:
    return AutomationCommand(
        command_key=command_key,
        action_type="hotkey",
        action_value=action_value,
        is_custom=True,
        short_description="Test",
        long_description="Test command",
    )


@pytest.mark.asyncio
async def test_add_command_persists_and_announces(command_management_service, command_storage, mock_event_bus):
    command_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    success, err = await command_management_service.add_command(_custom_command("my command"))

    assert success is True
    assert err == ""
    command_storage.write.assert_awaited_once()
    written = command_storage.write.await_args.kwargs["data"]
    assert "my command" in written.custom_commands
    assert any(isinstance(c.args[0], CommandMappingsUpdatedEvent) for c in mock_event_bus.publish.await_args_list)


@pytest.mark.asyncio
async def test_add_command_blocked_by_validation(
    command_management_service, command_storage, mock_protected_terms_validator, mock_event_bus
):
    mock_protected_terms_validator.validate_term.return_value = (False, "Protected term")
    command_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    success, err = await command_management_service.add_command(_custom_command("copy", "ctrl+c"))

    assert success is False
    assert err
    command_storage.write.assert_not_awaited()
    assert any(isinstance(c.args[0], CommandValidationErrorEvent) for c in mock_event_bus.publish.await_args_list)


@pytest.mark.asyncio
async def test_update_renames_custom_command(command_management_service, command_storage):
    cmd = _custom_command("old phrase", "ctrl+o")
    command_storage.read.return_value = CommandsData(custom_commands={"old phrase": cmd}, phrase_overrides={})

    success, _ = await command_management_service.update_command_phrase("old phrase", "new phrase")

    assert success is True
    written = command_storage.write.await_args.kwargs["data"]
    assert "new phrase" in written.custom_commands
    assert "old phrase" not in written.custom_commands


@pytest.mark.asyncio
async def test_update_sets_registry_phrase_override(command_management_service, command_storage):
    command_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    success, _ = await command_management_service.update_command_phrase("copy", "copy that")

    assert success is True
    written = command_storage.write.await_args.kwargs["data"]
    assert written.phrase_overrides["copy"] == "copy that"


@pytest.mark.asyncio
async def test_update_unknown_phrase_errors(command_management_service, command_storage, mock_event_bus):
    command_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    success, err = await command_management_service.update_command_phrase("not a real phrase", "renamed")

    assert success is False
    assert "could not find" in err.lower()
    command_storage.write.assert_not_awaited()
    assert any(isinstance(c.args[0], CommandValidationErrorEvent) for c in mock_event_bus.publish.await_args_list)


@pytest.mark.asyncio
async def test_delete_removes_custom_command(command_management_service, command_storage):
    cmd = _custom_command("delete me", "ctrl+d")
    command_storage.read.return_value = CommandsData(custom_commands={"delete me": cmd}, phrase_overrides={})

    success, _ = await command_management_service.delete_command(cmd)

    assert success is True
    written = command_storage.write.await_args.kwargs["data"]
    assert "delete me" not in written.custom_commands


@pytest.mark.asyncio
async def test_delete_missing_command_is_noop_success(command_management_service, command_storage):
    command_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={})

    success, _ = await command_management_service.delete_command(_custom_command("nonexistent", "ctrl+n"))

    assert success is True
    command_storage.write.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_to_defaults_writes_empty_commands(command_management_service, command_storage):
    command_storage.read.return_value = CommandsData()

    success, _ = await command_management_service.reset_to_defaults()

    assert success is True
    written = command_storage.write.await_args.kwargs["data"]
    assert isinstance(written, CommandsData)
    assert len(written.custom_commands) == 0
    assert len(written.phrase_overrides) == 0


@pytest.mark.asyncio
async def test_get_command_mappings_merges_custom_and_defaults(command_management_service, command_storage):
    cmd = _custom_command("custom", "ctrl+k")
    command_storage.read.return_value = CommandsData(custom_commands={"custom": cmd}, phrase_overrides={})

    mappings = await command_management_service.get_command_mappings()

    assert any(m.is_custom and m.command_key == "custom" for m in mappings)
    assert any(not m.is_custom for m in mappings)


@pytest.mark.asyncio
async def test_get_command_mappings_applies_phrase_override(command_management_service, command_storage):
    command_storage.read.return_value = CommandsData(custom_commands={}, phrase_overrides={"copy": "copy that"})

    phrases = [m.command_key for m in await command_management_service.get_command_mappings()]

    assert "copy that" in phrases
    assert "copy" not in phrases


@pytest.mark.asyncio
async def test_validate_command_phrase_detects_collision(command_management_service, command_storage):
    existing = _custom_command("existing command", "ctrl+e")
    command_storage.read.return_value = CommandsData(custom_commands={"existing command": existing}, phrase_overrides={})

    error = await command_management_service.validate_command_phrase("existing command")

    assert error is not None
    assert "already exists" in error.lower()


@pytest.mark.asyncio
async def test_validate_command_phrase_allows_excluded_self(command_management_service):
    assert await command_management_service.validate_command_phrase("same phrase", exclude_phrase="same phrase") is None
