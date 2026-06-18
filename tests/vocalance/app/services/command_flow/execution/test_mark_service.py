from unittest.mock import AsyncMock, patch

import pytest

from vocalance.app.config.command_types import MarkCreateCommand, MarkDeleteCommand, MarkExecuteCommand
from vocalance.app.events.command_events import MarkCommandParsedEvent
from vocalance.app.events.mark_events import MarkUiRequestEvent


@pytest.mark.asyncio
async def test_add_mark_persists_normalized_label(mark_service):
    created, _ = await mark_service.add_mark("Home", 10, 20)
    assert created is True
    assert await mark_service.get_mark_coordinates_internal("home") == (10, 20)


@pytest.mark.asyncio
async def test_add_mark_rejects_protected_label(mark_service, mock_protected_terms_validator):
    mock_protected_terms_validator.validate_term.return_value = (False, "protected")
    created, msg = await mark_service.add_mark("show grid", 1, 2)
    assert created is False
    assert msg == "protected"
    mark_service.storage.write.assert_not_called()


@pytest.mark.asyncio
async def test_add_mark_rejects_duplicate_label(mark_service):
    await mark_service.add_mark("home", 1, 2)
    created, msg = await mark_service.add_mark("home", 3, 4)
    assert created is False
    assert "already in use" in msg


@pytest.mark.asyncio
async def test_remove_mark_deletes_existing(mark_service):
    await mark_service.add_mark("home", 1, 2)
    assert await mark_service.remove_mark("home") is True
    assert await mark_service.get_mark_coordinates_internal("home") is None


@pytest.mark.asyncio
async def test_remove_mark_missing_reports_success(mark_service):
    assert await mark_service.remove_mark("ghost") is True


@pytest.mark.asyncio
async def test_execute_mark_clicks_stored_coordinates(mark_service):
    await mark_service.add_mark("home", 30, 40)
    with patch("pyautogui.click") as mock_click:
        result = await mark_service.execute_mark("home")
    assert result is True
    mock_click.assert_called_once_with(30, 40)


@pytest.mark.asyncio
async def test_execute_mark_missing_returns_false_without_click(mark_service):
    with patch("pyautogui.click") as mock_click:
        result = await mark_service.execute_mark("ghost")
    assert result is False
    mock_click.assert_not_called()


@pytest.mark.asyncio
async def test_reset_all_marks_returns_count_and_clears(mark_service):
    await mark_service.add_mark("a", 1, 1)
    await mark_service.add_mark("b", 2, 2)
    assert await mark_service.reset_all_marks() == 2
    assert await mark_service.get_all_marks_internal() == {}


@pytest.mark.asyncio
async def test_handle_create_command_persists_rounded_coordinates(mark_service):
    command = MarkCreateCommand(label="home", x=100.4, y=200.6)
    await mark_service.handle_mark_command_parsed(MarkCommandParsedEvent(command=command, source="speech"))
    assert await mark_service.get_mark_coordinates_internal("home") == (100, 201)


@pytest.mark.asyncio
async def test_handle_delete_command_removes_mark(mark_service):
    await mark_service.add_mark("home", 1, 2)
    command = MarkDeleteCommand(label="home")
    await mark_service.handle_mark_command_parsed(MarkCommandParsedEvent(command=command, source="speech"))
    assert await mark_service.get_mark_coordinates_internal("home") is None


@pytest.mark.asyncio
async def test_handle_execute_command_ignores_unknown_mark(mark_service):
    command = MarkExecuteCommand(label="ghost")
    with patch("pyautogui.click") as mock_click:
        await mark_service.handle_mark_command_parsed(MarkCommandParsedEvent(command=command, source="speech"))
    mock_click.assert_not_called()


@pytest.mark.parametrize("show", [True, False])
@pytest.mark.asyncio
async def test_set_visualization_includes_marks_only_when_shown(mark_service, show):
    await mark_service.add_mark("home", 1, 2)
    mark_service.event_bus = AsyncMock()
    await mark_service.set_visualization(show)
    published = mark_service.event_bus.publish.call_args.args[0]
    assert published.is_visible is show
    if show:
        assert "home" in published.marks
    else:
        assert published.marks is None


@pytest.mark.asyncio
async def test_handle_ui_request_create_persists_mark(mark_service):
    await mark_service.handle_mark_ui_request(MarkUiRequestEvent(op="create", name="home", x=5, y=6))
    assert await mark_service.get_mark_coordinates_internal("home") == (5, 6)
