import pytest

from vocalance.app.config.command_types import GridSelectCommand, GridShowCommand
from vocalance.app.events.command_events import GridCommandParsedEvent
from vocalance.app.events.grid_events import GridStateEvent


@pytest.mark.parametrize("num_rects", [1, 4, 9, 12, 15, 500, 1000])
def test_calculate_grid_dimensions_is_minimal_near_square_cover(grid_service, num_rects):
    rows, cols = grid_service._calculate_grid_dimensions(num_rects)
    assert rows * cols >= num_rects
    assert (rows - 1) * cols < num_rects
    assert cols - rows in (0, 1)


@pytest.mark.parametrize("click_mode", ["click", "hover", "drag"])
@pytest.mark.asyncio
async def test_handle_show_command_publishes_visible_state(grid_service, click_mode):
    command = GridShowCommand(num_rects=9, click_mode=click_mode)
    await grid_service._handle_grid_command(GridCommandParsedEvent(command=command, source="speech"))

    event = grid_service.event_bus.publish.call_args.args[0]
    assert isinstance(event, GridStateEvent)
    assert event.state == "visible"
    assert event.config["rows"] * event.config["cols"] >= 9
    assert event.config["click_mode"] == click_mode
    assert grid_service.is_grid_visible() is True


@pytest.mark.asyncio
async def test_handle_show_command_defaults_to_config_rect_count(grid_service, app_config):
    await grid_service._handle_grid_command(GridCommandParsedEvent(command=GridShowCommand(num_rects=None), source="speech"))

    event = grid_service.event_bus.publish.call_args.args[0]
    assert event.config["rows"] * event.config["cols"] >= app_config.grid.default_rect_count


@pytest.mark.parametrize("click_mode", ["click", "hover", "drag"])
@pytest.mark.asyncio
async def test_select_inherits_click_mode_from_preceding_show(grid_service, click_mode):
    await grid_service._handle_grid_command(
        GridCommandParsedEvent(command=GridShowCommand(num_rects=4, click_mode=click_mode), source="speech")
    )
    await grid_service._handle_grid_command(GridCommandParsedEvent(command=GridSelectCommand(selected_number=3), source="speech"))

    event = grid_service.event_bus.publish.call_args.args[0]
    assert event.state == "interaction_request"
    assert event.config["cell_label"] == "3"
    assert event.config["click_mode"] == click_mode


@pytest.mark.asyncio
async def test_handle_select_command_ignored_when_grid_hidden(grid_service):
    await grid_service._handle_grid_command(GridCommandParsedEvent(command=GridSelectCommand(selected_number=5), source="speech"))

    grid_service.event_bus.publish.assert_not_awaited()
