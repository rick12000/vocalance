import asyncio

import pytest
import pytest_asyncio

from vocalance.app.config.command_types import GridSelectCommand, GridShowCommand
from vocalance.app.events.command_events import GridCommandParsedEvent
from vocalance.app.services.grid.grid_service import GridService


@pytest_asyncio.fixture
async def grid_service(event_bus, app_config):
    service = GridService(event_bus, app_config)
    yield service


@pytest.mark.asyncio
async def test_grid_show_default(grid_service, app_config):
    """Test showing grid with default rectangle count."""
    service = grid_service
    event_bus = service.event_bus

    command = GridShowCommand(num_rects=None)
    event = GridCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert service._visible is True


@pytest.mark.asyncio
async def test_grid_show_with_custom_count(grid_service):
    """Test showing grid with custom rectangle count."""
    service = grid_service
    event_bus = service.event_bus

    command = GridShowCommand(num_rects=9)
    event = GridCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert service._visible is True


@pytest.mark.asyncio
async def test_grid_dimension_calculation(grid_service):
    """Test grid dimension calculations are correct."""
    service = grid_service

    test_cases = [(4, (2, 2)), (9, (3, 3)), (16, (4, 4)), (12, (3, 4)), (15, (4, 4))]

    for num_rects, expected_dims in test_cases:
        rows, cols = service._calculate_grid_dimensions(num_rects)
        assert rows * cols >= num_rects
        assert rows == expected_dims[0]
        assert cols == expected_dims[1]


@pytest.mark.asyncio
async def test_grid_select_cell(grid_service):
    """Test selecting a grid cell by number."""
    service = grid_service
    event_bus = service.event_bus

    service._visible = True

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    from vocalance.app.events.grid_events import GridStateEvent

    event_bus.subscribe(GridStateEvent, capture_event)

    command = GridSelectCommand(selected_number=5)
    event = GridCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    click_events = [e for e in captured_events if isinstance(e, GridStateEvent) and e.state == "interaction_request"]

    assert len(click_events) == 1
    assert click_events[0].config.get("cell_label") == "5"


@pytest.mark.asyncio
async def test_grid_select_cell_drag_mode(grid_service):
    """Selecting a cell after drag-mode show carries click_mode drag."""
    service = grid_service
    event_bus = service.event_bus

    service._visible = True
    service._current_click_mode = "drag"

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    from vocalance.app.events.grid_events import GridStateEvent

    event_bus.subscribe(GridStateEvent, capture_event)

    command = GridSelectCommand(selected_number=3)
    event = GridCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    click_events = [e for e in captured_events if isinstance(e, GridStateEvent) and e.state == "interaction_request"]
    assert len(click_events) == 1
    assert click_events[0].config.get("cell_label") == "3"
    assert click_events[0].config.get("click_mode") == "drag"


@pytest.mark.asyncio
async def test_grid_visibility_state_tracking(grid_service):
    """Test that grid visibility state is tracked correctly."""
    service = grid_service
    event_bus = service.event_bus

    assert service._visible is False

    show_command = GridShowCommand(num_rects=9)
    show_event = GridCommandParsedEvent(command=show_command, source="speech")
    await event_bus.publish(show_event)
    await asyncio.sleep(0.1)

    assert service._visible is True


@pytest.mark.parametrize("num_rects,expected_min_cells", [(4, 4), (9, 9), (16, 16), (12, 12), (25, 25)])
@pytest.mark.asyncio
async def test_grid_dimensions_sufficient_cells(grid_service, num_rects, expected_min_cells):
    """Test that calculated grid dimensions provide sufficient cells."""
    service = grid_service

    rows, cols = service._calculate_grid_dimensions(num_rects)
    total_cells = rows * cols

    assert total_cells >= expected_min_cells
