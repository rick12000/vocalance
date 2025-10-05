"""
Tests for GridService - grid overlay and cell selection functionality.
"""
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock

from iris.app.services.grid.grid_service import GridService
from iris.app.events.command_events import GridCommandParsedEvent
from iris.app.events.grid_events import GridVisibilityChangedEventData
from iris.app.events.core_events import CommandExecutedStatusEvent
from iris.app.config.command_types import GridShowCommand, GridSelectCommand, GridCancelCommand


@pytest_asyncio.fixture
async def grid_service(event_bus, app_config):
    """Create grid service for testing."""
    service = GridService(event_bus, app_config)
    service.setup_subscriptions()
    
    await event_bus.start_worker()
    yield service
    await event_bus.stop_worker()


@pytest.mark.asyncio
async def test_grid_show_default(grid_service, app_config):
    """Test showing grid with default rectangle count."""
    service = grid_service
    event_bus = service._event_bus
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(GridVisibilityChangedEventData, capture_event)
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = GridShowCommand(num_rects=None)
    event = GridCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    
    viz_events = [e for e in captured_events if isinstance(e, GridVisibilityChangedEventData)]
    status_events = [e for e in captured_events if isinstance(e, CommandExecutedStatusEvent)]
    
    assert len(viz_events) == 1
    assert viz_events[0].visible is True
    assert viz_events[0].rows is not None
    assert viz_events[0].cols is not None
    assert len(status_events) == 1
    assert status_events[0].success is True


@pytest.mark.asyncio
async def test_grid_show_with_custom_count(grid_service):
    """Test showing grid with custom rectangle count."""
    service = grid_service
    event_bus = service._event_bus
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(GridVisibilityChangedEventData, capture_event)
    
    command = GridShowCommand(num_rects=9)
    event = GridCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    
    viz_events = [e for e in captured_events if isinstance(e, GridVisibilityChangedEventData)]
    assert len(viz_events) == 1
    assert viz_events[0].visible is True
    assert viz_events[0].rows * viz_events[0].cols >= 9


@pytest.mark.asyncio
async def test_grid_dimension_calculation(grid_service):
    """Test grid dimension calculations are correct."""
    service = grid_service
    
    test_cases = [
        (4, (2, 2)),
        (9, (3, 3)),
        (16, (4, 4)),
        (12, (3, 4)),
        (15, (4, 4))
    ]
    
    for num_rects, expected_dims in test_cases:
        rows, cols = service._calculate_grid_dimensions(num_rects)
        assert rows * cols >= num_rects
        assert rows == expected_dims[0]
        assert cols == expected_dims[1]


@pytest.mark.asyncio
async def test_grid_cancel(grid_service):
    """Test canceling grid overlay."""
    service = grid_service
    event_bus = service._event_bus
    
    service._visible = True
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(GridVisibilityChangedEventData, capture_event)
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = GridCancelCommand()
    event = GridCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    
    viz_events = [e for e in captured_events if isinstance(e, GridVisibilityChangedEventData)]
    status_events = [e for e in captured_events if isinstance(e, CommandExecutedStatusEvent)]
    
    assert len(viz_events) == 1
    assert viz_events[0].visible is False
    assert len(status_events) == 1


@pytest.mark.asyncio
async def test_grid_select_cell(grid_service):
    """Test selecting a grid cell by number."""
    service = grid_service
    event_bus = service._event_bus
    
    service._visible = True
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    from iris.app.events.grid_events import ClickGridCellRequestEventData
    event_bus.subscribe(ClickGridCellRequestEventData, capture_event)
    
    command = GridSelectCommand(selected_number=5)
    event = GridCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    
    click_events = [e for e in captured_events if isinstance(e, ClickGridCellRequestEventData)]
    status_events = [e for e in captured_events if isinstance(e, CommandExecutedStatusEvent)]
    
    assert len(click_events) == 1
    assert click_events[0].cell_number == 5
    assert len(status_events) == 1


@pytest.mark.asyncio
async def test_grid_select_when_not_visible(grid_service):
    """Test that grid select is rejected when grid is not visible."""
    service = grid_service
    event_bus = service._event_bus
    
    service._visible = False
    
    captured_events = []
    async def capture_event(event):
        captured_events.append(event)
    
    event_bus.subscribe(CommandExecutedStatusEvent, capture_event)
    
    command = GridSelectCommand(selected_number=5)
    event = GridCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    
    status_events = [e for e in captured_events if isinstance(e, CommandExecutedStatusEvent)]
    assert len(status_events) == 1
    assert status_events[0].success is False


@pytest.mark.asyncio
async def test_grid_visibility_state_tracking(grid_service):
    """Test that grid visibility state is tracked correctly."""
    service = grid_service
    event_bus = service._event_bus
    
    assert service._visible is False
    
    show_command = GridShowCommand(num_rects=9)
    show_event = GridCommandParsedEvent(command=show_command, source="speech")
    await event_bus.publish(show_event)
    await asyncio.sleep(0.1)
    
    assert service._visible is True
    
    cancel_command = GridCancelCommand()
    cancel_event = GridCommandParsedEvent(command=cancel_command, source="speech")
    await event_bus.publish(cancel_event)
    await asyncio.sleep(0.1)
    
    assert service._visible is False


@pytest.mark.parametrize("num_rects,expected_min_cells", [
    (4, 4),
    (9, 9),
    (16, 16),
    (12, 12),
    (25, 25)
])
@pytest.mark.asyncio
async def test_grid_dimensions_sufficient_cells(grid_service, num_rects, expected_min_cells):
    """Test that calculated grid dimensions provide sufficient cells."""
    service = grid_service
    
    rows, cols = service._calculate_grid_dimensions(num_rects)
    total_cells = rows * cols
    
    assert total_cells >= expected_min_cells

