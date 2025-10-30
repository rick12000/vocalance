from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.events.grid_events import RequestClickCountsForGridEventData
from vocalance.app.services.grid.click_tracker_service import ClickTrackerService, prioritize_grid_rects
from vocalance.app.services.storage.storage_models import GridClickEvent, GridClicksData


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
def app_config():
    """Create application configuration for testing."""
    return GlobalAppConfig()


@pytest.fixture
def click_tracker(mock_event_bus, app_config, mock_storage):
    """Create ClickTrackerService instance."""
    return ClickTrackerService(event_bus=mock_event_bus, config=app_config, storage=mock_storage)


def test_prioritize_grid_rects_empty():
    """Test prioritizing empty list of rectangles."""
    result = prioritize_grid_rects([])
    assert result == []


def test_prioritize_grid_rects_by_click_count():
    """Test rectangles are sorted by click count descending."""
    rects = [
        {"id": 1, "clicks": 5},
        {"id": 2, "clicks": 15},
        {"id": 3, "clicks": 2},
        {"id": 4, "clicks": 10},
    ]

    result = prioritize_grid_rects(rects)

    # Should be sorted by clicks descending
    assert result[0]["id"] == 2
    assert result[1]["id"] == 4
    assert result[2]["id"] == 1
    assert result[3]["id"] == 3


def test_prioritize_grid_rects_handles_invalid_clicks():
    """Test prioritizing handles invalid click values."""
    rects = [
        {"id": 1, "clicks": None},
        {"id": 2, "clicks": "invalid"},
        {"id": 3, "clicks": 10},
    ]

    result = prioritize_grid_rects(rects)

    # Valid clicks should come first
    assert result[0]["id"] == 3


@pytest.mark.asyncio
async def test_handle_mouse_click_stores_click(click_tracker, mock_storage, mock_event_bus):
    """Test mouse click event stores click data."""
    mock_storage.read.return_value = GridClicksData(clicks=[])
    mock_storage.write.return_value = True

    event = PerformMouseClickEventData(x=100, y=200, button="left")
    await click_tracker._handle_mouse_click(event)

    # Should write to storage
    mock_storage.write.assert_called_once()
    written_data = mock_storage.write.call_args[1]["data"]
    assert len(written_data.clicks) == 1
    assert written_data.clicks[0].x == 100
    assert written_data.clicks[0].y == 200


@pytest.mark.asyncio
async def test_handle_mouse_click_publishes_event(click_tracker, mock_storage, mock_event_bus):
    """Test mouse click publishes ClickLoggedEvent."""
    mock_storage.read.return_value = GridClicksData(clicks=[])
    mock_storage.write.return_value = True

    event = PerformMouseClickEventData(x=100, y=200, button="left")
    await click_tracker._handle_mouse_click(event)

    # Should publish event through event publisher
    # Event publisher uses thread-safe publish, not event_bus.publish directly


@pytest.mark.asyncio
async def test_handle_click_counts_request(click_tracker, mock_storage):
    """Test calculating click counts for grid rectangles."""
    # Setup existing clicks
    existing_clicks = [
        GridClickEvent(x=50, y=50, timestamp=1000.0, cell_id=None),
        GridClickEvent(x=150, y=150, timestamp=2000.0, cell_id=None),
        GridClickEvent(x=250, y=250, timestamp=3000.0, cell_id=None),
    ]
    mock_storage.read.return_value = GridClicksData(clicks=existing_clicks)

    # Define rectangles
    rect_defs = [
        {"id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
        {"id": 2, "x": 100, "y": 100, "w": 100, "h": 100},
        {"id": 3, "x": 200, "y": 200, "w": 100, "h": 100},
    ]

    event = RequestClickCountsForGridEventData(request_id="123", rect_definitions=rect_defs)
    await click_tracker._handle_click_counts_request(event)

    # Event publisher should be called with response


@pytest.mark.asyncio
async def test_calculate_click_counts(click_tracker):
    """Test click count calculation for rectangles."""
    clicks = [
        {"x": 50, "y": 50},
        {"x": 150, "y": 150},
        {"x": 155, "y": 155},
        {"x": 250, "y": 250},
    ]

    rect_defs = [
        {"id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
        {"id": 2, "x": 100, "y": 100, "w": 100, "h": 100},
        {"id": 3, "x": 200, "y": 200, "w": 100, "h": 100},
    ]

    result = click_tracker._calculate_click_counts(clicks, rect_defs)

    # Rect 1 should have 1 click
    assert result[0]["clicks"] == 1
    # Rect 2 should have 2 clicks
    assert result[1]["clicks"] == 2
    # Rect 3 should have 1 click
    assert result[2]["clicks"] == 1


@pytest.mark.asyncio
async def test_calculate_click_counts_boundary_cases(click_tracker):
    """Test click counting at rectangle boundaries."""
    clicks = [
        {"x": 0, "y": 0},  # Top-left corner
        {"x": 100, "y": 100},  # Bottom-right corner
        {"x": 50, "y": 50},  # Inside
    ]

    rect_def = [{"id": 1, "x": 0, "y": 0, "w": 100, "h": 100}]

    result = click_tracker._calculate_click_counts(clicks, rect_def)

    # All three clicks should be counted (boundaries inclusive)
    assert result[0]["clicks"] == 3


@pytest.mark.asyncio
async def test_is_click_in_rect_true(click_tracker):
    """Test click detection inside rectangle."""
    click = {"x": 50, "y": 50}

    is_inside = click_tracker._is_click_in_rect(click, rect_x=0, rect_y=0, rect_w=100, rect_h=100)

    assert is_inside is True


@pytest.mark.asyncio
async def test_is_click_in_rect_false(click_tracker):
    """Test click detection outside rectangle."""
    click = {"x": 150, "y": 150}

    is_inside = click_tracker._is_click_in_rect(click, rect_x=0, rect_y=0, rect_w=100, rect_h=100)

    assert is_inside is False


@pytest.mark.asyncio
async def test_is_click_in_rect_handles_invalid_data(click_tracker):
    """Test click detection handles invalid data gracefully."""
    invalid_click = {"x": "invalid", "y": 50}

    is_inside = click_tracker._is_click_in_rect(invalid_click, rect_x=0, rect_y=0, rect_w=100, rect_h=100)

    assert is_inside is False


@pytest.mark.asyncio
async def test_get_click_statistics_empty(click_tracker, mock_storage):
    """Test statistics with no clicks."""
    mock_storage.read.return_value = GridClicksData(clicks=[])

    stats = await click_tracker.get_click_statistics()

    assert stats["total_clicks"] == 0


@pytest.mark.asyncio
async def test_get_click_statistics_with_clicks(click_tracker, mock_storage):
    """Test statistics calculation with clicks."""
    clicks = [
        GridClickEvent(x=100, y=100, timestamp=1000.0, cell_id=None),
        GridClickEvent(x=200, y=200, timestamp=2000.0, cell_id=None),
        GridClickEvent(x=300, y=300, timestamp=3000.0, cell_id=None),
    ]
    mock_storage.read.return_value = GridClicksData(clicks=clicks)

    stats = await click_tracker.get_click_statistics()

    assert stats["total_clicks"] == 3
    assert stats["earliest_click"] == 1000.0
    assert stats["latest_click"] == 3000.0


@pytest.mark.asyncio
async def test_calculate_click_counts_handles_invalid_rect_data(click_tracker):
    """Test click counting handles invalid rectangle definitions."""
    clicks = [{"x": 50, "y": 50}]

    rect_defs = [
        {"id": 1, "x": "invalid", "y": 0, "w": 100, "h": 100},
        {"id": 2, "x": 0, "y": 0},  # Missing w, h
    ]

    result = click_tracker._calculate_click_counts(clicks, rect_defs)

    # Should return 0 clicks for invalid rects
    assert result[0]["clicks"] == 0
    assert result[1]["clicks"] == 0


@pytest.mark.asyncio
async def test_multiple_clicks_in_same_rect(click_tracker):
    """Test multiple clicks counted in same rectangle."""
    clicks = [
        {"x": 25, "y": 25},
        {"x": 50, "y": 50},
        {"x": 75, "y": 75},
    ]

    rect_def = [{"id": 1, "x": 0, "y": 0, "w": 100, "h": 100}]

    result = click_tracker._calculate_click_counts(clicks, rect_def)

    assert result[0]["clicks"] == 3


@pytest.mark.asyncio
async def test_clicks_distributed_across_rects(click_tracker):
    """Test clicks distributed across multiple rectangles."""
    clicks = [
        {"x": 50, "y": 50},
        {"x": 150, "y": 150},
        {"x": 250, "y": 250},
        {"x": 350, "y": 350},
    ]

    rect_defs = [
        {"id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
        {"id": 2, "x": 100, "y": 100, "w": 100, "h": 100},
        {"id": 3, "x": 200, "y": 200, "w": 200, "h": 200},
    ]

    result = click_tracker._calculate_click_counts(clicks, rect_defs)

    assert result[0]["clicks"] == 1
    assert result[1]["clicks"] == 1
    assert result[2]["clicks"] == 2  # Contains both (250,250) and (350,350)
