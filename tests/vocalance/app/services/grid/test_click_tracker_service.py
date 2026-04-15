import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.services.grid.click_tracker_service import ClickTrackerService, prioritize_grid_rects
from vocalance.app.services.gui_async_bridge import GuiAsyncBridge


@pytest.fixture
def mock_event_bus():
    bus = Mock()
    bus.subscribe = Mock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_storage():
    storage = Mock()
    storage.read = AsyncMock()
    storage.write = AsyncMock()
    return storage


@pytest_asyncio.fixture
async def click_tracker(mock_event_bus, mock_storage):
    loop = asyncio.get_running_loop()
    bridge = GuiAsyncBridge(loop)
    svc = ClickTrackerService(
        event_bus=mock_event_bus,
        storage=mock_storage,
        gui_async_bridge=bridge,
        ui_refresh_debounce_s=0.001,
        persist_debounce_s=9999.0,
    )
    yield svc
    await svc.shutdown()


# ---------------------------------------------------------------------------
# prioritize_grid_rects
# ---------------------------------------------------------------------------


def test_prioritize_grid_rects_empty():
    assert prioritize_grid_rects([]) == []


def test_prioritize_grid_rects_by_click_count():
    rects = [
        {"id": 1, "clicks": 5},
        {"id": 2, "clicks": 15},
        {"id": 3, "clicks": 2},
        {"id": 4, "clicks": 10},
    ]
    result = prioritize_grid_rects(rects)
    assert result[0]["id"] == 2
    assert result[1]["id"] == 4
    assert result[2]["id"] == 1
    assert result[3]["id"] == 3


def test_prioritize_grid_rects_handles_invalid_clicks():
    rects = [
        {"id": 1, "clicks": None},
        {"id": 2, "clicks": "invalid"},
        {"id": 3, "clicks": 10},
    ]
    result = prioritize_grid_rects(rects)
    assert result[0]["id"] == 3


def test_prioritize_grid_rects_stable_ties_by_position():
    def make_rects():
        return [
            {"data": {"x": 100, "y": 200}, "clicks": 1},
            {"data": {"x": 0, "y": 0}, "clicks": 1},
        ]

    r1 = prioritize_grid_rects(make_rects())
    r2 = prioritize_grid_rects(make_rects())
    assert [(r["data"]["x"], r["data"]["y"]) for r in r1] == [(r["data"]["x"], r["data"]["y"]) for r in r2]
    assert r1[0]["data"]["y"] == 0


# ---------------------------------------------------------------------------
# _handle_mouse_click (internal, but tested via the stored state)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_physical_click_stores_click(click_tracker):
    click_tracker.record_physical_click(100, 200)
    clicks = click_tracker.get_all_clicks_sync()
    assert len(clicks) == 1
    assert clicks[0]["x"] == 100
    assert clicks[0]["y"] == 200


async def test_handle_mouse_click_stores_click(click_tracker):
    event = PerformMouseClickEventData(x=100, y=200, button="left")
    await click_tracker._handle_mouse_click(event)

    clicks = click_tracker.get_all_clicks_sync()
    assert len(clicks) == 1
    assert clicks[0]["x"] == 100
    assert clicks[0]["y"] == 200


@pytest.mark.asyncio
async def test_get_all_clicks_sync(click_tracker):
    for x, y in [(100, 100), (200, 200), (300, 300)]:
        await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=x, y=y, button="left"))

    clicks = click_tracker.get_all_clicks_sync()
    assert len(clicks) == 3


# ---------------------------------------------------------------------------
# get_clicks_for_rects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_clicks_for_rects_counts(click_tracker):
    await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=50, y=50, button="left"))
    await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=150, y=150, button="left"))
    await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=155, y=155, button="left"))
    await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=250, y=250, button="left"))

    rect_defs = [
        {"id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
        {"id": 2, "x": 100, "y": 100, "w": 100, "h": 100},
        {"id": 3, "x": 200, "y": 200, "w": 100, "h": 100},
    ]
    result = click_tracker.get_clicks_for_rects(rect_defs)

    assert result[0]["clicks"] == 1
    assert result[1]["clicks"] == 2
    assert result[2]["clicks"] == 1


@pytest.mark.asyncio
async def test_get_clicks_for_rects_boundary_inclusive(click_tracker):
    for x, y in [(0, 0), (100, 100), (50, 50)]:
        await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=x, y=y, button="left"))

    result = click_tracker.get_clicks_for_rects([{"id": 1, "x": 0, "y": 0, "w": 100, "h": 100}])
    assert result[0]["clicks"] == 3


@pytest.mark.asyncio
async def test_get_clicks_for_rects_invalid_rect_data(click_tracker):
    await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=50, y=50, button="left"))

    result = click_tracker.get_clicks_for_rects(
        [
            {"id": 1, "x": "invalid", "y": 0, "w": 100, "h": 100},
            {"id": 2, "x": 0, "y": 0},  # missing w, h
        ]
    )
    assert result[0]["clicks"] == 0
    assert result[1]["clicks"] == 0


@pytest.mark.asyncio
async def test_is_click_in_rect_true(click_tracker):
    assert click_tracker._is_click_in_rect({"x": 50, "y": 50}, 0, 0, 100, 100) is True


@pytest.mark.asyncio
async def test_is_click_in_rect_false(click_tracker):
    assert click_tracker._is_click_in_rect({"x": 150, "y": 150}, 0, 0, 100, 100) is False


@pytest.mark.asyncio
async def test_is_click_in_rect_invalid_data(click_tracker):
    assert click_tracker._is_click_in_rect({"x": "invalid", "y": 50}, 0, 0, 100, 100) is False


@pytest.mark.asyncio
async def test_multiple_clicks_same_rect(click_tracker):
    for x, y in [(25, 25), (50, 50), (75, 75)]:
        await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=x, y=y, button="left"))

    result = click_tracker.get_clicks_for_rects([{"id": 1, "x": 0, "y": 0, "w": 100, "h": 100}])
    assert result[0]["clicks"] == 3


@pytest.mark.asyncio
async def test_clicks_distributed_across_rects(click_tracker):
    for x, y in [(50, 50), (150, 150), (250, 250), (350, 350)]:
        await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=x, y=y, button="left"))

    result = click_tracker.get_clicks_for_rects(
        [
            {"id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
            {"id": 2, "x": 100, "y": 100, "w": 100, "h": 100},
            {"id": 3, "x": 200, "y": 200, "w": 200, "h": 200},
        ]
    )
    assert result[0]["clicks"] == 1
    assert result[1]["clicks"] == 1
    assert result[2]["clicks"] == 2
