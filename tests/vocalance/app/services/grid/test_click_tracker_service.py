import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.events.grid_events import GridClickHistoryChangedEvent
from vocalance.app.services.grid.click_tracker_service import (
    ClickTrackerService,
    click_point_in_rect,
    prioritize_grid_rects,
    rects_with_click_counts,
)


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
    lifecycle = Mock()
    lifecycle.spawn = Mock(side_effect=lambda coro, name="task": loop.create_task(coro, name=name))
    svc = ClickTrackerService(
        event_bus=mock_event_bus,
        storage=mock_storage,
        gui_event_loop=loop,
        lifecycle=lifecycle,
        ui_refresh_debounce_s=0.001,
        persist_debounce_s=9999.0,
    )
    yield svc
    await svc.shutdown()


def _last_published_event(mock_bus):
    calls = [c[0][0] for c in mock_bus.publish.call_args_list]
    return calls[-1]


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
# rects_with_click_counts / click_point_in_rect
# ---------------------------------------------------------------------------


def test_rects_with_click_counts_empty_clicks():
    rects = [{"id": 1, "x": 0, "y": 0, "w": 100, "h": 100}]
    assert rects_with_click_counts(rects, []) == [{"data": rects[0], "clicks": 0}]


@pytest.mark.asyncio
async def test_handle_mouse_click_stores_click(click_tracker, mock_event_bus):
    event = PerformMouseClickEventData(x=100, y=200, button="left")
    await click_tracker._handle_mouse_click(event)
    await click_tracker.publish_click_history_snapshot()
    published = _last_published_event(mock_event_bus)
    assert isinstance(published, GridClickHistoryChangedEvent)
    assert len(published.clicks_snapshot) == 1
    assert published.clicks_snapshot[0]["x"] == 100
    assert published.clicks_snapshot[0]["y"] == 200


@pytest.mark.asyncio
async def test_publish_click_history_snapshot_after_multiple(click_tracker, mock_event_bus):
    for x, y in [(100, 100), (200, 200), (300, 300)]:
        await click_tracker._handle_mouse_click(PerformMouseClickEventData(x=x, y=y, button="left"))
    await click_tracker.publish_click_history_snapshot()
    published = _last_published_event(mock_event_bus)
    assert len(published.clicks_snapshot) == 3


@pytest.mark.asyncio
async def test_rects_with_click_counts_counts():
    clicks = [{"x": 50, "y": 50}, {"x": 150, "y": 150}, {"x": 155, "y": 155}, {"x": 250, "y": 250}]
    rect_defs = [
        {"id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
        {"id": 2, "x": 100, "y": 100, "w": 100, "h": 100},
        {"id": 3, "x": 200, "y": 200, "w": 100, "h": 100},
    ]
    result = rects_with_click_counts(rect_defs, clicks)
    assert result[0]["clicks"] == 1
    assert result[1]["clicks"] == 2
    assert result[2]["clicks"] == 1


def test_rects_with_click_counts_boundary_inclusive():
    clicks = [{"x": 0, "y": 0}, {"x": 100, "y": 100}, {"x": 50, "y": 50}]
    result = rects_with_click_counts([{"id": 1, "x": 0, "y": 0, "w": 100, "h": 100}], clicks)
    assert result[0]["clicks"] == 3


def test_rects_with_click_counts_invalid_rect_data():
    clicks = [{"x": 50, "y": 50}]
    result = rects_with_click_counts(
        [
            {"id": 1, "x": "invalid", "y": 0, "w": 100, "h": 100},
            {"id": 2, "x": 0, "y": 0},
        ],
        clicks,
    )
    assert result[0]["clicks"] == 0
    assert result[1]["clicks"] == 0


def test_click_point_in_rect_true():
    assert click_point_in_rect({"x": 50, "y": 50}, 0, 0, 100, 100) is True


def test_click_point_in_rect_false():
    assert click_point_in_rect({"x": 150, "y": 150}, 0, 0, 100, 100) is False


def test_click_point_in_rect_invalid_data():
    assert click_point_in_rect({"x": "invalid", "y": 50}, 0, 0, 100, 100) is False


def test_multiple_clicks_same_rect():
    clicks = [{"x": 25, "y": 25}, {"x": 50, "y": 50}, {"x": 75, "y": 75}]
    result = rects_with_click_counts([{"id": 1, "x": 0, "y": 0, "w": 100, "h": 100}], clicks)
    assert result[0]["clicks"] == 3


def test_clicks_distributed_across_rects():
    clicks = [{"x": 50, "y": 50}, {"x": 150, "y": 150}, {"x": 250, "y": 250}, {"x": 350, "y": 350}]
    result = rects_with_click_counts(
        [
            {"id": 1, "x": 0, "y": 0, "w": 100, "h": 100},
            {"id": 2, "x": 100, "y": 100, "w": 100, "h": 100},
            {"id": 3, "x": 200, "y": 200, "w": 200, "h": 200},
        ],
        clicks,
    )
    assert result[0]["clicks"] == 1
    assert result[1]["clicks"] == 1
    assert result[2]["clicks"] == 2
