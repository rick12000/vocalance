import pytest

from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.events.grid_events import GridClickHistoryChangedEvent
from vocalance.app.services.command_flow.execution.grid.click_tracker_service import (
    click_point_in_rect,
    prioritize_grid_rects,
    rects_with_click_counts,
)
from vocalance.app.services.storage.storage_models import GridClickEvent


@pytest.mark.parametrize(
    "x, y, expected",
    [(50, 50, True), (0, 0, True), (100, 100, True), (101, 50, False), (50, -1, False)],
)
def test_click_point_in_rect_includes_boundaries(x, y, expected):
    click = GridClickEvent(x=x, y=y, timestamp=0.0)
    assert click_point_in_rect(click, 0, 0, 100, 100) is expected


def test_rects_with_click_counts_assigns_each_click_to_its_rect():
    rect_defs = [
        {"x": 0, "y": 0, "w": 100, "h": 100},
        {"x": 100, "y": 100, "w": 100, "h": 100},
        {"x": 200, "y": 200, "w": 100, "h": 100},
    ]
    clicks = [
        GridClickEvent(x=50, y=50, timestamp=0.0),
        GridClickEvent(x=150, y=150, timestamp=0.0),
        GridClickEvent(x=155, y=155, timestamp=0.0),
        GridClickEvent(x=250, y=250, timestamp=0.0),
    ]
    result = rects_with_click_counts(rect_defs, clicks)
    assert len(result) == len(rect_defs)
    assert [r["clicks"] for r in result] == [1, 2, 1]


def test_rects_with_click_counts_no_clicks_keeps_shape_with_zero_counts():
    rect_defs = [{"x": 0, "y": 0, "w": 100, "h": 100}, {"x": 100, "y": 0, "w": 100, "h": 100}]
    result = rects_with_click_counts(rect_defs, [])
    assert len(result) == len(rect_defs)
    assert all(r["clicks"] == 0 for r in result)


def test_prioritize_grid_rects_orders_by_clicks_then_position():
    rects = [
        {"data": {"x": 100, "y": 200}, "clicks": 1},
        {"data": {"x": 0, "y": 0}, "clicks": 1},
        {"data": {"x": 0, "y": 0}, "clicks": 15},
    ]
    result = prioritize_grid_rects(rects)
    assert [r["clicks"] for r in result] == [15, 1, 1]
    assert (result[1]["data"]["x"], result[1]["data"]["y"]) == (0, 0)


@pytest.mark.asyncio
async def test_handle_mouse_click_accumulates_into_published_snapshot(click_tracker_service, mock_event_bus):
    for x, y in [(10, 20), (30, 40), (50, 60)]:
        await click_tracker_service._handle_mouse_click(PerformMouseClickEventData(x=x, y=y))
    await click_tracker_service.publish_click_history_snapshot()

    published = mock_event_bus.publish.call_args.args[0]
    assert isinstance(published, GridClickHistoryChangedEvent)
    assert len(published.clicks_snapshot) == 3
    assert [(c.x, c.y) for c in published.clicks_snapshot] == [(10, 20), (30, 40), (50, 60)]
