from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from typing_extensions import TypedDict

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.events.grid_events import GridClickHistoryChangedEvent
from vocalance.app.lifecycle.lifecycle import AppLifecycle
from vocalance.app.services.base_service import Service
from vocalance.app.services.storage.storage_models import GridClickEvent, GridClicksData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class RectDefinition(TypedDict):
    x: float
    y: float
    w: float
    h: float
    center_x: float
    center_y: float


class RectWithClickCount(TypedDict):
    data: RectDefinition
    clicks: int


def click_point_in_rect(click: GridClickEvent, rect_x: int, rect_y: int, rect_w: int, rect_h: int) -> bool:
    """Return True if ``click`` x/y falls inside the axis-aligned rectangle."""
    return rect_x <= click.x <= rect_x + rect_w and rect_y <= click.y <= rect_y + rect_h


def rects_with_click_counts(rect_definitions: List[RectDefinition], all_clicks: List[GridClickEvent]) -> List[RectWithClickCount]:
    """Attach per-rectangle click counts using spatial bucketing for performance."""
    if not rect_definitions:
        return []
    if not all_clicks:
        return [RectWithClickCount(data=rect_def, clicks=0) for rect_def in rect_definitions]

    first_rect = rect_definitions[0]
    try:
        bucket_w = int(first_rect["w"])
        bucket_h = int(first_rect["h"])
    except (KeyError, TypeError, ValueError):
        return [RectWithClickCount(data=rect_def, clicks=0) for rect_def in rect_definitions]

    click_buckets: Dict[Tuple[int, int], List[GridClickEvent]] = {}
    for click in all_clicks:
        try:
            bucket_x = int(click.x // bucket_w)
            bucket_y = int(click.y // bucket_h)
            key = (bucket_x, bucket_y)
            click_buckets.setdefault(key, []).append(click)
        except (TypeError, ValueError):
            continue

    processed_rects: List[RectWithClickCount] = []
    for rect_def in rect_definitions:
        try:
            rect_x, rect_y = int(rect_def["x"]), int(rect_def["y"])
            rect_w, rect_h = int(rect_def["w"]), int(rect_def["h"])
            bucket_x = int(rect_x // bucket_w)
            bucket_y = int(rect_y // bucket_h)
            count = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = (bucket_x + dx, bucket_y + dy)
                    if key in click_buckets:
                        for click in click_buckets[key]:
                            if click_point_in_rect(click, rect_x, rect_y, rect_w, rect_h):
                                count += 1
            processed_rects.append(RectWithClickCount(data=rect_def, clicks=count))
        except (KeyError, ValueError, TypeError):
            processed_rects.append(RectWithClickCount(data=rect_def, clicks=0))
    return processed_rects


def prioritize_grid_rects(rect_details_with_clicks: List[RectWithClickCount]) -> List[RectWithClickCount]:
    """Sort rectangles by descending click count, then position and id for stability."""
    if not rect_details_with_clicks:
        return []

    def sort_key(rect: RectWithClickCount) -> Tuple[float, float, float, int]:
        clicks = rect.get("clicks", 0)
        if not isinstance(clicks, (int, float)):
            clicks = 0
        data = rect.get("data") or {}
        try:
            x, y = float(data.get("x", 0)), float(data.get("y", 0))
        except (TypeError, ValueError):
            x, y = 0.0, 0.0
        return (-float(clicks), y, x, 0)

    return sorted(rect_details_with_clicks, key=sort_key)


class ClickTrackerService(Service):
    """Owns grid click history, persists via StorageService, notifies UI only through the event bus."""

    def __init__(
        self,
        event_bus: EventBus,
        storage: StorageService,
        gui_event_loop: asyncio.AbstractEventLoop,
        lifecycle: AppLifecycle,
        ui_refresh_debounce_s: float,
        persist_debounce_s: float,
    ) -> None:
        super().__init__(event_bus)
        self._storage = storage
        self._gui_loop = gui_event_loop
        self._lifecycle = lifecycle
        self._ui_refresh_debounce_s = ui_refresh_debounce_s
        self._persist_debounce_s = persist_debounce_s
        self._lock = threading.RLock()
        self._clicks: List[GridClickEvent] = []
        self._ui_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None
        self.subscribe(PerformMouseClickEventData, self._handle_mouse_click)

    def _run_on_gui_loop(self, fn: Callable[[], None]) -> None:
        self._gui_loop.call_soon_threadsafe(fn)

    async def initialize(self) -> bool:
        """Hydrate click history from storage; failures are logged but do not block startup."""
        try:
            clicks_data = await self._storage.read(model_type=GridClicksData)
            with self._lock:
                self._clicks = list(clicks_data.clicks)
            logger.info("Loaded %d clicks from storage", len(self._clicks))
            return True
        except Exception as e:
            logger.error("Failed to load click history: %s", e, exc_info=True)
            return True

    def _append_click_locked(self, x: int, y: int) -> None:
        new_click = GridClickEvent(x=x, y=y, timestamp=time.time(), cell_id=None)
        with self._lock:
            self._clicks.append(new_click)

    def _reschedule_debounce_tasks(self) -> None:
        for t in (self._ui_task, self._persist_task):
            if t is not None and not t.done():
                t.cancel()
        self._ui_task = self._lifecycle.spawn(self._debounced_ui_notify(), name="click-tracker-ui-debounce")
        self._persist_task = self._lifecycle.spawn(self._debounced_persist(), name="click-tracker-persist-debounce")

    def _request_debounce_after_mutation(self) -> None:
        self._run_on_gui_loop(self._reschedule_debounce_tasks)

    async def _handle_mouse_click(self, event: PerformMouseClickEventData) -> None:
        self._append_click_locked(event.x, event.y)
        self._request_debounce_after_mutation()

    async def publish_click_history_snapshot(self) -> None:
        with self._lock:
            snap = list(self._clicks)
        await self.event_bus.publish(GridClickHistoryChangedEvent(clicks_snapshot=snap))

    async def _debounced_ui_notify(self) -> None:
        try:
            await asyncio.sleep(self._ui_refresh_debounce_s)
            await self.publish_click_history_snapshot()
        except asyncio.CancelledError:
            raise

    async def _debounced_persist(self) -> None:
        try:
            await asyncio.sleep(self._persist_debounce_s)
        except asyncio.CancelledError:
            raise
        with self._lock:
            snapshot = list(self._clicks)
        try:
            await self._storage.write(data=GridClicksData(clicks=snapshot))
            logger.debug("Persisted %d grid clicks", len(snapshot))
        except Exception as e:
            logger.error("Debounced click history write failed: %s", e, exc_info=True)

    async def shutdown(self) -> None:
        for t in (self._ui_task, self._persist_task):
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        self._ui_task = None
        self._persist_task = None
        with self._lock:
            clicks_to_save = list(self._clicks)
        if clicks_to_save:
            try:
                await self._storage.write(data=GridClicksData(clicks=clicks_to_save))
                logger.info("Saved %d clicks to storage on shutdown", len(clicks_to_save))
            except Exception as e:
                logger.error("Error saving clicks on shutdown: %s", e, exc_info=True)
        await super().shutdown()
