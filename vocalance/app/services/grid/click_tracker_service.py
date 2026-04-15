import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.events.grid_events import GridClickHistoryChangedEvent
from vocalance.app.services.base_service import Service
from vocalance.app.services.gui_async_bridge import GuiAsyncBridge
from vocalance.app.services.storage.storage_models import GridClickEvent, GridClicksData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


def prioritize_grid_rects(rect_details_with_clicks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rect_details_with_clicks:
        return []

    def sort_key(rect: Dict[str, Any]) -> tuple:
        clicks = rect.get("clicks", 0)
        if not isinstance(clicks, (int, float)):
            clicks = 0
        data = rect.get("data") or {}
        try:
            x, y = float(data.get("x", 0)), float(data.get("y", 0))
        except (TypeError, ValueError):
            x, y = 0.0, 0.0
        try:
            tie = int(rect.get("id", 0))
        except (TypeError, ValueError):
            tie = 0
        return (-float(clicks), y, x, tie)

    return sorted(rect_details_with_clicks, key=sort_key)


class ClickTrackerService(Service):
    """Click history in memory with debounced UI notify and async disk persistence."""

    def __init__(
        self,
        event_bus: EventBus,
        storage: StorageService,
        gui_async_bridge: GuiAsyncBridge,
        ui_refresh_debounce_s: float,
        persist_debounce_s: float,
    ) -> None:
        self._event_bus = event_bus
        self._storage = storage
        self._bridge = gui_async_bridge
        self._ui_refresh_debounce_s = ui_refresh_debounce_s
        self._persist_debounce_s = persist_debounce_s
        self._lock = threading.RLock()
        self._clicks: List[GridClickEvent] = []
        self._ui_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None
        event_bus.subscribe(PerformMouseClickEventData, self._handle_mouse_click)

    async def initialize(self) -> bool:
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
        self._ui_task = asyncio.create_task(self._debounced_ui_notify())
        self._persist_task = asyncio.create_task(self._debounced_persist())

    def _request_debounce_after_mutation(self) -> None:
        self._bridge.invoke_on_gui_loop(self._reschedule_debounce_tasks)

    def record_physical_click(self, x: int, y: int) -> None:
        """Record a screen-space click; safe from any thread (e.g. grid pyautogui worker)."""
        self._append_click_locked(x, y)
        self._request_debounce_after_mutation()

    async def _handle_mouse_click(self, event: PerformMouseClickEventData) -> None:
        self._append_click_locked(event.x, event.y)
        self._request_debounce_after_mutation()

    async def _debounced_ui_notify(self) -> None:
        try:
            await asyncio.sleep(self._ui_refresh_debounce_s)
            await self._event_bus.publish(GridClickHistoryChangedEvent())
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

    def _is_click_in_rect(self, click: Dict[str, Any], rx: int, ry: int, rw: int, rh: int) -> bool:
        try:
            return rx <= click.get("x", 0) <= rx + rw and ry <= click.get("y", 0) <= ry + rh
        except (TypeError, ValueError):
            return False

    def get_clicks_for_rects(self, rect_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self._lock:
            all_clicks = [c.model_dump() for c in self._clicks]
        result = []
        for rect in rect_definitions:
            try:
                rx, ry = int(rect["x"]), int(rect["y"])
                rw, rh = int(rect["w"]), int(rect["h"])
                count = sum(1 for c in all_clicks if self._is_click_in_rect(c, rx, ry, rw, rh))
                result.append({"data": rect, "clicks": count})
            except (KeyError, ValueError, TypeError):
                result.append({"data": rect, "clicks": 0})
        return result

    def get_all_clicks_sync(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [c.model_dump() for c in self._clicks]

    async def shutdown(self) -> None:
        self._event_bus.unsubscribe(PerformMouseClickEventData, self._handle_mouse_click)
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
        if not clicks_to_save:
            return
        try:
            await self._storage.write(data=GridClicksData(clicks=clicks_to_save))
            logger.info("Saved %d clicks to storage on shutdown", len(clicks_to_save))
        except Exception as e:
            logger.error("Error saving clicks on shutdown: %s", e, exc_info=True)
