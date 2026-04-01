import asyncio
import logging
import threading
from typing import Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.grid_events import (
    ClickGridCellRequestEventData,
    GridInteractionFailedEventData,
    GridInteractionSuccessEventData,
    GridVisibilityChangedEventData,
    HideGridRequestEventData,
    ShowGridRequestEventData,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtGridController(QtBaseController):
    """Bridges grid bus events to QtGridView and publishes follow-up grid events."""

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        grid_service,
        config: GlobalAppConfig,
    ):
        super().__init__(
            event_bus=event_bus,
            event_loop=event_loop,
            logger=logging.getLogger("QtGridController"),
        )

        self.grid_service = grid_service
        self.config = config
        self.grid_view = None

        self._state_lock = threading.RLock()
        self._grid_visible = False

        self._subscribe_to_events()
        self.logger.debug("QtGridController initialized")

    def _subscribe_to_events(self) -> None:
        try:
            self.event_bus.subscribe(ShowGridRequestEventData, self._handle_show_grid_request)
            self.event_bus.subscribe(HideGridRequestEventData, self._handle_hide_grid_request)
            self.event_bus.subscribe(GridVisibilityChangedEventData, self._handle_grid_visibility_changed)
            self.event_bus.subscribe(ClickGridCellRequestEventData, self._handle_click_grid_cell_request)
            self.logger.debug("Subscribed to grid events")
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def set_grid_view(self, grid_view) -> None:
        self.grid_view = grid_view
        if self.grid_view:
            self.logger.debug("Grid view reference set")

    def show_grid_overlay(self, num_rects: Optional[int] = None, click_mode: str = "click") -> None:
        if self.grid_view:
            self.grid_view.show(num_rects, click_mode)
        else:
            self.logger.error("Cannot show grid overlay: grid view not set")

    def hide_grid_overlay(self) -> None:
        if self.grid_view:
            self.grid_view.hide()
        else:
            self.logger.error("Cannot hide grid overlay: grid view not set")

    def is_grid_overlay_active(self) -> bool:
        return self.grid_view.is_active() if self.grid_view else False

    def handle_grid_selection(self, selection_key: str, click_mode: str = "click") -> bool:
        if self.grid_view:
            return self.grid_view.handle_selection(selection_key, click_mode)
        self.logger.error("Cannot handle grid selection: grid view not set")
        return False

    def on_grid_selection_success(self, selected_number: int, center_x: float, center_y: float) -> None:
        with self._state_lock:
            self._grid_visible = False

        interaction_event = GridInteractionSuccessEventData(
            operation="select_cell", details={"selected_number": str(selected_number), "x": center_x, "y": center_y}
        )
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(interaction_event), self.event_loop)

        visibility_event = GridVisibilityChangedEventData(visible=False)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(visibility_event), self.event_loop)

    def on_grid_selection_failed(self, selected_number: int, error_message: str) -> None:
        interaction_event = GridInteractionFailedEventData(
            operation="select_cell",
            reason=error_message,
            cell_label=str(selected_number),
            details={"selected_number": str(selected_number)},
        )
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(interaction_event), self.event_loop)

    async def _handle_show_grid_request(self, event_data) -> None:
        num_rects = None
        if event_data.rows and event_data.cols:
            num_rects = event_data.rows * event_data.cols
        click_mode = getattr(event_data, "click_mode", "click")
        self.show_grid_overlay(num_rects, click_mode)

    async def _handle_hide_grid_request(self, event_data) -> None:
        self.hide_grid_overlay()

    async def _handle_click_grid_cell_request(self, event_data) -> None:
        if not self._grid_visible:
            self.logger.warning(f"Grid not visible, cannot click cell {event_data.cell_label}")
            return
        if not self.grid_view:
            self.logger.error(f"Grid view not set, cannot click cell {event_data.cell_label}")
            return
        click_mode = getattr(event_data, "click_mode", "click")
        self.handle_grid_selection(event_data.cell_label, click_mode)

    async def _handle_grid_visibility_changed(self, event_data) -> None:
        with self._state_lock:
            self._grid_visible = event_data.visible

    def cleanup(self) -> None:
        try:
            self.event_bus.unsubscribe(ShowGridRequestEventData, self._handle_show_grid_request)
            self.event_bus.unsubscribe(HideGridRequestEventData, self._handle_hide_grid_request)
            self.event_bus.unsubscribe(GridVisibilityChangedEventData, self._handle_grid_visibility_changed)
            self.event_bus.unsubscribe(ClickGridCellRequestEventData, self._handle_click_grid_cell_request)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        if self.grid_view:
            self.grid_view.cleanup()

        super().cleanup()
