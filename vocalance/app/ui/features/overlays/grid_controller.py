import logging
from typing import TYPE_CHECKING

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.grid_events import GridClickHistoryChangedEvent, GridStateEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController

if TYPE_CHECKING:
    from vocalance.app.services.gui_async_bridge import GuiAsyncBridge


class QtGridController(QtBaseController):
    """Bridges grid bus events to QtGridView and publishes follow-up grid events."""

    def __init__(
        self,
        event_bus: EventBus,
        grid_service,
        config: GlobalAppConfig,
        gui_async_bridge: "GuiAsyncBridge",
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtGridController"),
        )

        self.grid_service = grid_service
        self.config = config
        self._gui_async_bridge = gui_async_bridge

        self._subscribe_to_events()
        self.logger.debug("QtGridController initialized")

    def _subscribe_to_events(self) -> None:
        try:
            self.event_bus.subscribe(GridStateEvent, self._handle_grid_state_event)
            self.event_bus.subscribe(GridClickHistoryChangedEvent, self._handle_grid_click_history_changed)
        except Exception as e:
            self.logger.error("Error subscribing to events: %s", e, exc_info=True)

    def show_grid_overlay(self, num_rects=None, click_mode: str = "click") -> None:
        view = self.get_view()
        if view:
            view.show(num_rects, click_mode)
        else:
            self.logger.error("Cannot show grid overlay: grid view not set")

    def hide_grid_overlay(self) -> None:
        view = self.get_view()
        if view:
            view.hide()
        else:
            self.logger.error("Cannot hide grid overlay: grid view not set")

    def is_grid_overlay_active(self) -> bool:
        view = self.get_view()
        return view.is_active() if view else False

    def handle_grid_selection(self, selection_key: str, click_mode: str = "click") -> bool:
        view = self.get_view()
        if view:
            return view.handle_selection(selection_key, click_mode)
        self.logger.error("Cannot handle grid selection: grid view not set")
        return False

    async def _publish_grid_state(self, event: GridStateEvent) -> None:
        await self.event_bus.publish(event)

    def _schedule_grid_state_event(self, event: GridStateEvent) -> None:
        self._gui_async_bridge.schedule_coro(self._publish_grid_state(event))

    def on_grid_selection_success(self, selected_number: int, center_x: float, center_y: float) -> None:
        self._schedule_grid_state_event(
            GridStateEvent(
                state="interaction_success",
                config={"selected_number": selected_number, "center_x": center_x, "center_y": center_y},
            )
        )

    def on_grid_selection_failed(self, selected_number: int, error_message: str) -> None:
        self._schedule_grid_state_event(
            GridStateEvent(
                state="interaction_failed",
                config={"selected_number": selected_number},
                message=error_message,
            )
        )

    async def _handle_grid_click_history_changed(self, _event: GridClickHistoryChangedEvent) -> None:
        view = self.get_view()
        if view is not None and view.is_active():
            view.refresh_click_labels_if_active()

    def _handle_grid_state_event(self, grid_state: GridStateEvent) -> None:
        if grid_state.state == "visible":
            config = grid_state.config or {}
            rows = config.get("rows")
            cols = config.get("cols")
            num_rects = (rows * cols) if (rows and cols) else None
            self.show_grid_overlay(num_rects, config.get("click_mode", "click"))
        elif grid_state.state == "hidden":
            self.hide_grid_overlay()
        elif grid_state.state == "interaction_request":
            if not self.is_grid_overlay_active():
                self.logger.warning("Grid not visible, cannot click cell %s", grid_state.config.get("cell_label"))
                return
            if not self.get_view():
                self.logger.error("Grid view not set, cannot click cell %s", grid_state.config.get("cell_label"))
                return
            config = grid_state.config or {}
            self.handle_grid_selection(config.get("cell_label"), config.get("click_mode", "click"))

    def cleanup(self) -> None:
        try:
            self.event_bus.unsubscribe(GridStateEvent, self._handle_grid_state_event)
            self.event_bus.unsubscribe(GridClickHistoryChangedEvent, self._handle_grid_click_history_changed)
        except Exception as e:
            self.logger.warning("Error during cleanup: %s", e)

        view = self.get_view()
        if view:
            view.cleanup()

        super().cleanup()
