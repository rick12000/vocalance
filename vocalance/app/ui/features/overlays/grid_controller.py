import logging

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.grid_events import GridClickHistoryChangedEvent, GridStateEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtGridController(QtBaseController):
    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtGridController"),
        )

        self.config = config

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

    async def _handle_grid_click_history_changed(self, event: GridClickHistoryChangedEvent) -> None:
        view = self.get_view()
        if view is not None:
            view.set_clicks_snapshot(event.clicks_snapshot)

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
