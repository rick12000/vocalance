import logging

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.grid_events import GridClickHistoryChangedEvent, GridStateEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtGridController(QtBaseController):
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger("QtGridController"))
        self.config = config
        self.event_bus.subscribe(GridStateEvent, self.on_grid_state)
        self.event_bus.subscribe(GridClickHistoryChangedEvent, self.on_click_history_changed)

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

    async def on_click_history_changed(self, event: GridClickHistoryChangedEvent) -> None:
        view = self.get_view()
        if view is not None:
            view.set_clicks_snapshot(event.clicks_snapshot)

    def on_grid_state(self, grid_state: GridStateEvent) -> None:
        if grid_state.state == "visible":
            cfg = grid_state.config or {}
            rows, cols = cfg.get("rows"), cfg.get("cols")
            num_rects = (rows * cols) if (rows and cols) else None
            self.show_grid_overlay(num_rects, cfg.get("click_mode", "click"))
            return
        if grid_state.state == "hidden":
            self.hide_grid_overlay()
            return
        if grid_state.state != "interaction_request":
            return
        view = self.get_view()
        cfg = grid_state.config or {}
        if not view or not view.is_active():
            self.logger.warning("Grid not visible, cannot click cell %s", cfg.get("cell_label"))
            return
        view.handle_selection(cfg.get("cell_label"), cfg.get("click_mode", "click"))

    def cleanup(self) -> None:
        self.event_bus.unsubscribe(GridStateEvent, self.on_grid_state)
        self.event_bus.unsubscribe(GridClickHistoryChangedEvent, self.on_click_history_changed)
        view = self.get_view()
        if view:
            view.cleanup()
        super().cleanup()
