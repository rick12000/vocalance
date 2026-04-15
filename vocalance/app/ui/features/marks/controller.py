import asyncio
import logging
from typing import Any, List, Optional, Union

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.mark_events import (
    MarkData,
    MarksChangedEventData,
    MarkUiRequestEvent,
    MarkUiResponseEvent,
    MarkVisualizationStateChangedEventData,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtMarksController(QtBaseController):
    marks_loaded = Signal(list)
    mark_overlay_shown = Signal()
    mark_overlay_hidden = Signal()
    operation_error = Signal(str)

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger("QtMarksController"))
        self.config = config
        self.marks_list: List[MarkData] = []
        self.event_bus.subscribe(MarksChangedEventData, self._on_marks_changed)
        self.event_bus.subscribe(MarkVisualizationStateChangedEventData, self._on_visualization_state_changed)
        self.event_bus.subscribe(MarkUiResponseEvent, self._on_mark_ui_response)

    def _mark_ui(self, **fields: Any) -> None:
        asyncio.create_task(self.event_bus.publish(MarkUiRequestEvent(**fields)))

    def refresh_marks(self) -> None:
        self._mark_ui(op="refresh_list")

    def update_marks_list(self, marks: dict) -> None:
        if marks:
            self.marks_list = [
                MarkData(name=m["name"], x=m["x"], y=m["y"], description=m.get("description", "")) for m in marks.values()
            ]
        else:
            self.marks_list = []
        self.marks_loaded.emit(self.marks_list)
        overlay = self.get_view()
        if overlay:
            overlay.update_marks(self.marks_list)

    def _push_positions_to_overlay(self, overlay: Any) -> None:
        overlay.marks = {mark.name: (mark.x, mark.y) for mark in self.marks_list}

    def create_mark(self, name: Optional[str], x: int, y: int, description: Optional[str] = None) -> None:
        self._mark_ui(op="create", name=name, x=x, y=y, description=description)

    def delete_mark(self, mark_name: str) -> None:
        self._mark_ui(op="delete", mark_name=mark_name)

    def delete_all_marks(self) -> None:
        self._mark_ui(op="delete_all")

    def execute_mark(self, identifier: Union[str, int]) -> None:
        self._mark_ui(op="execute", identifier=str(identifier))

    def request_show_overlay(self) -> None:
        self._mark_ui(op="set_visualization", visible=True)

    def request_hide_overlay(self) -> None:
        self._mark_ui(op="set_visualization", visible=False)

    def show_mark_overlay(self) -> None:
        if not self.get_view():
            self.logger.error("Cannot show mark overlay: mark view not set")
            return
        self._mark_ui(op="prepare_overlay")

    def hide_mark_overlay(self) -> None:
        overlay = self.get_view()
        if overlay:
            overlay.hide_requested.emit()
        else:
            self.logger.error("Cannot hide mark overlay: mark view not set")

    def is_mark_overlay_active(self) -> bool:
        overlay = self.get_view()
        return overlay.is_active() if overlay else False

    def _on_marks_changed(self, marks_snapshot: MarksChangedEventData) -> None:
        self.update_marks_list(marks_snapshot.marks)

    async def _on_visualization_state_changed(self, viz_state: MarkVisualizationStateChangedEventData) -> None:
        overlay = self.get_view()
        if not overlay:
            return
        if viz_state.is_visible and not overlay.is_active():
            self.update_marks_list(viz_state.marks or {})
            overlay = self.get_view()
            if overlay:
                self._push_positions_to_overlay(overlay)
                overlay.show_requested.emit()
        elif not viz_state.is_visible and overlay.is_active():
            overlay.hide_requested.emit()

    def _on_mark_ui_response(self, event: MarkUiResponseEvent) -> None:
        if event.kind == "overlay_marks":
            self.update_marks_list(event.marks)
            overlay = self.get_view()
            if overlay:
                self._push_positions_to_overlay(overlay)
                overlay.show_requested.emit()
        elif event.kind == "create_result" and not event.success:
            self.notify_status(event.message, True)

    def notify_status(self, message: str, is_error: bool = False) -> None:
        self.emit_status(message, is_error)
        if is_error:
            self.operation_error.emit(message)

    def cleanup(self) -> None:
        self.event_bus.unsubscribe(MarksChangedEventData, self._on_marks_changed)
        self.event_bus.unsubscribe(MarkVisualizationStateChangedEventData, self._on_visualization_state_changed)
        self.event_bus.unsubscribe(MarkUiResponseEvent, self._on_mark_ui_response)
        overlay = self.get_view()
        if overlay:
            overlay.cleanup()
        super().cleanup()
