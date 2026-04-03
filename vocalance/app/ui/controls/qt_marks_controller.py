import asyncio
import logging
from typing import List, Optional, Union

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.mark_events import MarkData  # noqa: F401
from vocalance.app.events.mark_events import (
    AllMarksClearedEventData,
    MarkCreatedEventData,
    MarkCreateRequestEventData,
    MarkDeleteAllRequestEventData,
    MarkDeleteByNameRequestEventData,
    MarkDeletedEventData,
    MarkExecuteRequestEventData,
    MarkGetAllRequestEventData,
    MarkOperationFailedEventData,
    MarkOperationSuccessEventData,
    MarksChangedEventData,
    MarkVisualizationStateChangedEventData,
    MarkVisualizeAllRequestEventData,
    MarkVisualizeCancelRequestEventData,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtMarksController(QtBaseController):
    """Controller for marks functionality — orchestrates between mark service and view."""

    marks_loaded = Signal(list)
    mark_created = Signal(str, int, int)
    mark_deleted = Signal(str)
    all_marks_deleted = Signal()
    mark_overlay_shown = Signal()
    mark_overlay_hidden = Signal()
    operation_error = Signal(str)
    status_updated = Signal(str, bool)

    def __init__(
        self,
        event_bus: EventBus,
        mark_service,
        config: GlobalAppConfig,
    ):
        """Initialize marks controller.

        Args:
            event_bus: Event bus for pub/sub.
            mark_service: Mark service instance.
            config: Global app configuration.
        """
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtMarksController"),
        )

        self.mark_service = mark_service
        self.config = config
        self.mark_view = None
        self.marks_list: List[MarkData] = []

        self._subscribe_to_events()
        self.logger.debug("QtMarksController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to mark service events."""
        try:
            self.event_bus.subscribe(MarksChangedEventData, self._on_marks_changed)
            self.event_bus.subscribe(MarkOperationSuccessEventData, self._on_mark_operation_status)
            self.event_bus.subscribe(MarkOperationFailedEventData, self._on_mark_operation_status)
            self.event_bus.subscribe(MarkCreatedEventData, self._handle_mark_list_changed)
            self.event_bus.subscribe(MarkDeletedEventData, self._handle_mark_list_changed)
            self.event_bus.subscribe(AllMarksClearedEventData, self._handle_mark_list_changed)
            self.event_bus.subscribe(MarkVisualizationStateChangedEventData, self._handle_mark_visualization_state_changed)
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def set_mark_view(self, mark_view) -> None:
        """Set the mark view reference.

        Args:
            mark_view: Mark overlay view instance.
        """
        self.mark_view = mark_view
        if self.mark_view:
            self.logger.debug("Mark view reference set")

    def refresh_marks(self) -> None:
        """Publish a request for all marks."""
        asyncio.ensure_future(self.event_bus.publish(MarkGetAllRequestEventData()))

    def create_mark(self, name: Optional[str], x: int, y: int, description: Optional[str] = None) -> None:
        """Publish a mark creation request.

        Args:
            name: Mark name.
            x: Screen x-coordinate.
            y: Screen y-coordinate.
            description: Optional description.
        """
        asyncio.ensure_future(self.event_bus.publish(MarkCreateRequestEventData(name=name, x=x, y=y, description=description)))

    def delete_mark_by_name(self, mark_name: str) -> None:
        """Publish a mark deletion request by name.

        Args:
            mark_name: Name of the mark to delete.
        """
        asyncio.ensure_future(self.event_bus.publish(MarkDeleteByNameRequestEventData(name=mark_name)))

    def delete_mark(self, mark_name: str) -> None:
        """Delete a mark by name.

        Args:
            mark_name: Name of the mark to delete.
        """
        self.delete_mark_by_name(mark_name)

    def delete_all_marks(self) -> None:
        """Publish a request to delete all marks."""
        asyncio.ensure_future(self.event_bus.publish(MarkDeleteAllRequestEventData()))

    def execute_mark(self, identifier: Union[str, int]) -> None:
        """Publish a mark execution request.

        Args:
            identifier: Mark name or numeric ID.
        """
        asyncio.ensure_future(self.event_bus.publish(MarkExecuteRequestEventData(name_or_id=identifier)))

    def request_show_overlay(self) -> None:
        """Publish a request to show the mark visualization overlay."""
        asyncio.ensure_future(self.event_bus.publish(MarkVisualizeAllRequestEventData()))

    def request_hide_overlay(self) -> None:
        """Publish a request to hide the mark visualization overlay."""
        asyncio.ensure_future(self.event_bus.publish(MarkVisualizeCancelRequestEventData()))

    def show_mark_overlay(self) -> None:
        """Fetch fresh marks from the service then show the overlay."""
        if not self.mark_view:
            self.logger.error("Cannot show mark overlay: mark view not set")
            return
        asyncio.ensure_future(self._show_mark_overlay_async())

    def show_marks_overlay(self) -> None:
        """Alias for show_mark_overlay."""
        self.show_mark_overlay()

    async def _show_mark_overlay_async(self) -> None:
        """Fetch marks from the service then show the overlay once they arrive."""
        self.refresh_marks()

        for _ in range(50):
            await asyncio.sleep(0.01)
            if self.marks_list:
                break
        else:
            self.logger.warning("Timeout waiting for marks from service before showing overlay")

        if self.mark_view:
            self.mark_view.marks = {mark.name: (mark.x, mark.y) for mark in self.marks_list}
            self.mark_view.show_requested.emit()

    def hide_mark_overlay(self) -> None:
        """Hide the mark overlay directly via the view."""
        if self.mark_view:
            self.mark_view.hide_requested.emit()
        else:
            self.logger.error("Cannot hide mark overlay: mark view not set")

    def is_mark_overlay_active(self) -> bool:
        """Return True if the mark overlay is currently active."""
        return self.mark_view.is_active() if self.mark_view else False

    def update_mark_view_data(self, marks_list: List[MarkData]) -> None:
        """Push updated mark data to the view.

        Args:
            marks_list: List of MarkData instances to display.
        """
        if self.mark_view:
            self.mark_view.update_marks(marks_list)

    def on_mark_visualization_shown(self) -> None:
        """Handle successful mark visualization show from the view."""
        asyncio.ensure_future(self.event_bus.publish(MarkVisualizationStateChangedEventData(is_visible=True)))

    def on_mark_visualization_hidden(self) -> None:
        """Handle mark visualization hide from the view."""
        asyncio.ensure_future(self.event_bus.publish(MarkVisualizationStateChangedEventData(is_visible=False)))

    def on_mark_visualization_failed(self, error_message: str) -> None:
        """Handle failed mark visualization from the view.

        Args:
            error_message: Description of the failure.
        """
        self.notify_status(f"Mark visualization failed: {error_message}", True)

    async def _on_marks_changed(self, event) -> None:
        """Handle marks changed event and update cached marks list."""
        if hasattr(event, "marks"):
            self.marks_list = [
                MarkData(name=m["name"], x=m["x"], y=m["y"], description=m.get("description", "")) for m in event.marks.values()
            ]
        else:
            self.marks_list = []

        self.marks_loaded.emit(self.marks_list)
        if self.mark_view:
            self.update_mark_view_data(self.marks_list)

    async def _on_mark_operation_status(self, event) -> None:
        """Handle mark operation success/failure events."""
        self.notify_status(getattr(event, "message", "Mark operation completed."), not getattr(event, "success", True))

    async def _handle_mark_list_changed(self, event) -> None:
        """Handle mark list change events by refreshing from the service."""
        self.refresh_marks()

    async def _handle_mark_visualization_state_changed(self, event_data) -> None:
        """Handle mark visualization state change from the service layer."""
        if not self.mark_view:
            return

        if event_data.is_visible and not self.mark_view.is_active():
            self.refresh_marks()
            for _ in range(50):
                await asyncio.sleep(0.01)
                if self.marks_list:
                    break
            else:
                self.logger.warning("Timeout waiting for marks before showing overlay")

            if self.mark_view:
                self.mark_view.marks = {mark.name: (mark.x, mark.y) for mark in self.marks_list}
                self.mark_view.show_requested.emit()

        elif not event_data.is_visible and self.mark_view.is_active():
            self.mark_view.hide_requested.emit()

    def notify_status(self, message: str, is_error: bool = False) -> None:
        """Emit a status update signal.

        Args:
            message: Status message text.
            is_error: True if this represents an error condition.
        """
        self.status_updated.emit(message, is_error)
        if is_error:
            self.operation_error.emit(message)

    def cleanup(self) -> None:
        """Unsubscribe from all events, clean up the view, and release resources."""
        try:
            self.event_bus.unsubscribe(MarksChangedEventData, self._on_marks_changed)
            self.event_bus.unsubscribe(MarkOperationSuccessEventData, self._on_mark_operation_status)
            self.event_bus.unsubscribe(MarkOperationFailedEventData, self._on_mark_operation_status)
            self.event_bus.unsubscribe(MarkCreatedEventData, self._handle_mark_list_changed)
            self.event_bus.unsubscribe(MarkDeletedEventData, self._handle_mark_list_changed)
            self.event_bus.unsubscribe(AllMarksClearedEventData, self._handle_mark_list_changed)
            self.event_bus.unsubscribe(MarkVisualizationStateChangedEventData, self._handle_mark_visualization_state_changed)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        if self.mark_view:
            self.mark_view.cleanup()

        super().cleanup()
