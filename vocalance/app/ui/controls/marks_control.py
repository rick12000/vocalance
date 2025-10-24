from typing import List, Optional, Union

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.events.mark_events import (
    AllMarksClearedEventData,
    MarkCreatedEventData,
    MarkCreateRequestEventData,
    MarkData,
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
from vocalance.app.ui.controls.base_control import BaseController


class MarksController(BaseController):
    """Controller for marks functionality - orchestrates between service and view."""

    def __init__(self, event_bus, event_loop, logger, config: GlobalAppConfig):
        super().__init__(event_bus, event_loop, logger, "MarksController")
        self.config = config

        # Mark view reference (will be set by main window)
        self.mark_view = None

        # Mark visualization state tracking
        self._visualization_active = False

        self.subscribe_to_events(
            [
                (MarksChangedEventData, self._on_marks_changed),
                (MarkOperationSuccessEventData, self._on_mark_operation_status),
                (MarkOperationFailedEventData, self._on_mark_operation_status),
                (MarkCreatedEventData, self._handle_mark_list_changed),
                (MarkDeletedEventData, self._handle_mark_list_changed),
                (AllMarksClearedEventData, self._handle_mark_list_changed),
                (MarkVisualizationStateChangedEventData, self._handle_mark_visualization_state_changed),
            ]
        )

    def set_mark_view(self, mark_view):
        """Set the mark view reference and establish callbacks."""
        self.mark_view = mark_view
        if self.mark_view:
            self.mark_view.set_controller_callback(self)

    # --- Mark Service Request Methods ---

    def refresh_marks(self) -> None:
        """Refresh the marks list via service layer."""
        event = MarkGetAllRequestEventData()
        self.publish_event(event)

    def create_mark(self, name: Optional[str], x: int, y: int, description: Optional[str]) -> None:
        """Create a new mark via service layer."""
        event = MarkCreateRequestEventData(name=name, x=x, y=y, description=description)
        self.publish_event(event)

    def delete_mark_by_name(self, mark_name: str) -> None:
        """Delete a mark by name via service layer."""
        event = MarkDeleteByNameRequestEventData(name=mark_name)
        self.publish_event(event)

    def delete_all_marks(self) -> None:
        """Delete all marks via service layer."""
        event = MarkDeleteAllRequestEventData()
        self.publish_event(event)

    def execute_mark(self, identifier: Union[str, int]) -> None:
        """Execute a mark via service layer."""
        event = MarkExecuteRequestEventData(name_or_id=identifier)
        self.publish_event(event)

    def request_show_overlay(self) -> None:
        """Request mark visualization overlay via service layer."""
        event = MarkVisualizeAllRequestEventData()
        self.publish_event(event)

    def request_hide_overlay(self) -> None:
        """Request to hide mark visualization overlay via service layer."""
        event = MarkVisualizeCancelRequestEventData()
        self.publish_event(event)

    # --- Direct Mark View Methods ---

    def show_mark_overlay(self) -> None:
        """Directly show the mark overlay via view."""
        if self.mark_view:
            self.mark_view.show()
        else:
            self.logger.error("Cannot show mark overlay: mark view not set")

    def hide_mark_overlay(self) -> None:
        """Directly hide the mark overlay via view."""
        if self.mark_view:
            self.mark_view.hide()
        else:
            self.logger.error("Cannot hide mark overlay: mark view not set")

    def is_mark_overlay_active(self) -> bool:
        """Check if mark overlay is currently active."""
        return self.mark_view.is_active() if self.mark_view else False

    def update_mark_view_data(self, marks_list: List[MarkData]) -> None:
        """Update the mark view with new data."""
        if self.mark_view:
            self.mark_view.update_marks(marks_list)

    # --- Mark View Callback Methods ---

    def on_mark_visualization_shown(self) -> None:
        """Handle successful mark visualization show from view."""
        self._visualization_active = True
        state_event = MarkVisualizationStateChangedEventData(is_visible=True)
        self.publish_event(state_event)

    def on_mark_visualization_hidden(self) -> None:
        """Handle mark visualization hide from view."""
        self._visualization_active = False
        state_event = MarkVisualizationStateChangedEventData(is_visible=False)
        self.publish_event(state_event)

    def on_mark_visualization_failed(self, error_message: str) -> None:
        """Handle failed mark visualization from view."""
        self._visualization_active = False
        self.notify_status(f"Mark visualization failed: {error_message}", True)

    # --- Event Handlers ---

    async def _on_marks_changed(self, event):
        """Handle marks changed event."""
        if hasattr(event, "marks"):
            # Convert dictionary values to MarkData objects
            marks_list = []
            for mark_dict in event.marks.values():
                mark_data = MarkData(
                    name=mark_dict["name"], x=mark_dict["x"], y=mark_dict["y"], description=mark_dict.get("description", "")
                )
                marks_list.append(mark_data)

            # Update both the main view callback and the mark overlay view
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_marks_updated, marks_list)

            self.schedule_ui_update(self.update_mark_view_data, marks_list)
        else:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_marks_updated, [])
            self.schedule_ui_update(self.update_mark_view_data, [])

    async def _on_mark_operation_status(self, event):
        """Handle mark operation status events."""
        message = getattr(event, "message", "Mark operation completed.")
        is_error = not getattr(event, "success", True)
        self.notify_status(message, is_error)

    async def _handle_mark_list_changed(self, event) -> None:
        """Handle mark list changed events."""
        self.refresh_marks()

    async def _handle_mark_visualization_state_changed(self, event_data) -> None:
        """Handle mark visualization state changed event from service layer."""
        # Sync view state with service state
        if self.mark_view:
            if event_data.is_visible and not self.mark_view.is_active():
                self.show_mark_overlay()
            elif not event_data.is_visible and self.mark_view.is_active():
                self.hide_mark_overlay()

    def cleanup(self) -> None:
        """Clean up resources when controller is destroyed."""
        if self.mark_view:
            self.mark_view.cleanup()
        super().cleanup()
