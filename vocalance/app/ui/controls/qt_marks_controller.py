"""Qt-based marks controller - EXACT LEGACY MATCH.

Manages mark visualization overlay and mark data loading/manipulation matching legacy CustomTkinter implementation.
"""

import asyncio
import logging
import threading
from typing import List, Optional, Union

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus

# Ensure MarkData import is available
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
    """Controller for marks functionality - orchestrates between service and view.

    Thread Safety:
    - _visualization_active protected by _state_lock
    - Event handlers run in GUI event loop thread
    - UI updates marshalled to main thread via signals
    """

    # Signals for marks operations
    marks_loaded = Signal(list)  # List of MarkData objects
    mark_created = Signal(str, int, int)  # name, x, y
    mark_deleted = Signal(str)  # name
    all_marks_deleted = Signal()
    mark_overlay_shown = Signal()
    mark_overlay_hidden = Signal()
    operation_error = Signal(str)
    status_updated = Signal(str, bool)  # message, is_error

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        mark_service,
        config: GlobalAppConfig,
        main_window,
    ):
        """Initialize marks controller.

        Args:
            event_bus: Event bus for pub/sub.
            event_loop: Asyncio event loop.
            mark_service: Mark service instance.
            config: Global app configuration.
            main_window: Main window reference.
        """
        super().__init__(
            event_bus=event_bus,
            event_loop=event_loop,
            logger=logging.getLogger("QtMarksController"),
        )

        self.mark_service = mark_service
        self.config = config
        self.main_window = main_window

        # Mark view reference (will be set by main window)
        self.mark_view = None

        # Mark visualization state tracking (protected by _state_lock)
        self._state_lock = threading.RLock()
        self._visualization_active = False

        # Cache of marks for overlay display
        self.marks_list: List[MarkData] = []

        # Subscribe to mark service events
        self._subscribe_to_events()

        self.logger.debug("QtMarksController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to mark-related events using exact legacy event types."""
        try:
            self.event_bus.subscribe(MarksChangedEventData, self._on_marks_changed)
            self.event_bus.subscribe(MarkOperationSuccessEventData, self._on_mark_operation_status)
            self.event_bus.subscribe(MarkOperationFailedEventData, self._on_mark_operation_status)
            self.event_bus.subscribe(MarkCreatedEventData, self._handle_mark_list_changed)
            self.event_bus.subscribe(MarkDeletedEventData, self._handle_mark_list_changed)
            self.event_bus.subscribe(AllMarksClearedEventData, self._handle_mark_list_changed)
            self.event_bus.subscribe(MarkVisualizationStateChangedEventData, self._handle_mark_visualization_state_changed)
            self.logger.debug("Subscribed to mark events (legacy types)")
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def set_mark_view(self, mark_view):
        """Set the mark view reference and establish callbacks."""
        self.mark_view = mark_view
        if self.mark_view:
            # Mark view will call controller methods directly
            self.logger.debug("Mark view reference set")

    # --- Mark Service Request Methods (Publish Events) ---

    def refresh_marks(self) -> None:
        """Refresh the marks list via service layer."""
        event = MarkGetAllRequestEventData()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def create_mark(self, name: Optional[str], x: int, y: int, description: Optional[str] = None) -> None:
        """Create a new mark via service layer."""
        event = MarkCreateRequestEventData(name=name, x=x, y=y, description=description)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def delete_mark_by_name(self, mark_name: str) -> None:
        """Delete a mark by name via service layer."""
        event = MarkDeleteByNameRequestEventData(name=mark_name)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def delete_all_marks(self) -> None:
        """Delete all marks via service layer."""
        event = MarkDeleteAllRequestEventData()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def execute_mark(self, identifier: Union[str, int]) -> None:
        """Execute a mark via service layer."""
        event = MarkExecuteRequestEventData(name_or_id=identifier)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def request_show_overlay(self) -> None:
        """Request mark visualization overlay via service layer."""
        event = MarkVisualizeAllRequestEventData()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def request_hide_overlay(self) -> None:
        """Request to hide mark visualization overlay via service layer."""
        event = MarkVisualizeCancelRequestEventData()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    # --- Direct Mark View Methods ---

    def show_mark_overlay(self) -> None:
        """Directly show the mark overlay via view."""
        if not self.mark_view:
            self.logger.error("Cannot show mark overlay: mark view not set")
            return

        # CRITICAL: Must ensure marks are freshly loaded from service BEFORE showing
        self.logger.info("show_mark_overlay: Starting mark fetch...")

        # Use coroutine to fetch fresh marks then show
        asyncio.run_coroutine_threadsafe(self._show_mark_overlay_async(), self.event_loop)

    async def _show_mark_overlay_async(self) -> None:
        """Async helper to ensure marks are loaded before showing overlay."""
        self.logger.info("_show_mark_overlay_async: Starting...")
        self.logger.info(f"_show_mark_overlay_async: marks_list before refresh: {self.marks_list}")

        # Explicitly request all marks from service
        self.logger.info("_show_mark_overlay_async: Requesting marks from service via refresh_marks()")
        self.refresh_marks()

        # Wait for marks to be delivered and processed
        # The MarksChangedEventData handler (_on_marks_changed) will populate self.marks_list
        self.logger.info("_show_mark_overlay_async: Waiting for marks to arrive...")
        for attempt in range(50):  # 500ms timeout with 10ms checks
            await asyncio.sleep(0.01)
            self.logger.debug(f"_show_mark_overlay_async: Attempt {attempt}: marks_list={self.marks_list}")
            if self.marks_list:
                self.logger.info(f"_show_mark_overlay_async: Got marks on attempt {attempt}: {len(self.marks_list)} marks")
                for mark in self.marks_list:
                    self.logger.info(f"  - Mark: name='{mark.name}', x={mark.x}, y={mark.y}")
                break
        else:
            self.logger.warning("_show_mark_overlay_async: Timeout waiting for marks from service")
            self.logger.warning(f"_show_mark_overlay_async: Final marks_list: {self.marks_list}")

        self.logger.info(f"_show_mark_overlay_async: Setting {len(self.marks_list)} marks in view and showing")

        # Now update view with fresh marks and show
        if self.mark_view:
            marks_dict = {mark.name: (mark.x, mark.y) for mark in self.marks_list}
            self.logger.info(f"_show_mark_overlay_async: Created marks dict: {marks_dict}")
            self.mark_view.marks = marks_dict
            self.logger.info(f"_show_mark_overlay_async: Marks set in view: {self.mark_view.marks}")
            self.logger.info("_show_mark_overlay_async: Emitting show_requested signal")
            self.mark_view.show_requested.emit()
        else:
            self.logger.error("_show_mark_overlay_async: mark_view is None!")

    def hide_mark_overlay(self) -> None:
        """Directly hide the mark overlay via view."""
        if self.mark_view:
            # Emit signal to hide on Qt main thread
            self.mark_view.hide_requested.emit()
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
        """Handle successful mark visualization show from view. Thread-safe and non-blocking."""
        with self._state_lock:
            self._visualization_active = True
        # Schedule event publish without blocking Qt main thread
        state_event = MarkVisualizationStateChangedEventData(is_visible=True)
        try:
            # Use call_soon_threadsafe for immediate non-blocking scheduling
            self.event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self.event_bus.publish(state_event)))
        except Exception as e:
            self.logger.error(f"Error scheduling mark visualization shown event: {e}")

    def on_mark_visualization_hidden(self) -> None:
        """Handle mark visualization hide from view. Thread-safe and non-blocking."""
        with self._state_lock:
            self._visualization_active = False
        # Schedule event publish without blocking Qt main thread
        state_event = MarkVisualizationStateChangedEventData(is_visible=False)
        try:
            # Use call_soon_threadsafe for immediate non-blocking scheduling
            self.event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self.event_bus.publish(state_event)))
        except Exception as e:
            self.logger.error(f"Error scheduling mark visualization hidden event: {e}")

    def on_mark_visualization_failed(self, error_message: str) -> None:
        """Handle failed mark visualization from view. Thread-safe."""
        with self._state_lock:
            self._visualization_active = False
        self.notify_status(f"Mark visualization failed: {error_message}", True)

    # --- Event Handlers ---

    async def _on_marks_changed(self, event):
        """Handle marks changed event."""
        self.logger.info(f"_on_marks_changed: Received event, hasattr(marks)={hasattr(event, 'marks')}")
        if hasattr(event, "marks"):
            marks_list = []
            self.logger.info(f"_on_marks_changed: event.marks = {event.marks}")
            for mark_dict in event.marks.values():
                mark_data = MarkData(
                    name=mark_dict["name"], x=mark_dict["x"], y=mark_dict["y"], description=mark_dict.get("description", "")
                )
                marks_list.append(mark_data)

            # Store in instance variable for later use
            self.marks_list = marks_list
            self.logger.info(f"_on_marks_changed: Stored {len(self.marks_list)} marks in controller cache")

            # Emit signal for view
            self.marks_loaded.emit(marks_list)

            # Update overlay if active
            if self.mark_view:
                self.logger.info(f"_on_marks_changed: Mark view is set, updating with {len(marks_list)} marks")
                self.update_mark_view_data(marks_list)
        else:
            self.logger.warning("_on_marks_changed: No marks attribute in event")
            self.marks_list = []
            self.marks_loaded.emit([])
            if self.mark_view:
                self.update_mark_view_data([])

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
        if self.mark_view:
            if event_data.is_visible and not self.mark_view.is_active():
                # CRITICAL: Ensure marks are loaded BEFORE showing overlay
                # Use same polling pattern as show_mark_overlay() to guarantee marks arrival
                self.logger.info("_handle_mark_visualization_state_changed: Visualization requested - ensuring marks are loaded")

                # Explicitly request all marks from service
                self.logger.info("_handle_mark_visualization_state_changed: Requesting marks from service via refresh_marks()")
                self.refresh_marks()

                # Poll with timeout for marks to arrive
                # The MarksChangedEventData handler (_on_marks_changed) will populate self.marks_list
                self.logger.info("_handle_mark_visualization_state_changed: Waiting for marks to arrive...")
                for attempt in range(50):  # 500ms timeout with 10ms checks
                    await asyncio.sleep(0.01)
                    self.logger.debug(
                        f"_handle_mark_visualization_state_changed: Attempt {attempt}: marks_list={len(self.marks_list)}"
                    )
                    if self.marks_list:
                        self.logger.info(
                            f"_handle_mark_visualization_state_changed: Got marks on attempt {attempt}: {len(self.marks_list)} marks"
                        )
                        for mark in self.marks_list:
                            self.logger.info(f"  - Mark: name='{mark.name}', x={mark.x}, y={mark.y}")
                        break
                else:
                    self.logger.warning("_handle_mark_visualization_state_changed: Timeout waiting for marks from service")
                    self.logger.warning(f"_handle_mark_visualization_state_changed: Final marks_list: {len(self.marks_list)}")

                self.logger.info(
                    f"_handle_mark_visualization_state_changed: Setting {len(self.marks_list)} marks in view and showing"
                )

                # Set marks in view before showing (guaranteed to be populated now)
                if self.mark_view:
                    marks_dict = {mark.name: (mark.x, mark.y) for mark in self.marks_list}
                    self.logger.info(f"_handle_mark_visualization_state_changed: Created marks dict: {marks_dict}")
                    self.mark_view.marks = marks_dict
                    self.logger.info("_handle_mark_visualization_state_changed: Marks set in view, emitting show_requested")
                    # Marshal to Qt main thread via signals - don't call directly from async handler
                    self.mark_view.show_requested.emit()

            elif not event_data.is_visible and self.mark_view.is_active():
                # Marshal to Qt main thread via signals - don't call directly from async handler
                self.mark_view.hide_requested.emit()

    def notify_status(self, message: str, is_error: bool = False) -> None:
        """Notify status message."""
        self.status_updated.emit(message, is_error)
        if is_error:
            self.operation_error.emit(message)

    def cleanup(self) -> None:
        """Clean up resources when controller is destroyed."""
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
