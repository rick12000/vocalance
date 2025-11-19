"""Qt-based grid controller - EXACT LEGACY MATCH.

Controller for grid functionality - orchestrates between service and view with click tracking.
"""

import asyncio
import logging
import threading
from typing import Optional

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import ClickLoggedEventData
from vocalance.app.events.grid_events import (
    ClickGridCellRequestEventData,
    GridConfigUpdatedEventData,
    GridInteractionFailedEventData,
    GridInteractionSuccessEventData,
    GridVisibilityChangedEventData,
    HideGridRequestEventData,
    ShowGridRequestEventData,
    UpdateGridConfigRequestEventData,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtGridController(QtBaseController):
    """Controller for grid functionality - orchestrates between service and view.

    Thread Safety:
    - _grid_visible protected by _state_lock
    - Event handlers run in GUI event loop thread
    - UI updates marshalled to main thread via signals
    """

    # Signals for grid operations
    grid_overlay_shown = Signal()
    grid_overlay_hidden = Signal()
    grid_visibility_changed = Signal(bool, object, object)  # visible, rows, cols
    grid_config_updated = Signal(object)  # event_data
    grid_interaction_success = Signal(str, dict)  # operation, details
    grid_interaction_failed = Signal(str, str, str)  # operation, reason, cell_label
    operation_error = Signal(str)

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        grid_service,
        config: GlobalAppConfig,
        main_window,
        storage_service=None,
    ):
        """Initialize grid controller.

        Args:
            event_bus: Event bus for pub/sub.
            event_loop: Asyncio event loop.
            grid_service: Grid service instance.
            config: Global app configuration.
            main_window: Main window reference.
            storage_service: Storage service for click tracking (optional).
        """
        super().__init__(
            event_bus=event_bus,
            event_loop=event_loop,
            logger=logging.getLogger("QtGridController"),
        )

        self.grid_service = grid_service
        self.config = config
        self.main_window = main_window
        self.storage_service = storage_service

        # Grid view reference (will be set by main window)
        self.grid_view = None

        # Grid state tracking (protected by _state_lock)
        self._state_lock = threading.RLock()
        self._grid_visible = False
        self._current_click_mode = "click"  # Track current click mode ("click" or "hover")

        # Subscribe to grid events
        self._subscribe_to_events()

        self.logger.debug("QtGridController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to grid-related events using exact legacy event types."""
        try:
            self.event_bus.subscribe(ShowGridRequestEventData, self._handle_show_grid_request)
            self.event_bus.subscribe(HideGridRequestEventData, self._handle_hide_grid_request)
            self.event_bus.subscribe(GridVisibilityChangedEventData, self._handle_grid_visibility_changed)
            self.event_bus.subscribe(GridConfigUpdatedEventData, self._handle_grid_config_updated)
            self.event_bus.subscribe(GridInteractionSuccessEventData, self._handle_grid_interaction_status)
            self.event_bus.subscribe(GridInteractionFailedEventData, self._handle_grid_interaction_status)
            self.event_bus.subscribe(ClickLoggedEventData, self._handle_click_logged)
            self.event_bus.subscribe(ClickGridCellRequestEventData, self._handle_click_grid_cell_request)
            self.logger.debug("Subscribed to grid events (legacy types)")
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def set_grid_view(self, grid_view):
        """Set the grid view reference and establish callbacks."""
        self.grid_view = grid_view
        if self.grid_view:
            self.logger.debug("Grid view reference set")

    # --- Grid Service Request Methods (Publish Events) ---

    def request_show_grid(self, rows: Optional[int] = None, cols: Optional[int] = None, click_mode: str = "click") -> None:
        """Request to show the grid via service layer."""
        event = ShowGridRequestEventData(rows=rows, cols=cols, click_mode=click_mode)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def request_hide_grid(self) -> None:
        """Request to hide the grid via service layer."""
        event = HideGridRequestEventData()
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def request_click_grid_cell(self, cell_label: str, click_mode: str = "click") -> None:
        """Request to click a grid cell via service layer."""
        event = ClickGridCellRequestEventData(cell_label=cell_label, click_mode=click_mode)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    def request_update_grid_config(
        self, rows: Optional[int] = None, cols: Optional[int] = None, show_numbers: Optional[bool] = None
    ) -> None:
        """Request to update grid configuration via service layer."""
        event = UpdateGridConfigRequestEventData(rows=rows, cols=cols, show_labels=show_numbers)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)

    # --- Direct Grid View Methods ---

    def show_grid_overlay(self, num_rects: Optional[int] = None, click_mode: str = "click") -> None:
        """Directly show the grid overlay via view."""
        if self.grid_view:
            self._current_click_mode = click_mode
            self.grid_view.show(num_rects, click_mode)
            self.grid_overlay_shown.emit()
        else:
            self.logger.error("Cannot show grid overlay: grid view not set")

    def hide_grid_overlay(self) -> None:
        """Directly hide the grid overlay via view."""
        if self.grid_view:
            self.grid_view.hide()
            self.grid_overlay_hidden.emit()
        else:
            self.logger.error("Cannot hide grid overlay: grid view not set")

    def refresh_grid_overlay(self) -> None:
        """Refresh the grid overlay display."""
        if self.grid_view:
            self.grid_view.refresh_display()

    def is_grid_overlay_active(self) -> bool:
        """Check if grid overlay is currently active."""
        return self.grid_view.is_active() if self.grid_view else False

    def handle_grid_selection(self, selection_key: str, click_mode: str = "click") -> bool:
        """Handle grid cell selection via view."""
        if self.grid_view:
            return self.grid_view.handle_selection(selection_key, click_mode)
        else:
            self.logger.error("Cannot handle grid selection: grid view not set")
            return False

    # --- Grid View Callback Methods ---

    def on_grid_selection_success(self, selected_number: int, center_x: float, center_y: float) -> None:
        """Handle successful grid selection from view. Thread-safe."""
        with self._state_lock:
            self._grid_visible = False

        # Publish interaction success event
        interaction_event = GridInteractionSuccessEventData(
            operation="select_cell", details={"selected_number": str(selected_number), "x": center_x, "y": center_y}
        )
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(interaction_event), self.event_loop)

        # Publish visibility changed event
        visibility_event = GridVisibilityChangedEventData(visible=False)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(visibility_event), self.event_loop)

        # Emit signal
        self.grid_interaction_success.emit("select_cell", {"selected_number": str(selected_number), "x": center_x, "y": center_y})

    def on_grid_selection_failed(self, selected_number: int, error_message: str) -> None:
        """Handle failed grid selection from view."""
        interaction_event = GridInteractionFailedEventData(
            operation="select_cell",
            reason=error_message,
            cell_label=str(selected_number),
            details={"selected_number": str(selected_number)},
        )
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(interaction_event), self.event_loop)

        # Emit signal
        self.grid_interaction_failed.emit("select_cell", error_message, str(selected_number))

    # --- Event Handlers ---

    async def _handle_show_grid_request(self, event_data) -> None:
        """Handle request to show the grid."""
        num_rects = None
        if event_data.rows and event_data.cols:
            num_rects = event_data.rows * event_data.cols

        # Pass click_mode to grid view
        click_mode = getattr(event_data, "click_mode", "click")
        self.show_grid_overlay(num_rects, click_mode)

    async def _handle_hide_grid_request(self, event_data) -> None:
        """Handle request to hide the grid."""
        self.hide_grid_overlay()

    async def _handle_click_grid_cell_request(self, event_data) -> None:
        """Handle request to click a grid cell by label."""
        if not self._grid_visible:
            self.logger.warning(f"Grid not visible, cannot click cell {event_data.cell_label}")
            return

        if not self.grid_view:
            self.logger.error(f"Grid view not set, cannot click cell {event_data.cell_label}")
            return

        # Get click_mode from event
        click_mode = getattr(event_data, "click_mode", "click")
        self.handle_grid_selection(event_data.cell_label, click_mode)

    async def _handle_click_logged(self, event_data) -> None:
        """Handle click logged event to refresh grid if visible. Thread-safe."""
        with self._state_lock:
            grid_visible = self._grid_visible

        if grid_visible and self.is_grid_overlay_active():
            self.refresh_grid_overlay()

    async def _handle_grid_visibility_changed(self, event_data) -> None:
        """Handle grid visibility changed event. Thread-safe."""
        with self._state_lock:
            self._grid_visible = event_data.visible

        if self.grid_view:
            if event_data.visible and not self.grid_view.is_active():
                num_rects = None
                if event_data.rows and event_data.cols:
                    num_rects = event_data.rows * event_data.cols
                self.show_grid_overlay(num_rects)
            elif not event_data.visible and self.grid_view.is_active():
                self.hide_grid_overlay()

        # Emit signal
        self.grid_visibility_changed.emit(event_data.visible, event_data.rows, event_data.cols)

    async def _handle_grid_config_updated(self, event_data) -> None:
        """Handle grid config updated event."""
        self.grid_config_updated.emit(event_data)

    async def _handle_grid_interaction_status(self, event_data) -> None:
        """Handle grid interaction status events."""
        if isinstance(event_data, GridInteractionSuccessEventData):
            self.logger.info(f"Grid interaction success: {event_data.operation}")
        else:
            self.logger.error(f"Grid interaction failed: {event_data.operation} - {event_data.reason}")

    # --- Click Tracking Integration ---

    async def initialize_click_cache(self) -> None:
        """Load historical click data from storage into cache. Thread-safe."""
        if not self.storage_service:
            self.logger.warning("Storage service not available for click cache")
            return

        if self.grid_view and hasattr(self.grid_view, "initialize_click_cache"):
            try:
                await self.grid_view.initialize_click_cache()
                self.logger.info("Grid click cache initialized")
            except Exception as e:
                self.logger.error(f"Error initializing click cache: {e}", exc_info=True)

    def cleanup(self) -> None:
        """Clean up resources when controller is destroyed."""
        try:
            self.event_bus.unsubscribe(ShowGridRequestEventData, self._handle_show_grid_request)
            self.event_bus.unsubscribe(HideGridRequestEventData, self._handle_hide_grid_request)
            self.event_bus.unsubscribe(GridVisibilityChangedEventData, self._handle_grid_visibility_changed)
            self.event_bus.unsubscribe(GridConfigUpdatedEventData, self._handle_grid_config_updated)
            self.event_bus.unsubscribe(GridInteractionSuccessEventData, self._handle_grid_interaction_status)
            self.event_bus.unsubscribe(GridInteractionFailedEventData, self._handle_grid_interaction_status)
            self.event_bus.unsubscribe(ClickLoggedEventData, self._handle_click_logged)
            self.event_bus.unsubscribe(ClickGridCellRequestEventData, self._handle_click_grid_cell_request)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        if self.grid_view:
            self.grid_view.cleanup()

        super().cleanup()
