import logging
from typing import Optional
import asyncio
from iris.app.ui.controls.base_control import BaseController
from iris.app.events.grid_events import (
    ShowGridRequestEventData, HideGridRequestEventData, ClickGridCellRequestEventData,
    UpdateGridConfigRequestEventData, GridVisibilityChangedEventData, GridConfigUpdatedEventData,
    GridInteractionSuccessEventData, GridInteractionFailedEventData
)
from iris.app.events.core_events import ClickLoggedEventData


class GridController(BaseController):
    """Controller for grid functionality - orchestrates between service and view."""
    
    def __init__(self, event_bus, event_loop, logger):
        super().__init__(event_bus, event_loop, logger, "GridController")
        
        # Grid view reference (will be set by main window)
        self.grid_view = None
        
        # Grid state tracking
        self._grid_visible = False
        
        self.subscribe_to_events([
            (ShowGridRequestEventData, self._handle_show_grid_request),
            (HideGridRequestEventData, self._handle_hide_grid_request),
            (GridVisibilityChangedEventData, self._handle_grid_visibility_changed),
            (GridConfigUpdatedEventData, self._handle_grid_config_updated),
            (GridInteractionSuccessEventData, self._handle_grid_interaction_status),
            (GridInteractionFailedEventData, self._handle_grid_interaction_status),
            (ClickLoggedEventData, self._handle_click_logged),
            (ClickGridCellRequestEventData, self._handle_click_grid_cell_request),
        ])

    def set_grid_view(self, grid_view):
        """Set the grid view reference and establish callbacks."""
        self.grid_view = grid_view
        if self.grid_view:
            self.grid_view.set_controller_callback(self)

    # --- Grid Service Request Methods ---

    def request_show_grid(self, rows: Optional[int] = None, cols: Optional[int] = None) -> None:
        """Request to show the grid via service layer."""
        event = ShowGridRequestEventData(rows=rows, cols=cols)
        self.publish_event(event)

    def request_hide_grid(self) -> None:
        """Request to hide the grid via service layer."""
        event = HideGridRequestEventData()
        self.publish_event(event)

    def request_click_grid_cell(self, cell_label: str) -> None:
        """Request to click a grid cell via service layer."""
        event = ClickGridCellRequestEventData(cell_label=cell_label)
        self.publish_event(event)

    def request_update_grid_config(self, rows: Optional[int] = None, cols: Optional[int] = None, 
                                 show_numbers: Optional[bool] = None) -> None:
        """Request to update grid configuration via service layer."""
        event = UpdateGridConfigRequestEventData(rows=rows, cols=cols, show_numbers=show_numbers)
        self.publish_event(event)

    # --- Direct Grid View Methods ---

    def show_grid_overlay(self, num_rects: Optional[int] = None) -> None:
        """Directly show the grid overlay via view."""
        if self.grid_view:
            self.grid_view.show(num_rects)
        else:
            self.logger.error("Cannot show grid overlay: grid view not set")

    def hide_grid_overlay(self) -> None:
        """Directly hide the grid overlay via view."""
        if self.grid_view:
            self.grid_view.hide()
        else:
            self.logger.error("Cannot hide grid overlay: grid view not set")

    def refresh_grid_overlay(self) -> None:
        """Refresh the grid overlay display."""
        if self.grid_view:
            self.grid_view.refresh_display()

    def is_grid_overlay_active(self) -> bool:
        """Check if grid overlay is currently active."""
        return self.grid_view.is_active() if self.grid_view else False

    def handle_grid_selection(self, selection_key: str) -> bool:
        """Handle grid cell selection via view."""
        if self.grid_view:
            return self.grid_view.handle_selection(selection_key)
        else:
            self.logger.error("Cannot handle grid selection: grid view not set")
            return False

    # --- Grid View Callback Methods ---

    def on_grid_selection_success(self, selected_number: int, center_x: float, center_y: float) -> None:
        """Handle successful grid selection from view."""
        self._grid_visible = False
        
        # Publish interaction success event
        interaction_event = GridInteractionSuccessEventData(
            operation="select_cell", 
            details={"selected_number": str(selected_number), "x": center_x, "y": center_y}
        )
        self.publish_event(interaction_event)
        
        # Publish visibility changed event
        visibility_event = GridVisibilityChangedEventData(visible=False)
        self.publish_event(visibility_event)

    def on_grid_selection_failed(self, selected_number: int, error_message: str) -> None:
        """Handle failed grid selection from view."""
        interaction_event = GridInteractionFailedEventData(
            operation="select_cell",
            reason=error_message,
            cell_label=str(selected_number),
            details={"selected_number": str(selected_number)}
        )
        self.publish_event(interaction_event)

    # --- Event Handlers ---

    async def _handle_show_grid_request(self, event_data) -> None:
        """Handle request to show the grid."""
        num_rects = None
        if event_data.rows and event_data.cols:
            num_rects = event_data.rows * event_data.cols
        
        self.show_grid_overlay(num_rects)

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
        
        self.handle_grid_selection(event_data.cell_label)

    async def _handle_click_logged(self, event_data) -> None:
        """Handle click logged event to refresh grid if visible."""
        if self._grid_visible and self.is_grid_overlay_active():
            self.refresh_grid_overlay()

    async def _handle_grid_visibility_changed(self, event_data) -> None:
        """Handle grid visibility changed event."""
        self._grid_visible = event_data.visible
        
        # Sync view state with service state
        if self.grid_view:
            if event_data.visible and not self.grid_view.is_active():
                num_rects = None
                if event_data.rows and event_data.cols:
                    num_rects = event_data.rows * event_data.cols
                self.show_grid_overlay(num_rects)
            elif not event_data.visible and self.grid_view.is_active():
                self.hide_grid_overlay()
        
        # Notify main window callback
        if self.view_callback:
            self.view_callback.on_grid_visibility_changed(
                event_data.visible, 
                event_data.rows, 
                event_data.cols,
                getattr(event_data, 'show_numbers', None)
            )

    async def _handle_grid_config_updated(self, event_data) -> None:
        """Handle grid config updated event."""
        if self.view_callback:
            self.view_callback.on_grid_config_updated(event_data)

    async def _handle_grid_interaction_status(self, event_data) -> None:
        """Handle grid interaction status events."""
        if isinstance(event_data, GridInteractionSuccessEventData):
            self.logger.info(f"Grid interaction success: {event_data.operation}")
        else:
            self.logger.error(f"Grid interaction failed: {event_data.operation} - {event_data.reason}")

    def cleanup(self) -> None:
        """Clean up resources when controller is destroyed."""
        if self.grid_view:
            self.grid_view.cleanup()
        super().cleanup() 