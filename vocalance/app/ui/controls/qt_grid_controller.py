import asyncio
import logging

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
        grid_service,
        config: GlobalAppConfig,
    ):
        """Initialize grid controller.

        Args:
            event_bus: Event bus for pub/sub.
            grid_service: GridService instance.
            config: Global app configuration.
        """
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtGridController"),
        )

        self.grid_service = grid_service
        self.config = config
        self.grid_view = None

        self._subscribe_to_events()
        self.logger.debug("QtGridController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to grid bus events."""
        try:
            self.event_bus.subscribe(ShowGridRequestEventData, self._handle_show_grid_request)
            self.event_bus.subscribe(HideGridRequestEventData, self._handle_hide_grid_request)
            self.event_bus.subscribe(ClickGridCellRequestEventData, self._handle_click_grid_cell_request)
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def set_grid_view(self, grid_view) -> None:
        """Set the grid overlay view reference.

        Args:
            grid_view: QtGridView instance.
        """
        self.grid_view = grid_view
        if self.grid_view:
            self.logger.debug("Grid view reference set")

    def show_grid_overlay(self, num_rects=None, click_mode: str = "click") -> None:
        """Show the grid overlay with the given configuration.

        Args:
            num_rects: Number of grid cells, or None to use view default.
            click_mode: Interaction mode — 'click', 'hover', or 'drag'.
        """
        if self.grid_view:
            self.grid_view.show(num_rects, click_mode)
        else:
            self.logger.error("Cannot show grid overlay: grid view not set")

    def hide_grid_overlay(self) -> None:
        """Hide the grid overlay."""
        if self.grid_view:
            self.grid_view.hide()
        else:
            self.logger.error("Cannot hide grid overlay: grid view not set")

    def is_grid_overlay_active(self) -> bool:
        """Return True if the grid overlay is currently visible."""
        return self.grid_view.is_active() if self.grid_view else False

    def handle_grid_selection(self, selection_key: str, click_mode: str = "click") -> bool:
        """Delegate a cell selection to the grid view.

        Args:
            selection_key: Cell label to select.
            click_mode: Interaction mode for the selection.

        Returns:
            True if the selection was handled, False otherwise.
        """
        if self.grid_view:
            return self.grid_view.handle_selection(selection_key, click_mode)
        self.logger.error("Cannot handle grid selection: grid view not set")
        return False

    def on_grid_selection_success(self, selected_number: int, center_x: float, center_y: float) -> None:
        """Publish success events after a cell is selected by the view.

        Args:
            selected_number: The selected cell number.
            center_x: X coordinate of the cell centre.
            center_y: Y coordinate of the cell centre.
        """
        asyncio.ensure_future(
            self.event_bus.publish(
                GridInteractionSuccessEventData(
                    operation="select_cell",
                    details={"selected_number": str(selected_number), "x": center_x, "y": center_y},
                )
            )
        )
        asyncio.ensure_future(self.event_bus.publish(GridVisibilityChangedEventData(visible=False)))

    def on_grid_selection_failed(self, selected_number: int, error_message: str) -> None:
        """Publish a failure event after a cell selection attempt by the view.

        Args:
            selected_number: The cell number that failed.
            error_message: Description of the failure.
        """
        asyncio.ensure_future(
            self.event_bus.publish(
                GridInteractionFailedEventData(
                    operation="select_cell",
                    reason=error_message,
                    cell_label=str(selected_number),
                    details={"selected_number": str(selected_number)},
                )
            )
        )

    async def _handle_show_grid_request(self, event_data) -> None:
        """Handle a show-grid request event from the service."""
        num_rects = (event_data.rows * event_data.cols) if (event_data.rows and event_data.cols) else None
        self.show_grid_overlay(num_rects, getattr(event_data, "click_mode", "click"))

    async def _handle_hide_grid_request(self, event_data) -> None:
        """Handle a hide-grid request event from the service."""
        self.hide_grid_overlay()

    async def _handle_click_grid_cell_request(self, event_data) -> None:
        """Handle a click-grid-cell request event from the service."""
        if not self.is_grid_overlay_active():
            self.logger.warning(f"Grid not visible, cannot click cell {event_data.cell_label}")
            return
        if not self.grid_view:
            self.logger.error(f"Grid view not set, cannot click cell {event_data.cell_label}")
            return
        self.handle_grid_selection(event_data.cell_label, getattr(event_data, "click_mode", "click"))

    def cleanup(self) -> None:
        """Unsubscribe from all events, clean up the view, and release resources."""
        try:
            self.event_bus.unsubscribe(ShowGridRequestEventData, self._handle_show_grid_request)
            self.event_bus.unsubscribe(HideGridRequestEventData, self._handle_hide_grid_request)
            self.event_bus.unsubscribe(ClickGridCellRequestEventData, self._handle_click_grid_cell_request)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        if self.grid_view:
            self.grid_view.cleanup()

        super().cleanup()
