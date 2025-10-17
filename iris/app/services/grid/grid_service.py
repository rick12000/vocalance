import logging
import math
from typing import Any, Dict, Optional

from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.command_types import GridCancelCommand, GridSelectCommand, GridShowCommand
from iris.app.event_bus import EventBus
from iris.app.events.command_events import GridCommandParsedEvent
from iris.app.events.core_events import CommandExecutedStatusEvent
from iris.app.events.grid_events import (
    ClickGridCellRequestEventData,
    GridConfigUpdatedEventData,
    GridVisibilityChangedEventData,
    HideGridRequestEventData,
    ShowGridRequestEventData,
    UpdateGridConfigRequestEventData,
)
from iris.app.utils.event_utils import EventSubscriptionManager, ThreadSafeEventPublisher

logger = logging.getLogger(__name__)


class GridService:
    """Grid service for command processing and UI state management.

    Handles grid show/hide/select commands, calculates optimal grid dimensions,
    and manages grid configuration updates through event-driven architecture.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        self._event_bus = event_bus
        self._config = config
        self._visible: bool = False
        self.event_publisher = ThreadSafeEventPublisher(event_bus=event_bus)
        self.subscription_manager = EventSubscriptionManager(event_bus=event_bus, component_name="GridService")

        logger.info("GridService initialized")

    def setup_subscriptions(self) -> None:
        subscriptions = [
            (GridCommandParsedEvent, self._handle_grid_command),
            (UpdateGridConfigRequestEventData, self._handle_config_update),
        ]

        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)

        logger.info("GridService subscriptions set up")

    def _calculate_grid_dimensions(self, num_rects: int) -> tuple[int, int]:
        cols = math.ceil(math.sqrt(num_rects))
        rows = math.ceil(num_rects / cols)
        return rows, cols

    def _publish_visibility_event(self, visible: bool, rows: Optional[int] = None, cols: Optional[int] = None) -> None:
        self._visible = visible
        event = GridVisibilityChangedEventData(visible=visible, rows=rows, cols=cols)
        self.event_publisher.publish(event)

    def _publish_command_status(
        self, command_type: str, success: bool, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        status_event = CommandExecutedStatusEvent(
            command={"command_type": command_type, "details": details or {}}, success=success, message=message, source="grid"
        )
        self.event_publisher.publish(status_event)

    async def _handle_grid_command(self, event_data: GridCommandParsedEvent) -> None:
        command = event_data.command
        command_type = type(command).__name__

        try:
            if isinstance(command, GridShowCommand):
                num_rects = command.num_rects or self._config.grid.default_rect_count
                rows, cols = self._calculate_grid_dimensions(num_rects)

                # Publish show request and update visibility
                show_event = ShowGridRequestEventData(rows=rows, cols=cols)
                self.event_publisher.publish(show_event)
                self._publish_visibility_event(True, rows, cols)

                self._publish_command_status(command_type, True, f"Grid shown with {num_rects} cells", {"num_rects": num_rects})

            elif isinstance(command, GridSelectCommand):
                if not self._visible:
                    self._publish_command_status(command_type, False, "Grid not visible")
                    return

                # Publish cell click request
                click_event = ClickGridCellRequestEventData(cell_label=str(command.selected_number))
                self.event_publisher.publish(click_event)

                self._publish_command_status(
                    command_type,
                    True,
                    f"Grid cell {command.selected_number} selected",
                    {"selected_number": command.selected_number},
                )

            elif isinstance(command, GridCancelCommand):
                if self._visible:
                    hide_event = HideGridRequestEventData()
                    self.event_publisher.publish(hide_event)
                    self._publish_visibility_event(False)

                self._publish_command_status(command_type, True, "Grid hidden")

            else:
                logger.warning(f"Unknown grid command type: {command_type}")

        except Exception as e:
            logger.error(f"Error executing {command_type}: {e}", exc_info=True)
            self._publish_command_status(command_type, False, f"Error: {e}")

    async def _handle_config_update(self, event_data: UpdateGridConfigRequestEventData) -> None:
        config_fields = [
            "rows",
            "cols",
            "cell_width",
            "cell_height",
            "line_color",
            "label_color",
            "font_size",
            "font_name",
            "show_labels",
            "default_rect_count",
            "trigger_keyword",
            "cancel_phrases",
        ]

        updated_fields = {}
        for field in config_fields:
            value = getattr(event_data, field, None)
            if value is not None and hasattr(self._config.grid, field):
                if field == "cancel_phrases" and isinstance(value, list):
                    value = list(set(value))  # Remove duplicates
                setattr(self._config.grid, field, value)
                updated_fields[field] = value

        if updated_fields:
            # Create and publish config update event
            config_event = GridConfigUpdatedEventData(
                rows=self._config.grid.rows,
                cols=self._config.grid.cols,
                cell_width=self._config.grid.cell_width,
                cell_height=self._config.grid.cell_height,
                line_color=self._config.grid.line_color,
                label_color=self._config.grid.label_color,
                font_size=self._config.grid.font_size,
                font_name=self._config.grid.font_name,
                show_labels=self._config.grid.show_labels,
                default_rect_count=self._config.grid.default_rect_count,
                trigger_keyword=self._config.grid.trigger_keyword,
                cancel_phrases=list(self._config.grid.cancel_phrases),
                message=f"Updated: {list(updated_fields.keys())}",
            )
            self.event_publisher.publish(config_event)
            logger.info(f"Grid config updated: {updated_fields}")

    def is_grid_visible(self) -> bool:
        return self._visible

    def get_current_config(self):
        return self._config.grid

    async def shutdown(self) -> None:
        logger.info("Shutting down GridService")
        self.subscription_manager.unsubscribe_all()
