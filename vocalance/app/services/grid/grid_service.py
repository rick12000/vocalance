import asyncio
import logging
import math
from typing import Any, Dict, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import GridSelectCommand, GridShowCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import GridCommandParsedEvent
from vocalance.app.events.core_events import CommandExecutedStatusEvent
from vocalance.app.events.grid_events import (
    ClickGridCellRequestEventData,
    GridConfigUpdatedEventData,
    GridVisibilityChangedEventData,
    ShowGridRequestEventData,
    UpdateGridConfigRequestEventData,
)
from vocalance.app.utils.event_utils import EventSubscriptionManager, ThreadSafeEventPublisher

logger = logging.getLogger(__name__)


class GridService:
    """Grid service for command processing and UI state management.

    Handles grid show/hide/select commands, calculates optimal grid dimensions,
    and manages grid configuration updates through event-driven architecture.
    All state access is protected with async locks for thread safety.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        """Initialize grid service with dependencies.

        Args:
            event_bus: EventBus for pub/sub messaging.
            config: Global application configuration.
        """
        self._event_bus = event_bus
        self._config = config
        self._visible: bool = False
        self._state_lock = asyncio.Lock()
        self.event_publisher = ThreadSafeEventPublisher(event_bus=event_bus)
        self.subscription_manager = EventSubscriptionManager(event_bus=event_bus, component_name="GridService")

        logger.debug("GridService initialized")

    def setup_subscriptions(self) -> None:
        subscriptions = [
            (GridCommandParsedEvent, self._handle_grid_command),
            (UpdateGridConfigRequestEventData, self._handle_config_update),
        ]

        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)

        logger.debug("GridService subscriptions set up")

    def _calculate_grid_dimensions(self, num_rects: int) -> tuple[int, int]:
        cols = math.ceil(math.sqrt(num_rects))
        rows = math.ceil(num_rects / cols)
        return rows, cols

    async def _publish_visibility_event(self, visible: bool, rows: Optional[int] = None, cols: Optional[int] = None) -> None:
        async with self._state_lock:
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

        if isinstance(command, GridShowCommand):
            num_rects = command.num_rects or self._config.grid.default_rect_count
            rows, cols = self._calculate_grid_dimensions(num_rects)

            show_event = ShowGridRequestEventData(rows=rows, cols=cols)
            self.event_publisher.publish(show_event)
            await self._publish_visibility_event(True, rows, cols)

            self._publish_command_status(command_type, True, f"Grid shown with {num_rects} cells", {"num_rects": num_rects})

        elif isinstance(command, GridSelectCommand):
            async with self._state_lock:
                is_visible = self._visible

            if not is_visible:
                self._publish_command_status(command_type, False, "Grid not visible")
                return

            click_event = ClickGridCellRequestEventData(cell_label=str(command.selected_number))
            self.event_publisher.publish(click_event)

            self._publish_command_status(
                command_type,
                True,
                f"Grid cell {command.selected_number} selected",
                {"selected_number": command.selected_number},
            )

        else:
            logger.warning(f"Unknown grid command type: {command_type}")

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
                message=f"Updated: {list(updated_fields.keys())}",
            )
            self.event_publisher.publish(config_event)
            logger.info(f"Grid config updated: {updated_fields}")

    async def is_grid_visible(self) -> bool:
        async with self._state_lock:
            return self._visible

    def get_current_config(self):
        return self._config.grid

    async def shutdown(self) -> None:
        logger.info("Shutting down GridService")
        self.subscription_manager.unsubscribe_all()
