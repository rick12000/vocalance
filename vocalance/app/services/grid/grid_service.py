import asyncio
import logging
import math

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import GridSelectCommand, GridShowCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import GridCommandParsedEvent
from vocalance.app.events.grid_events import GridStateEvent
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
        self._current_click_mode: str = "click"  # Track current click mode
        self._state_lock = asyncio.Lock()
        self.event_publisher = ThreadSafeEventPublisher(event_bus=event_bus)
        self.subscription_manager = EventSubscriptionManager(event_bus=event_bus, component_name="GridService")

        logger.debug("GridService initialized")

    def setup_subscriptions(self) -> None:
        subscriptions = [
            (GridCommandParsedEvent, self._handle_grid_command),
            (GridStateEvent, self._handle_grid_state_event),
        ]

        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)

        logger.debug("GridService subscriptions set up")

    def _calculate_grid_dimensions(self, num_rects: int) -> tuple[int, int]:
        """Calculate optimal grid dimensions for given number of rectangles.

        Uses square root approximation to create a nearly square grid layout.

        Args:
            num_rects: Number of cells to fit in grid.

        Returns:
            Tuple of (rows, cols) dimensions.
        """
        cols = math.ceil(math.sqrt(num_rects))
        rows = math.ceil(num_rects / cols)
        return rows, cols

    async def _publish_visibility_event(self, visible: bool) -> None:
        """Update internal visibility state."""
        async with self._state_lock:
            self._visible = visible

    async def _handle_grid_command(self, event_data: GridCommandParsedEvent) -> None:
        """Handle grid commands (show/select) with mode-specific processing."""
        command = event_data.command
        command_type = type(command).__name__

        if isinstance(command, GridShowCommand):
            num_rects = command.num_rects or self._config.grid.default_rect_count
            rows, cols = self._calculate_grid_dimensions(num_rects)
            click_mode = command.click_mode

            async with self._state_lock:
                self._current_click_mode = click_mode

            # CRITICAL PATH OPTIMIZATION: Publish show event immediately
            # The controller will handle showing the grid synchronously
            show_event = GridStateEvent(state="visible", config={"rows": rows, "cols": cols, "click_mode": click_mode})
            self.event_publisher.publish(show_event)

            # Publish visibility event for state tracking (non-blocking)
            # This does NOT trigger another show operation (fixed in controller)
            await self._publish_visibility_event(True)

        elif isinstance(command, GridSelectCommand):
            async with self._state_lock:
                is_visible = self._visible

            if not is_visible:
                return

            # Get the click_mode from the most recent GridShowCommand (stored in state)
            async with self._state_lock:
                click_mode = self._current_click_mode

            click_event = GridStateEvent(
                state="interaction_request", config={"cell_label": str(command.selected_number), "click_mode": click_mode}
            )
            self.event_publisher.publish(click_event)

        else:
            logger.warning(f"Unknown grid command type: {command_type}")

    async def _handle_grid_state_event(self, event_data: GridStateEvent) -> None:
        """Handle grid state events (like config updates)."""
        if event_data.state == "config_updated" and event_data.config:
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
                value = event_data.config.get(field)
                if value is not None and hasattr(self._config.grid, field):
                    if field == "cancel_phrases" and isinstance(value, list):
                        value = list(set(value))
                    setattr(self._config.grid, field, value)
                    updated_fields[field] = value

            if updated_fields:
                logger.info(f"Grid config updated: {updated_fields}")

    async def is_grid_visible(self) -> bool:
        async with self._state_lock:
            return self._visible

    def get_current_config(self):
        return self._config.grid

    async def shutdown(self) -> None:
        logger.info("Shutting down GridService")
        self.subscription_manager.unsubscribe_all()
