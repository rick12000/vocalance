import logging
import math

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import GridSelectCommand, GridShowCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import GridCommandParsedEvent
from vocalance.app.events.grid_events import GridStateEvent
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class GridService(Service):
    """Grid service for command processing and UI state management."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        self._event_bus = event_bus
        self._config = config
        self._visible: bool = False
        self._current_click_mode: str = "click"
        event_bus.subscribe(GridCommandParsedEvent, self._handle_grid_command)
        event_bus.subscribe(GridStateEvent, self._handle_grid_state_event)

    def _calculate_grid_dimensions(self, num_rects: int) -> tuple[int, int]:
        cols = math.ceil(math.sqrt(num_rects))
        rows = math.ceil(num_rects / cols)
        return rows, cols

    async def _handle_grid_command(self, event: GridCommandParsedEvent) -> None:
        command = event.command
        if isinstance(command, GridShowCommand):
            num_rects = command.num_rects or self._config.grid.default_rect_count
            rows, cols = self._calculate_grid_dimensions(num_rects)
            self._current_click_mode = command.click_mode
            self._visible = True
            await self._event_bus.publish(
                GridStateEvent(state="visible", config={"rows": rows, "cols": cols, "click_mode": command.click_mode})
            )
        elif isinstance(command, GridSelectCommand):
            if not self._visible:
                return
            await self._event_bus.publish(
                GridStateEvent(
                    state="interaction_request",
                    config={"cell_label": str(command.selected_number), "click_mode": self._current_click_mode},
                )
            )
        else:
            logger.warning("Unknown grid command type: %s", type(command).__name__)

    async def _handle_grid_state_event(self, event: GridStateEvent) -> None:
        if event.state == "config_updated" and event.config:
            for field in (
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
            ):
                value = event.config.get(field)
                if value is not None and hasattr(self._config.grid, field):
                    if field == "cancel_phrases" and isinstance(value, list):
                        value = list(set(value))
                    setattr(self._config.grid, field, value)

    def is_grid_visible(self) -> bool:
        return self._visible

    def get_current_config(self):
        return self._config.grid

    async def shutdown(self) -> None:
        self._event_bus.unsubscribe(GridCommandParsedEvent, self._handle_grid_command)
        self._event_bus.unsubscribe(GridStateEvent, self._handle_grid_state_event)
