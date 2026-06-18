from __future__ import annotations

import logging
import math

from vocalance.app.config.app_config import GlobalAppConfig, GridConfig
from vocalance.app.config.command_types import BaseCommand, GridSelectCommand, GridShowCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import GridCommandParsedEvent
from vocalance.app.events.grid_events import GridStateEvent
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class GridService(Service):
    """Handle grid voice commands and merge overlay-driven config updates into ``GlobalAppConfig``."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus)
        self._config = config
        self._visible: bool = False
        self._current_click_mode: str = "click"
        self.subscribe(GridCommandParsedEvent, self._handle_grid_command)

    def _calculate_grid_dimensions(self, num_rects: int) -> tuple[int, int]:
        cols = math.ceil(math.sqrt(num_rects))
        rows = math.ceil(num_rects / cols)
        return rows, cols

    async def _handle_grid_command(self, event: GridCommandParsedEvent) -> None:
        command: BaseCommand = event.command
        if isinstance(command, GridShowCommand):
            num_rects = command.num_rects or self._config.grid.default_rect_count
            rows, cols = self._calculate_grid_dimensions(num_rects)
            self._current_click_mode = command.click_mode
            self._visible = True
            await self.event_bus.publish(
                GridStateEvent(state="visible", config={"rows": rows, "cols": cols, "click_mode": command.click_mode})
            )
        elif isinstance(command, GridSelectCommand):
            if not self._visible:
                return
            await self.event_bus.publish(
                GridStateEvent(
                    state="interaction_request",
                    config={"cell_label": str(command.selected_number), "click_mode": self._current_click_mode},
                )
            )
        else:
            logger.warning("Unknown grid command type: %s", type(command).__name__)

    def is_grid_visible(self) -> bool:
        return self._visible

    def get_current_config(self) -> GridConfig:
        return self._config.grid
