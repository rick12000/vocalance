import logging

from vocalance.app.config.command_types import PauseCommand, ResumeCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import SystemControlCommandParsedEvent
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class PauseStateManager(Service):
    """Tracks whether the application is paused or resumed."""

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus)
        self.pause_active: bool = False
        self.subscribe(SystemControlCommandParsedEvent, self.handle_system_control_command)

    async def handle_system_control_command(self, event: SystemControlCommandParsedEvent) -> None:
        if isinstance(event.command, PauseCommand):
            self.pause_active = True
            logger.debug("Application paused")
        elif isinstance(event.command, ResumeCommand):
            self.pause_active = False
            logger.debug("Application resumed")

    def is_paused(self) -> bool:
        return self.pause_active
