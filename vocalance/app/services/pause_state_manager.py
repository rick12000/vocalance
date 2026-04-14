import logging

from vocalance.app.config.command_types import PauseCommand, ResumeCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import SystemControlCommandParsedEvent
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class PauseStateManager(Service):
    """Tracks application pause/resume state."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._is_paused = False
        event_bus.subscribe(SystemControlCommandParsedEvent, self._handle)

    async def _handle(self, event: SystemControlCommandParsedEvent) -> None:
        if isinstance(event.command, PauseCommand):
            self._is_paused = True
            logger.info("Application paused")
        elif isinstance(event.command, ResumeCommand):
            self._is_paused = False
            logger.info("Application resumed")

    def is_paused(self) -> bool:
        return self._is_paused

    async def shutdown(self) -> None:
        self._event_bus.unsubscribe(SystemControlCommandParsedEvent, self._handle)
