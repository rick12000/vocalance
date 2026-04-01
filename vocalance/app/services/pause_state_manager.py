import asyncio
import logging

from vocalance.app.config.command_types import PauseCommand, ResumeCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import SystemControlCommandParsedEvent

logger = logging.getLogger(__name__)


class PauseStateManager:
    """Manages application pause/resume state.

    Controls whether audio processing and command execution should be active.
    When paused, all command execution is blocked except for the resume command.
    """

    def __init__(self, event_bus: EventBus) -> None:
        """Initialize pause state manager.

        Args:
            event_bus: EventBus for pub/sub messaging.
        """
        self._event_bus = event_bus
        self._is_paused = False
        self._state_lock = asyncio.Lock()
        logger.debug("PauseStateManager initialized")

    def setup_subscriptions(self) -> None:
        """Setup event subscriptions for system control commands."""
        self._event_bus.subscribe(event_type=SystemControlCommandParsedEvent, handler=self._handle_system_control_command)
        logger.info("PauseStateManager subscriptions set up")

    async def _handle_system_control_command(self, event: SystemControlCommandParsedEvent) -> None:
        """Handle system control commands (pause/resume).

        Args:
            event: System control command event.
        """
        command = event.command

        if isinstance(command, PauseCommand):
            await self._set_paused(True)
            logger.info("Application paused - audio processing disabled")
        elif isinstance(command, ResumeCommand):
            await self._set_paused(False)
            logger.info("Application resumed - audio processing enabled")

    async def _set_paused(self, paused: bool) -> None:
        """Set pause state.

        Args:
            paused: True to pause, False to resume.
        """
        async with self._state_lock:
            self._is_paused = paused

    async def is_paused(self) -> bool:
        """Check if application is currently paused.

        Returns:
            True if paused, False otherwise.
        """
        async with self._state_lock:
            return self._is_paused

    def is_paused_sync(self) -> bool:
        """Synchronous check if application is paused.

        Note: This is not thread-safe but useful for quick checks
        in synchronous contexts where blocking is not acceptable.

        Returns:
            True if paused, False otherwise.
        """
        return self._is_paused
