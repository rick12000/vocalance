import asyncio

import pytest

from vocalance.app.config.command_types import PauseCommand, ResumeCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import SystemControlCommandParsedEvent
from vocalance.app.services.command_flow.pause_state_manager import PauseStateManager


@pytest.mark.asyncio
async def test_pause_state_manager_initial_state():
    """Test that pause state manager starts in unpaused state."""
    event_bus = EventBus()
    manager = PauseStateManager(event_bus=event_bus)

    assert manager.is_paused() is False


@pytest.mark.asyncio
async def test_pause_state_manager_pause():
    """Test pausing the application."""
    event_bus = EventBus()
    manager = PauseStateManager(event_bus=event_bus)
    event_bus.start(asyncio.get_running_loop())

    assert manager.is_paused() is False

    # Send pause command
    await event_bus.publish(SystemControlCommandParsedEvent(command=PauseCommand(), source="test"))
    await asyncio.sleep(0.1)

    # Should be paused now
    assert manager.is_paused() is True


@pytest.mark.asyncio
async def test_pause_state_manager_resume():
    """Test resuming the application."""
    event_bus = EventBus()
    manager = PauseStateManager(event_bus=event_bus)
    event_bus.start(asyncio.get_running_loop())

    # Pause first
    await event_bus.publish(SystemControlCommandParsedEvent(command=PauseCommand(), source="test"))
    await asyncio.sleep(0.1)
    assert manager.is_paused() is True

    # Resume
    await event_bus.publish(SystemControlCommandParsedEvent(command=ResumeCommand(), source="test"))
    await asyncio.sleep(0.1)

    # Should be unpaused now
    assert manager.is_paused() is False


@pytest.mark.asyncio
async def test_pause_resume_toggle():
    """Test multiple pause/resume cycles."""
    event_bus = EventBus()
    manager = PauseStateManager(event_bus=event_bus)
    event_bus.start(asyncio.get_running_loop())

    # Cycle 1
    await event_bus.publish(SystemControlCommandParsedEvent(command=PauseCommand(), source="test"))
    await asyncio.sleep(0.1)
    assert manager.is_paused() is True

    await event_bus.publish(SystemControlCommandParsedEvent(command=ResumeCommand(), source="test"))
    await asyncio.sleep(0.1)
    assert manager.is_paused() is False

    # Cycle 2
    await event_bus.publish(SystemControlCommandParsedEvent(command=PauseCommand(), source="test"))
    await asyncio.sleep(0.1)
    assert manager.is_paused() is True

    await event_bus.publish(SystemControlCommandParsedEvent(command=ResumeCommand(), source="test"))
    await asyncio.sleep(0.1)
    assert manager.is_paused() is False
