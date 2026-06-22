import pytest

from vocalance.app.config.command_types import PauseCommand, ResumeCommand
from vocalance.app.events.command_events import SystemControlCommandParsedEvent


@pytest.mark.parametrize("command, expected_paused", [(PauseCommand(), True), (ResumeCommand(), False)])
@pytest.mark.asyncio
async def test_command_sets_pause_state(pause_state_manager, command, expected_paused):
    event = SystemControlCommandParsedEvent(command=command, source="test")
    await pause_state_manager.handle_system_control_command(event)
    assert pause_state_manager.is_paused() is expected_paused


@pytest.mark.asyncio
async def test_resume_lifts_active_pause(pause_state_manager):
    await pause_state_manager.handle_system_control_command(SystemControlCommandParsedEvent(command=PauseCommand(), source="test"))
    assert pause_state_manager.is_paused() is True

    await pause_state_manager.handle_system_control_command(
        SystemControlCommandParsedEvent(command=ResumeCommand(), source="test")
    )
    assert pause_state_manager.is_paused() is False
