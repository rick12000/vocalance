"""
Tests for MarkService - mark creation, execution, and visualization.
"""
import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio

from iris.app.config.command_types import MarkCreateCommand
from iris.app.events.command_events import MarkCommandParsedEvent
from iris.app.events.mark_events import MarkCreatedEventData
from iris.app.services.mark_service import MarkService


@pytest_asyncio.fixture
async def mark_service(event_bus, app_config, mock_storage_service, mock_protected_terms_validator):
    """Create mark service with mocked storage."""
    service = MarkService(
        event_bus=event_bus,
        config=app_config,
        storage=mock_storage_service,
        protected_terms_validator=mock_protected_terms_validator,
    )
    service.setup_subscriptions()

    await event_bus.start_worker()
    yield service
    await event_bus.stop_worker()


@pytest.mark.asyncio
@patch("pyautogui.moveTo")
async def test_mark_create_command(mock_move, mark_service):
    """Test creating a new mark at cursor position."""
    service = mark_service
    event_bus = service._event_bus

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(MarkCreatedEventData, capture_event)

    command = MarkCreateCommand(label="home", x=100.0, y=200.0)
    event = MarkCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    created_events = [e for e in captured_events if isinstance(e, MarkCreatedEventData)]

    assert len(created_events) == 1
    assert created_events[0].name == "home"


@pytest.mark.asyncio
async def test_reserved_label_rejection(mark_service, mock_protected_terms_validator):
    """Test that reserved labels cannot be used for marks."""
    service = mark_service
    event_bus = service._event_bus

    # Make validator reject "show grid"
    mock_protected_terms_validator.validate_term.return_value = (False, "'show grid' is a protected term and cannot be used")

    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    event_bus.subscribe(MarkCreatedEventData, capture_event)

    command = MarkCreateCommand(label="show grid", x=100.0, y=200.0)
    event = MarkCommandParsedEvent(command=command, source="speech")
    await event_bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(captured_events) == 0
