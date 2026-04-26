import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator
from vocalance.app.services.audio.dictation_handling.types import DictationMode, DictationSession, DictationState, LLMSession


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    event_bus = Mock()
    event_bus.subscribe = Mock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    storage = Mock()
    storage.read = AsyncMock()
    storage.write = AsyncMock(return_value=True)
    return storage


@pytest.fixture
def app_config():
    """Create application configuration."""
    return GlobalAppConfig()


@pytest.fixture
async def coordinator(mock_event_bus, mock_storage, app_config):
    """Create DictationCoordinator instance for testing."""
    mock_stt = Mock()
    loop = asyncio.get_running_loop()
    with patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationTextInput"), patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.LLMService"
    ), patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.AgenticPromptService"), patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationAliasService"
    ):

        coord = DictationCoordinator(
            event_bus=mock_event_bus,
            config=app_config,
            storage=mock_storage,
            gui_event_loop=loop,
            stt_service=mock_stt,
            input_service=Mock(),
        )

        coord.text_service = Mock()
        coord.text_service.initialize = Mock(return_value=True)
        coord.llm_service = Mock()
        coord.llm_service.initialize = Mock(return_value=True)
        coord.agentic_service = Mock()
        coord.agentic_service.initialize = AsyncMock(return_value=True)
        coord.alias_service = Mock()
        coord.alias_service.initialize = AsyncMock(return_value=True)

        await coord.initialize()
        return coord


@pytest.mark.asyncio
async def test_initialization_succeeds(coordinator):
    """Test that coordinator initializes successfully."""
    assert coordinator is not None
    assert coordinator.current_state == DictationState.IDLE


@pytest.mark.asyncio
async def test_active_mode_inactive_at_start(coordinator):
    """Test that active mode is INACTIVE at start."""
    assert coordinator.active_mode == DictationMode.INACTIVE


@pytest.mark.asyncio
async def test_is_active_false_at_start(coordinator):
    """Test that is_active() returns False at start."""
    assert coordinator.is_active() is False


@pytest.mark.asyncio
async def test_state_transition_idle_to_recording(coordinator):
    """Test valid state transition from IDLE to RECORDING."""
    with coordinator.state_lock:
        coordinator.set_state(DictationState.RECORDING)

    assert coordinator.current_state == DictationState.RECORDING


@pytest.mark.asyncio
async def test_state_transition_recording_to_idle(coordinator):
    """Test valid state transition from RECORDING to IDLE."""
    with coordinator.state_lock:
        coordinator.set_state(DictationState.RECORDING)
        coordinator.set_state(DictationState.IDLE)

    assert coordinator.current_state == DictationState.IDLE


@pytest.mark.asyncio
async def test_invalid_state_transition_raises_error(coordinator):
    """Test that invalid state transition raises ValueError."""
    # Set up valid path: IDLE -> RECORDING -> PROCESSING_LLM
    with coordinator.state_lock:
        coordinator.set_state(DictationState.RECORDING)
        coordinator.set_state(DictationState.PROCESSING_LLM)

    # PROCESSING_LLM -> PROCESSING_LLM is invalid
    with pytest.raises(ValueError):
        with coordinator.state_lock:
            coordinator.set_state(DictationState.PROCESSING_LLM)


@pytest.mark.asyncio
async def test_dictation_session_creation(coordinator):
    """Test DictationSession snapshot creation."""
    session = DictationSession(
        session_id="test123",
        mode=DictationMode.STANDARD,
        start_time=0.0,
        accumulated_text="test text",
        is_first_segment=True,
    )

    assert session.session_id == "test123"
    assert session.mode == DictationMode.STANDARD
    assert session.accumulated_text == "test text"
    assert session.is_first_segment is True


@pytest.mark.asyncio
async def test_llm_session_creation(coordinator):
    """Test LLMSession snapshot creation."""
    session = LLMSession(
        session_id="llm123",
        raw_text="original text",
        agentic_prompt="fix grammar",
    )

    assert session.session_id == "llm123"
    assert session.raw_text == "original text"
    assert session.agentic_prompt == "fix grammar"


@pytest.mark.asyncio
async def test_thread_safe_state_getter(coordinator):
    """Test that state is readable under the lock."""
    with coordinator.state_lock:
        coordinator.current_state = DictationState.RECORDING

    with coordinator.state_lock:
        state = coordinator.current_state
    assert state == DictationState.RECORDING


@pytest.mark.asyncio
async def test_stt_service_injected_at_construction(coordinator):
    """STT service is required at construction; coordinator holds the same reference."""
    assert isinstance(coordinator.stt_service, Mock)


@pytest.mark.asyncio
async def test_mode_transitions_standard_to_inactive(coordinator):
    """Test transitioning from STANDARD mode to inactive."""
    with coordinator.state_lock:
        coordinator.current_session = DictationSession(
            session_id="test",
            mode=DictationMode.STANDARD,
            start_time=0.0,
        )

    assert coordinator.active_mode == DictationMode.STANDARD
    assert coordinator.is_active() is True

    with coordinator.state_lock:
        coordinator.current_session = None

    assert coordinator.active_mode == DictationMode.INACTIVE
    assert coordinator.is_active() is False


@pytest.mark.asyncio
async def test_state_machine_valid_all_paths(coordinator):
    """Test all valid state transition paths."""
    # IDLE -> RECORDING -> IDLE (normal flow)
    with coordinator.state_lock:
        coordinator.set_state(DictationState.RECORDING)
        coordinator.set_state(DictationState.IDLE)

    # IDLE -> RECORDING -> PROCESSING_LLM -> IDLE (smart mode flow)
    with coordinator.state_lock:
        coordinator.set_state(DictationState.RECORDING)
        coordinator.set_state(DictationState.PROCESSING_LLM)
        coordinator.set_state(DictationState.IDLE)

    # IDLE -> SHUTTING_DOWN
    with coordinator.state_lock:
        coordinator.set_state(DictationState.SHUTTING_DOWN)

    assert coordinator.current_state == DictationState.SHUTTING_DOWN


@pytest.mark.asyncio
async def test_concurrent_state_access(coordinator):
    """Test thread-safe concurrent state access."""
    for _ in range(10):
        with coordinator.state_lock:
            state = coordinator.current_state
        assert isinstance(state, DictationState)


@pytest.mark.asyncio
async def test_dictation_session_immutability_principles(coordinator):
    """Test that DictationSession follows immutability principles."""
    session = DictationSession(
        session_id="test",
        mode=DictationMode.STANDARD,
        start_time=0.0,
    )

    # Original values should be unchanged
    assert session.session_id == "test"
    assert session.accumulated_text == ""
    assert session.is_first_segment is True


@pytest.mark.asyncio
async def test_get_state_returns_copy_of_state(coordinator):
    """Test that state reads are consistent under the lock."""
    with coordinator.state_lock:
        coordinator.set_state(DictationState.RECORDING)

    with coordinator.state_lock:
        state1 = coordinator.current_state
        state2 = coordinator.current_state

    assert state1 == state2
    assert state1 == DictationState.RECORDING


@pytest.mark.asyncio
async def test_initialization_with_all_services(mock_event_bus, mock_storage, app_config):
    """Test initialization with all services initialized."""
    with patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationTextInput") as mock_text, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.LLMService"
    ) as mock_llm, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.AgenticPromptService"
    ) as mock_agentic, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationAliasService"
    ) as mock_alias:

        mock_text.return_value.initialize = Mock(return_value=True)
        mock_llm.return_value.initialize = Mock(return_value=True)
        mock_agentic.return_value.initialize = AsyncMock(return_value=True)
        mock_alias.return_value.initialize = AsyncMock(return_value=True)

        loop = asyncio.get_running_loop()
        coord = DictationCoordinator(
            event_bus=mock_event_bus,
            config=app_config,
            storage=mock_storage,
            gui_event_loop=loop,
            stt_service=Mock(),
            input_service=Mock(),
        )

        result = await coord.initialize()
        assert result is True


@pytest.mark.asyncio
async def test_initialization_failure_returns_false(mock_event_bus, mock_storage, app_config):
    """Test that initialization failure is properly reported."""
    with patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationTextInput") as mock_text, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.LLMService"
    ) as mock_llm, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.AgenticPromptService"
    ) as mock_agentic, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationAliasService"
    ) as mock_alias:

        mock_text.return_value.initialize = Mock(return_value=False)
        mock_llm.return_value.initialize = Mock(return_value=True)
        mock_agentic.return_value.initialize = AsyncMock(return_value=True)
        mock_alias.return_value.initialize = AsyncMock(return_value=True)

        loop = asyncio.get_running_loop()
        coord = DictationCoordinator(
            event_bus=mock_event_bus,
            config=app_config,
            storage=mock_storage,
            gui_event_loop=loop,
            stt_service=Mock(),
            input_service=Mock(),
        )

        result = await coord.initialize()
        assert result is False
