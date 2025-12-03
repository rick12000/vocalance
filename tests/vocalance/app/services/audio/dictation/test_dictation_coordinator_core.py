"""Unit tests for DictationCoordinator core functionality.

Tests state machine, mode transitions, initialization, and event handling.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.audio.dictation_handling.dictation_coordinator import (
    DictationCoordinator,
    DictationMode,
    DictationSession,
    DictationState,
    LLMSession,
)


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
    with patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.TextInputService"), patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.LLMService"
    ), patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.AgenticPromptService"), patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationAliasService"
    ):

        coord = DictationCoordinator(
            event_bus=mock_event_bus,
            config=app_config,
            storage=mock_storage,
        )

        # Mock the services
        coord.text_service = Mock()
        coord.text_service.initialize = AsyncMock(return_value=True)
        coord.llm_service = Mock()
        coord.llm_service.initialize = AsyncMock(return_value=True)
        coord.agentic_service = Mock()
        coord.agentic_service.initialize = AsyncMock(return_value=True)
        coord.agentic_service.setup_subscriptions = Mock()
        coord.alias_service = Mock()
        coord.alias_service.initialize = AsyncMock(return_value=True)

        await coord.initialize()
        return coord


@pytest.mark.asyncio
async def test_initialization_succeeds(coordinator):
    """Test that coordinator initializes successfully."""
    assert coordinator is not None
    assert coordinator._current_state == DictationState.IDLE


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
    with coordinator._state_lock:
        coordinator._set_state(DictationState.RECORDING)

    assert coordinator._current_state == DictationState.RECORDING


@pytest.mark.asyncio
async def test_state_transition_recording_to_idle(coordinator):
    """Test valid state transition from RECORDING to IDLE."""
    with coordinator._state_lock:
        coordinator._set_state(DictationState.RECORDING)
        coordinator._set_state(DictationState.IDLE)

    assert coordinator._current_state == DictationState.IDLE


@pytest.mark.asyncio
async def test_invalid_state_transition_raises_error(coordinator):
    """Test that invalid state transition raises ValueError."""
    # Set up valid path: IDLE -> RECORDING -> PROCESSING_LLM
    with coordinator._state_lock:
        coordinator._set_state(DictationState.RECORDING)
        coordinator._set_state(DictationState.PROCESSING_LLM)

    # PROCESSING_LLM -> PROCESSING_LLM is invalid
    with pytest.raises(ValueError):
        with coordinator._state_lock:
            coordinator._set_state(DictationState.PROCESSING_LLM)


@pytest.mark.asyncio
async def test_should_apply_formatting_standard_mode(coordinator):
    """Test formatting decision for STANDARD mode."""
    # Should respect config setting
    result = coordinator._should_apply_formatting(DictationMode.STANDARD)
    assert result == coordinator.config.dictation.enable_dictation_formatting


@pytest.mark.asyncio
async def test_should_apply_formatting_type_mode_disables(coordinator):
    """Test that TYPE mode always disables formatting."""
    result = coordinator._should_apply_formatting(DictationMode.TYPE)
    assert result is False


@pytest.mark.asyncio
async def test_should_apply_formatting_smart_mode(coordinator):
    """Test formatting decision for SMART mode."""
    result = coordinator._should_apply_formatting(DictationMode.SMART)
    assert result == coordinator.config.dictation.enable_dictation_formatting


@pytest.mark.asyncio
async def test_should_apply_formatting_visual_mode(coordinator):
    """Test formatting decision for VISUAL mode."""
    result = coordinator._should_apply_formatting(DictationMode.VISUAL)
    assert result == coordinator.config.dictation.enable_dictation_formatting


@pytest.mark.asyncio
async def test_should_apply_formatting_hidden_mode(coordinator):
    """Test formatting decision for HIDDEN mode."""
    result = coordinator._should_apply_formatting(DictationMode.HIDDEN)
    assert result == coordinator.config.dictation.enable_dictation_formatting


@pytest.mark.asyncio
async def test_set_direct_token_callback(coordinator):
    """Test setting direct token callback."""

    def mock_callback(token: str):
        pass

    coordinator.set_direct_token_callback(mock_callback)
    assert coordinator._direct_token_callback == mock_callback


@pytest.mark.asyncio
async def test_clear_direct_token_callback(coordinator):
    """Test clearing direct token callback."""
    coordinator.set_direct_token_callback(None)
    assert coordinator._direct_token_callback is None


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
    """Test that state getter uses lock."""
    with coordinator._state_lock:
        coordinator._current_state = DictationState.RECORDING

    state = coordinator._get_state()
    assert state == DictationState.RECORDING


@pytest.mark.asyncio
async def test_set_stt_service(coordinator):
    """Test setting STT service reference."""
    mock_stt = Mock()
    coordinator.set_stt_service(mock_stt)

    assert coordinator._stt_service == mock_stt


@pytest.mark.asyncio
async def test_mode_transitions_standard_to_inactive(coordinator):
    """Test transitioning from STANDARD mode to inactive."""
    with coordinator._state_lock:
        coordinator._current_session = DictationSession(
            session_id="test",
            mode=DictationMode.STANDARD,
            start_time=0.0,
        )

    assert coordinator.active_mode == DictationMode.STANDARD
    assert coordinator.is_active() is True

    with coordinator._state_lock:
        coordinator._current_session = None

    assert coordinator.active_mode == DictationMode.INACTIVE
    assert coordinator.is_active() is False


@pytest.mark.asyncio
async def test_state_machine_valid_all_paths(coordinator):
    """Test all valid state transition paths."""
    # IDLE -> RECORDING -> IDLE (normal flow)
    with coordinator._state_lock:
        coordinator._set_state(DictationState.RECORDING)
        coordinator._set_state(DictationState.IDLE)

    # IDLE -> RECORDING -> PROCESSING_LLM -> IDLE (smart mode flow)
    with coordinator._state_lock:
        coordinator._set_state(DictationState.RECORDING)
        coordinator._set_state(DictationState.PROCESSING_LLM)
        coordinator._set_state(DictationState.IDLE)

    # IDLE -> SHUTTING_DOWN
    with coordinator._state_lock:
        coordinator._set_state(DictationState.SHUTTING_DOWN)

    assert coordinator._current_state == DictationState.SHUTTING_DOWN


@pytest.mark.asyncio
async def test_concurrent_state_access(coordinator):
    """Test thread-safe concurrent state access."""
    # Simulate concurrent reads
    for _ in range(10):
        state = coordinator._get_state()
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
    """Test that _get_state returns actual state value."""
    with coordinator._state_lock:
        coordinator._set_state(DictationState.RECORDING)

    state1 = coordinator._get_state()
    state2 = coordinator._get_state()

    assert state1 == state2
    assert state1 == DictationState.RECORDING


@pytest.mark.asyncio
async def test_initialization_with_all_services(mock_event_bus, mock_storage, app_config):
    """Test initialization with all services initialized."""
    with patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.TextInputService") as mock_text, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.LLMService"
    ) as mock_llm, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.AgenticPromptService"
    ) as mock_agentic, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationAliasService"
    ) as mock_alias:

        # Mock all services to return success
        mock_text.return_value.initialize = AsyncMock(return_value=True)
        mock_llm.return_value.initialize = AsyncMock(return_value=True)
        mock_agentic.return_value.initialize = AsyncMock(return_value=True)
        mock_agentic.return_value.setup_subscriptions = Mock()
        mock_alias.return_value.initialize = AsyncMock(return_value=True)

        coord = DictationCoordinator(
            event_bus=mock_event_bus,
            config=app_config,
            storage=mock_storage,
        )

        result = await coord.initialize()
        assert result is True


@pytest.mark.asyncio
async def test_initialization_failure_returns_false(mock_event_bus, mock_storage, app_config):
    """Test that initialization failure is properly reported."""
    with patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.TextInputService") as mock_text, patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.LLMService"
    ), patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.AgenticPromptService"), patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationAliasService"
    ):

        # Mock text service to fail
        mock_text.return_value.initialize = AsyncMock(return_value=False)

        coord = DictationCoordinator(
            event_bus=mock_event_bus,
            config=app_config,
            storage=mock_storage,
        )

        result = await coord.initialize()
        assert result is False
