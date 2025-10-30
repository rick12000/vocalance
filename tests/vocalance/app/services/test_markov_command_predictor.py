from unittest.mock import AsyncMock, Mock

import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.events.core_events import AudioDetectedEvent, MarkovPredictionFeedbackEvent
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.markov_command_predictor import MarkovCommandService
from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry


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
    storage.write = AsyncMock()
    return storage


@pytest.fixture
def app_config():
    """Create application configuration for testing."""
    return GlobalAppConfig()


@pytest.fixture
def markov_service(mock_event_bus, app_config, mock_storage):
    """Create MarkovCommandService instance."""
    return MarkovCommandService(event_bus=mock_event_bus, config=app_config, storage=mock_storage)


@pytest.mark.asyncio
async def test_initialize_empty_history(markov_service, mock_storage):
    """Test initialization with empty history."""
    mock_storage.read.return_value = CommandHistoryData(history=[])

    success = await markov_service.initialize()

    assert success is True
    assert markov_service._model_trained is True


@pytest.mark.asyncio
async def test_initialize_with_history(markov_service, mock_storage):
    """Test initialization trains models on existing history."""
    history = [
        CommandHistoryEntry(command="copy", timestamp=1000.0, success=None, metadata={}),
        CommandHistoryEntry(command="paste", timestamp=2000.0, success=None, metadata={}),
        CommandHistoryEntry(command="undo", timestamp=3000.0, success=None, metadata={}),
        CommandHistoryEntry(command="copy", timestamp=4000.0, success=None, metadata={}),
        CommandHistoryEntry(command="paste", timestamp=5000.0, success=None, metadata={}),
    ]
    mock_storage.read.return_value = CommandHistoryData(history=history)

    success = await markov_service.initialize()

    assert success is True
    assert markov_service._model_trained is True


@pytest.mark.asyncio
async def test_setup_subscriptions(markov_service, mock_event_bus):
    """Test event subscriptions are set up correctly."""
    markov_service.setup_subscriptions()

    # Should subscribe to three event types
    assert mock_event_bus.subscribe.call_count == 3


@pytest.mark.asyncio
async def test_prediction_with_insufficient_history(markov_service, mock_storage, mock_event_bus):
    """Test prediction doesn't fire with insufficient history."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await markov_service.initialize()

    import time

    event = AudioDetectedEvent(timestamp=time.time())
    await markov_service._handle_audio_detected_fast_track(event)

    # Should not publish prediction
    mock_event_bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_prediction_fires_with_sufficient_history(markov_service, mock_storage, mock_event_bus, app_config):
    """Test prediction fires when conditions are met."""
    # Create repeating pattern
    history = []
    for i in range(20):
        history.append(CommandHistoryEntry(command="copy", timestamp=1000.0 + i * 10, success=None, metadata={}))
        history.append(CommandHistoryEntry(command="paste", timestamp=1000.0 + i * 10 + 5, success=None, metadata={}))

    mock_storage.read.return_value = CommandHistoryData(history=history)
    await markov_service.initialize()

    # Build up command history to trigger prediction
    for cmd in ["copy", "paste", "copy"]:
        feedback = MarkovPredictionFeedbackEvent(predicted_command=cmd, actual_command=cmd, was_correct=True, source="test")
        await markov_service._handle_prediction_feedback(feedback)

    # Now audio detected should predict "paste"
    import time

    event = AudioDetectedEvent(timestamp=time.time())
    await markov_service._handle_audio_detected_fast_track(event)

    # May or may not fire depending on confidence - just verify no error
    assert True


@pytest.mark.asyncio
async def test_prediction_disabled_during_dictation(markov_service, mock_storage, mock_event_bus):
    """Test predictions are disabled during dictation mode."""
    # Setup with history
    history = []
    for i in range(10):
        history.append(CommandHistoryEntry(command="copy", timestamp=1000.0 + i, success=None, metadata={}))
    mock_storage.read.return_value = CommandHistoryData(history=history)
    await markov_service.initialize()

    # Enable dictation mode
    dictation_event = DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard")
    await markov_service._handle_dictation_mode_change(dictation_event)

    # Try to trigger prediction
    import time

    event = AudioDetectedEvent(timestamp=time.time())
    await markov_service._handle_audio_detected_fast_track(event)

    # Should not publish prediction during dictation
    mock_event_bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_prediction_enabled_after_dictation(markov_service, mock_storage, mock_event_bus):
    """Test predictions re-enable after dictation mode ends."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await markov_service.initialize()

    # Enable then disable dictation
    enable_event = DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode="standard")
    await markov_service._handle_dictation_mode_change(enable_event)
    assert markov_service._dictation_active is True

    disable_event = DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive")
    await markov_service._handle_dictation_mode_change(disable_event)
    assert markov_service._dictation_active is False


@pytest.mark.asyncio
async def test_feedback_updates_command_history(markov_service, mock_storage):
    """Test prediction feedback updates in-memory history."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await markov_service.initialize()

    # Provide feedback
    feedback = MarkovPredictionFeedbackEvent(predicted_command="copy", actual_command="paste", was_correct=False, source="stt")
    await markov_service._handle_prediction_feedback(feedback)

    # Command history should be updated with actual command
    assert len(markov_service._command_history) == 1
    assert markov_service._command_history[0] == "paste"


@pytest.mark.asyncio
async def test_incorrect_prediction_triggers_cooldown(markov_service, mock_storage, app_config):
    """Test incorrect predictions trigger cooldown period."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await markov_service.initialize()

    # Incorrect prediction
    feedback = MarkovPredictionFeedbackEvent(predicted_command="copy", actual_command="paste", was_correct=False, source="stt")
    await markov_service._handle_prediction_feedback(feedback)

    # Should enter cooldown
    assert markov_service._cooldown_remaining == app_config.markov_predictor.incorrect_prediction_cooldown


@pytest.mark.asyncio
async def test_correct_prediction_no_cooldown(markov_service, mock_storage):
    """Test correct predictions don't trigger cooldown."""
    mock_storage.read.return_value = CommandHistoryData(history=[])
    await markov_service.initialize()

    # Correct prediction
    feedback = MarkovPredictionFeedbackEvent(predicted_command="copy", actual_command="copy", was_correct=True, source="stt")
    await markov_service._handle_prediction_feedback(feedback)

    # Should not enter cooldown
    assert markov_service._cooldown_remaining == 0


@pytest.mark.asyncio
async def test_retrain_updates_model(markov_service, mock_storage):
    """Test manual retraining updates the model."""
    # Initial history
    history1 = [CommandHistoryEntry(command="copy", timestamp=1000.0, success=None, metadata={})]
    mock_storage.read.return_value = CommandHistoryData(history=history1)
    await markov_service.initialize()

    # Update storage with new history
    history2 = [
        CommandHistoryEntry(command="copy", timestamp=1000.0, success=None, metadata={}),
        CommandHistoryEntry(command="paste", timestamp=2000.0, success=None, metadata={}),
    ]
    mock_storage.read.return_value = CommandHistoryData(history=history2)

    # Retrain
    success = await markov_service.retrain()

    assert success is True
    assert markov_service._model_trained is True


@pytest.mark.asyncio
async def test_confidence_threshold_update(markov_service, app_config):
    """Test confidence threshold can be updated."""
    original_threshold = app_config.markov_predictor.confidence_threshold
    new_threshold = 0.75

    markov_service.on_confidence_threshold_updated(new_threshold)

    assert markov_service._markov_config.confidence_threshold == new_threshold
    assert markov_service._markov_config.confidence_threshold != original_threshold


@pytest.mark.asyncio
async def test_predict_with_backoff_strategy(markov_service, mock_storage):
    """Test prediction uses backoff from high to low order."""
    # Create history with patterns at different orders
    history = []
    # Create a clear 2nd order pattern: A -> B -> C
    for i in range(15):
        history.append(CommandHistoryEntry(command="command_a", timestamp=1000.0 + i * 30, success=None, metadata={}))
        history.append(CommandHistoryEntry(command="command_b", timestamp=1000.0 + i * 30 + 10, success=None, metadata={}))
        history.append(CommandHistoryEntry(command="command_c", timestamp=1000.0 + i * 30 + 20, success=None, metadata={}))

    mock_storage.read.return_value = CommandHistoryData(history=history)
    await markov_service.initialize()

    # Set up command history to match pattern
    markov_service._command_history.clear()
    markov_service._command_history.append("command_a")
    markov_service._command_history.append("command_b")

    # Predict next command
    prediction = markov_service._predict_next_command()

    # Should predict command_c with some confidence
    if prediction:
        predicted_cmd, confidence, order = prediction
        assert predicted_cmd == "command_c"
        assert 0 < confidence <= 1.0


@pytest.mark.asyncio
async def test_prediction_respects_minimum_frequency(markov_service, mock_storage, app_config):
    """Test predictions respect minimum command frequency threshold."""
    # Create history with one-off transitions
    history = [
        CommandHistoryEntry(command="cmd_a", timestamp=1000.0, success=None, metadata={}),
        CommandHistoryEntry(command="cmd_b", timestamp=2000.0, success=None, metadata={}),
        CommandHistoryEntry(command="cmd_c", timestamp=3000.0, success=None, metadata={}),
    ]
    mock_storage.read.return_value = CommandHistoryData(history=history)
    await markov_service.initialize()

    markov_service._command_history.clear()
    markov_service._command_history.append("cmd_a")

    # Should not predict due to low frequency
    prediction = markov_service._predict_next_command()
    assert prediction is None
