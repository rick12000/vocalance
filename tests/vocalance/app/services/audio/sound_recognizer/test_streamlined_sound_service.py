from unittest.mock import Mock, patch

import numpy as np
import pytest

from vocalance.app.events.core_events import CustomSoundRecognizedEvent
from vocalance.app.services.audio.sound_recognizer.streamlined_sound_service import SoundService


@pytest.fixture
def sound_service(mock_event_bus, mock_config, mock_storage_factory, mock_recognizer):
    """Create a sound service instance with mocked dependencies."""
    # Mock asset paths
    mock_config.asset_paths = Mock()
    mock_config.asset_paths.yamnet_model_path = "/fake/yamnet/path"

    with patch(
        "vocalance.app.services.audio.sound_recognizer.streamlined_sound_service.SoundRecognizer",
        return_value=mock_recognizer,
    ):
        service = SoundService(mock_event_bus, mock_config, mock_storage_factory)
        return service


@pytest.mark.asyncio
async def test_service_initialization_success(sound_service, mock_recognizer):
    """Test successful service initialization."""
    mock_recognizer.initialize.return_value = True

    result = await sound_service.initialize()

    assert result is True
    assert sound_service.is_initialized is True
    mock_recognizer.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_service_initialization_failure(sound_service, mock_recognizer):
    """Test failed service initialization."""
    mock_recognizer.initialize.return_value = False

    result = await sound_service.initialize()

    assert result is False
    assert sound_service.is_initialized is False


def test_audio_chunk_preprocessing(sound_service):
    """Test that audio chunks are properly preprocessed."""
    # Create mock audio bytes (int16 format)
    audio_int16 = np.array([1000, -2000, 3000, -4000], dtype=np.int16)
    audio_bytes = audio_int16.tobytes()

    result = sound_service._preprocess_audio_chunk(audio_bytes)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == len(audio_int16)
    # Should be normalized to [-1, 1] range
    assert np.max(np.abs(result)) <= 1.0


@pytest.mark.asyncio
async def test_audio_chunk_recognition(sound_service, mock_recognizer, mock_event_bus):
    """Test that audio chunks trigger sound recognition."""
    sound_service.is_initialized = True
    sound_service._training_active = False

    # Mock recognition result
    mock_recognizer.recognize_sound.return_value = ("test_sound", 0.8)
    mock_recognizer.get_mapping.return_value = "test_command"

    # Create mock event data
    event_data = Mock()
    event_data.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
    event_data.sample_rate = 16000

    await sound_service._process_audio_chunk(event_data)

    # Should have called recognize_sound
    mock_recognizer.recognize_sound.assert_called_once()
    # Should have published recognition event
    mock_event_bus.publish.assert_called_once()

    # Verify the published event
    published_event = mock_event_bus.publish.call_args[0][0]
    assert isinstance(published_event, CustomSoundRecognizedEvent)
    assert published_event.label == "test_sound"
    assert published_event.confidence == 0.8
    assert published_event.mapped_command == "test_command"


@pytest.mark.asyncio
async def test_audio_chunk_no_recognition(sound_service, mock_recognizer, mock_event_bus):
    """Test that no event is published when no sound is recognized."""
    sound_service.is_initialized = True
    sound_service._training_active = False

    # Mock no recognition
    mock_recognizer.recognize_sound.return_value = None

    event_data = Mock()
    event_data.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
    event_data.sample_rate = 16000

    await sound_service._process_audio_chunk(event_data)

    # Should not publish any event
    mock_event_bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_esc50_sounds_ignored(sound_service, mock_recognizer, mock_event_bus):
    """Test that ESC-50 background sounds are ignored."""
    sound_service.is_initialized = True
    sound_service._training_active = False

    # Mock ESC-50 recognition (should be ignored)
    mock_recognizer.recognize_sound.return_value = ("esc50_breathing", 0.7)

    event_data = Mock()
    event_data.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
    event_data.sample_rate = 16000

    await sound_service._process_audio_chunk(event_data)

    # Should not publish any event for ESC-50 sounds
    mock_event_bus.publish.assert_not_called()
