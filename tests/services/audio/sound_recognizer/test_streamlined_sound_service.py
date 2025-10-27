"""
Unit tests for StreamlinedSoundService.

Tests the service layer functionality in isolation using mocks.
"""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vocalance.app.events.core_events import CustomSoundRecognizedEvent, ProcessAudioChunkForSoundRecognitionEvent
from vocalance.app.services.audio.sound_recognizer.streamlined_sound_service import StreamlinedSoundService


class TestStreamlinedSoundService:
    """Test the StreamlinedSoundService class."""

    @pytest.fixture
    def service(self, mock_event_bus, mock_config, mock_storage_factory, mock_recognizer):
        """Create a service instance with mocked dependencies."""
        with patch(
            "vocalance.app.services.audio.sound_recognizer.streamlined_sound_service.StreamlinedSoundRecognizer",
            return_value=mock_recognizer,
        ):
            service = StreamlinedSoundService(mock_event_bus, mock_config, mock_storage_factory)
            return service

    def test_init(self, mock_event_bus, mock_config, mock_storage_factory):
        """Test service initialization."""
        with patch(
            "vocalance.app.services.audio.sound_recognizer.streamlined_sound_service.StreamlinedSoundRecognizer"
        ) as mock_recognizer_class:
            service = StreamlinedSoundService(mock_event_bus, mock_config, mock_storage_factory)

            assert service.event_bus == mock_event_bus
            assert service.config == mock_config
            assert service.is_initialized is False
            assert service._training_active is False
            assert service._current_training_label is None
            assert service._training_samples == []

            # Should subscribe to 7 events total
            assert mock_event_bus.subscribe.call_count == 7

            # Check that it subscribes to the main audio chunk event
            mock_event_bus.subscribe.assert_any_call(ProcessAudioChunkForSoundRecognitionEvent, service._handle_audio_chunk)

            # Should create recognizer
            mock_recognizer_class.assert_called_once_with(mock_config, mock_storage_factory)

    @pytest.mark.asyncio
    async def test_initialize_success(self, service, mock_recognizer):
        """Test successful service initialization."""
        mock_recognizer.initialize.return_value = True

        result = await service.initialize()

        assert result is True
        assert service.is_initialized is True
        mock_recognizer.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, service, mock_recognizer):
        """Test failed service initialization."""
        mock_recognizer.initialize.return_value = False

        result = await service.initialize()

        assert result is False
        assert service.is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize_exception(self, service, mock_recognizer):
        """Test initialization with exception."""
        mock_recognizer.initialize.side_effect = Exception("Test error")

        result = await service.initialize()

        assert result is False
        assert service.is_initialized is False

    def test_preprocess_audio_chunk(self, service):
        """Test audio chunk preprocessing."""
        # Create mock audio bytes (int16 format)
        audio_int16 = np.array([1000, -2000, 3000, -4000], dtype=np.int16)
        audio_bytes = audio_int16.tobytes()

        result = service._preprocess_audio_chunk(audio_bytes)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == len(audio_int16)
        # Should be normalized to [-1, 1] range
        assert np.max(np.abs(result)) <= 1.0
        # Check specific conversion
        expected = audio_int16.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_not_initialized(self, service, mock_recognizer):
        """Test handling audio chunk when service is not initialized."""
        service.is_initialized = False

        event_data = Mock()
        event_data.audio_chunk = b"\x00\x01\x02\x03"
        event_data.sample_rate = 16000

        # Should return early without processing
        await service._handle_audio_chunk(event_data)

        mock_recognizer.recognize_sound.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_recognition_mode(self, service, mock_recognizer, mock_event_bus):
        """Test handling audio chunk in recognition mode."""
        service.is_initialized = True
        service._training_active = False

        # Mock recognition result
        mock_recognizer.recognize_sound.return_value = ("test_sound", 0.8)
        mock_recognizer.get_mapping.return_value = "test_command"

        # Create mock event data
        event_data = Mock()
        event_data.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
        event_data.sample_rate = 16000

        await service._handle_audio_chunk(event_data)

        # Should call recognizer
        mock_recognizer.recognize_sound.assert_called_once()

        # Should publish recognition event
        mock_event_bus.publish.assert_called_once()
        published_event = mock_event_bus.publish.call_args[0][0]
        assert isinstance(published_event, CustomSoundRecognizedEvent)
        assert published_event.label == "test_sound"
        assert published_event.confidence == 0.8
        assert published_event.mapped_command == "test_command"

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_no_recognition(self, service, mock_recognizer, mock_event_bus):
        """Test handling audio chunk when no sound is recognized."""
        service.is_initialized = True
        service._training_active = False

        # Mock no recognition
        mock_recognizer.recognize_sound.return_value = None

        event_data = Mock()
        event_data.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
        event_data.sample_rate = 16000

        await service._handle_audio_chunk(event_data)

        # Should not publish any event
        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_esc50_sound(self, service, mock_recognizer, mock_event_bus):
        """Test handling audio chunk when ESC-50 sound is recognized."""
        service.is_initialized = True
        service._training_active = False

        # Mock ESC-50 recognition (should be ignored)
        mock_recognizer.recognize_sound.return_value = ("esc50_breathing", 0.7)

        event_data = Mock()
        event_data.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
        event_data.sample_rate = 16000

        await service._handle_audio_chunk(event_data)

        # Should not publish event for ESC-50 sounds
        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_audio_chunk_training_mode(self, service, mock_recognizer):
        """Test handling audio chunk in training mode."""
        service.is_initialized = True
        service._training_active = True
        service._current_training_label = "test_sound"
        service._training_samples = []
        service._target_samples = 5  # Set target to prevent auto-finish

        event_data = Mock()
        event_data.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
        event_data.sample_rate = 16000

        await service._handle_audio_chunk(event_data)

        # Should collect training sample
        assert len(service._training_samples) == 1
        audio, sr = service._training_samples[0]
        assert isinstance(audio, np.ndarray)
        assert sr == 16000

        # Should not call recognizer in training mode
        mock_recognizer.recognize_sound.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_training_success(self, service):
        """Test starting training mode."""
        service.is_initialized = True

        result = await service.start_training("test_sound")

        assert result is True
        assert service._training_active is True
        assert service._current_training_label == "test_sound"
        assert service._training_samples == []

    @pytest.mark.asyncio
    async def test_start_training_not_initialized(self, service):
        """Test starting training when service is not initialized."""
        service.is_initialized = False

        result = await service.start_training("test_sound")

        assert result is False
        assert service._training_active is False

    @pytest.mark.asyncio
    async def test_start_training_already_active(self, service):
        """Test starting training when already active."""
        service.is_initialized = True
        service._training_active = True
        service._current_training_label = "existing_sound"

        result = await service.start_training("new_sound")

        assert result is False
        assert service._current_training_label == "existing_sound"  # Unchanged

    @pytest.mark.asyncio
    async def test_finish_training_success(self, service, mock_recognizer):
        """Test finishing training successfully."""
        # Create fixed test data to avoid numpy array comparison issues
        test_audio = np.ones(1000)
        test_samples = [(test_audio, 16000)]

        service._training_active = True
        service._current_training_label = "test_sound"
        service._training_samples = test_samples

        mock_recognizer.train_sound.return_value = True

        result = await service.finish_training()

        assert result is True
        assert service._training_active is False
        assert service._current_training_label is None
        assert service._training_samples == []

        # Check that train_sound was called with correct arguments
        mock_recognizer.train_sound.assert_called_once()
        args, kwargs = mock_recognizer.train_sound.call_args
        assert args[0] == "test_sound"
        assert len(args[1]) == 1
        assert args[1][0][1] == 16000  # sample rate
        np.testing.assert_array_equal(args[1][0][0], test_audio)  # audio data

    @pytest.mark.asyncio
    async def test_finish_training_no_active_session(self, service, mock_recognizer):
        """Test finishing training when no session is active."""
        service._training_active = False

        result = await service.finish_training()

        assert result is False
        mock_recognizer.train_sound.assert_not_called()

    @pytest.mark.asyncio
    async def test_finish_training_no_samples(self, service, mock_recognizer):
        """Test finishing training with no collected samples."""
        service._training_active = True
        service._current_training_label = "test_sound"
        service._training_samples = []

        result = await service.finish_training()

        assert result is False
        assert service._training_active is False
        mock_recognizer.train_sound.assert_not_called()

    @pytest.mark.asyncio
    async def test_finish_training_failure(self, service, mock_recognizer):
        """Test finishing training when recognizer training fails."""
        service._training_active = True
        service._current_training_label = "test_sound"
        service._training_samples = [(np.random.randn(1000), 16000)]

        mock_recognizer.train_sound.return_value = False

        result = await service.finish_training()

        assert result is False
        assert service._training_active is False

    @pytest.mark.asyncio
    async def test_finish_training_exception(self, service, mock_recognizer):
        """Test finishing training with exception."""
        service._training_active = True
        service._current_training_label = "test_sound"
        service._training_samples = [(np.random.randn(1000), 16000)]

        mock_recognizer.train_sound.side_effect = Exception("Training error")

        result = await service.finish_training()

        assert result is False
        assert service._training_active is False

    def test_cancel_training(self, service):
        """Test canceling training session."""
        service._training_active = True
        service._current_training_label = "test_sound"
        service._training_samples = [(np.random.randn(1000), 16000)]

        service.cancel_training()

        assert service._training_active is False
        assert service._current_training_label is None
        assert service._training_samples == []

    def test_cancel_training_not_active(self, service):
        """Test canceling training when not active."""
        service._training_active = False

        # Should not raise error
        service.cancel_training()

        assert service._training_active is False

    @pytest.mark.asyncio
    async def test_set_sound_mapping(self, service, mock_recognizer):
        """Test setting sound mapping."""
        mock_recognizer.set_mapping.return_value = True

        success = await service.set_sound_mapping("test_sound", "test_command")

        assert success is True
        mock_recognizer.set_mapping.assert_called_once_with(sound_label="test_sound", command="test_command")

    def test_get_sound_mapping(self, service, mock_recognizer):
        """Test getting sound mapping."""
        mock_recognizer.get_mapping.return_value = "test_command"

        result = service.get_sound_mapping("test_sound")

        assert result == "test_command"
        mock_recognizer.get_mapping.assert_called_once_with(sound_label="test_sound")

    def test_get_stats(self, service, mock_recognizer):
        """Test getting service statistics."""
        service.is_initialized = True
        service._training_active = True
        service._current_training_label = "test_sound"
        service._training_samples = [(np.random.randn(1000), 16000)]

        mock_recognizer.get_stats.return_value = {
            "total_embeddings": 10,
            "custom_sounds": 2,
            "esc50_samples": 8,
            "mappings": 2,
            "model_ready": True,
        }

        stats = service.get_stats()

        assert stats["service_initialized"] is True
        assert stats["training_active"] is True
        assert stats["current_training_label"] == "test_sound"
        assert stats["training_samples_collected"] == 1
        assert stats["total_embeddings"] == 10
        assert stats["custom_sounds"] == 2

    def test_is_training_active(self, service):
        """Test checking if training is active."""
        service._training_active = False
        assert service.is_training_active() is False

        service._training_active = True
        assert service.is_training_active() is True

    def test_get_current_training_label(self, service):
        """Test getting current training label."""
        service._current_training_label = None
        assert service.get_current_training_label() is None

        service._current_training_label = "test_sound"
        assert service.get_current_training_label() == "test_sound"

    @pytest.mark.asyncio
    async def test_collect_training_sample_auto_finish(self, service, mock_recognizer, mock_config):
        """Test auto-finishing training after collecting enough samples."""
        # Set up service for training
        service.is_initialized = True
        service._training_active = True
        service._current_training_label = "test_sound"
        service._training_samples = []
        service._target_samples = 2  # Set target samples directly

        mock_recognizer.train_sound.return_value = True

        # Create event data
        event_data = Mock()
        event_data.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
        event_data.sample_rate = 16000

        # Collect first sample
        await service._handle_audio_chunk(event_data)
        assert len(service._training_samples) == 1
        assert service._training_active is True

        # Collect second sample (should auto-finish)
        await service._handle_audio_chunk(event_data)
        assert len(service._training_samples) == 0  # Should be reset after training
        assert service._training_active is False  # Should be finished

        # Should have called train_sound
        mock_recognizer.train_sound.assert_called_once()
