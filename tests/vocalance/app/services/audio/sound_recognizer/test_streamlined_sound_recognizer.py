from unittest.mock import Mock

import numpy as np
import pytest

from vocalance.app.services.audio.sound_recognizer.streamlined_sound_recognizer import AudioPreprocessor


def test_preprocessor_handles_mono_audio(sample_rate, mock_config):
    """Test preprocessing of mono audio."""
    preprocessor = AudioPreprocessor(config=mock_config.sound_recognizer)

    # Create test audio: 0.5 seconds of sine wave
    duration = 0.5
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5

    processed = preprocessor.preprocess_audio(audio, sample_rate)

    assert isinstance(processed, np.ndarray)
    assert processed.ndim == 1
    assert len(processed) > 0
    assert np.max(np.abs(processed)) <= 1.0  # Normalized


def test_preprocessor_handles_stereo_audio(sample_rate, mock_config):
    """Test conversion of stereo audio to mono."""
    preprocessor = AudioPreprocessor(config=mock_config.sound_recognizer)

    # Create stereo test audio
    duration = 0.5
    samples = int(duration * sample_rate)
    stereo_audio = np.random.randn(samples, 2) * 0.3

    processed = preprocessor.preprocess_audio(stereo_audio, sample_rate)

    assert processed.ndim == 1
    assert len(processed) > 0


def test_preprocessor_resamples_audio(mock_config):
    """Test audio resampling functionality."""
    preprocessor = AudioPreprocessor(config=mock_config.sound_recognizer)

    # Create audio at different sample rate
    original_sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(duration * original_sr))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5

    processed = preprocessor.preprocess_audio(audio, original_sr)

    # Should be resampled to target rate
    expected_length = int(duration * 16000)
    assert abs(len(processed) - expected_length) < 100  # Allow small tolerance


def test_preprocessor_pads_short_audio(sample_rate, mock_config):
    """Test padding of short audio to minimum duration."""
    mock_config.sound_recognizer.min_sound_duration = 0.2
    preprocessor = AudioPreprocessor(config=mock_config.sound_recognizer)

    # Create very short audio
    short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.05, int(0.05 * sample_rate))) * 0.5

    processed = preprocessor.preprocess_audio(short_audio, sample_rate)

    # Should be padded to minimum duration
    min_samples = int(0.2 * sample_rate)
    assert len(processed) >= min_samples


def test_preprocessor_truncates_long_audio(sample_rate, mock_config):
    """Test truncation of long audio to maximum duration."""
    mock_config.sound_recognizer.max_sound_duration = 1.0
    preprocessor = AudioPreprocessor(config=mock_config.sound_recognizer)

    # Create long audio
    long_duration = 2.0
    t = np.linspace(0, long_duration, int(long_duration * sample_rate))
    long_audio = np.sin(2 * np.pi * 440 * t) * 0.5

    processed = preprocessor.preprocess_audio(long_audio, sample_rate)

    # Should be truncated to maximum duration
    max_samples = int(1.0 * sample_rate)
    assert len(processed) <= max_samples


@pytest.mark.asyncio
async def test_recognizer_initializes_successfully(isolated_recognizer):
    """Test successful initialization."""
    result = await isolated_recognizer.initialize()
    assert result is True


def test_embedding_extraction_works(isolated_recognizer, sample_rate):
    """Test embedding extraction with mocked YAMNet."""
    # Create test audio
    duration = 0.5
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5

    embedding = isolated_recognizer._extract_embedding(audio, sample_rate)

    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1024,)  # YAMNet embedding size


def test_recognize_returns_none_without_training(isolated_recognizer, sample_rate):
    """Test recognition with no trained sounds."""
    duration = 0.5
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5

    result = isolated_recognizer.recognize_sound(audio, sample_rate)

    assert result is None


def test_recognize_fails_with_high_confidence_threshold(isolated_recognizer, sample_rate):
    """Test recognition fails with very high confidence threshold."""
    # Add some training data
    isolated_recognizer.embeddings = np.random.randn(3, 1024)
    isolated_recognizer.labels = ["test_sound"] * 3
    isolated_recognizer.scaler.fit(isolated_recognizer.embeddings)

    # Set very high confidence threshold
    isolated_recognizer.confidence_threshold = 0.99

    duration = 0.5
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5

    result = isolated_recognizer.recognize_sound(audio, sample_rate)

    # Should fail due to low confidence
    assert result is None


@pytest.mark.asyncio
async def test_training_adds_embeddings(isolated_recognizer, sample_rate):
    """Test that training adds embeddings to the model."""
    # Create training data
    duration = 0.5
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    samples = [(audio, sample_rate)]

    initial_count = len(isolated_recognizer.embeddings)

    result = await isolated_recognizer.train_sound("test_sound", samples)

    assert result is True
    assert len(isolated_recognizer.embeddings) > initial_count
    assert "test_sound" in isolated_recognizer.labels


@pytest.mark.asyncio
async def test_train_sound_handles_invalid_input(isolated_recognizer):
    """Test training with samples that fail embedding extraction."""
    await isolated_recognizer.initialize()

    # Mock _extract_embedding to return None
    isolated_recognizer._extract_embedding = Mock(return_value=None)

    samples = [(np.random.randn(1000), 16000)]
    result = await isolated_recognizer.train_sound("test_sound", samples)

    assert result is False
    assert len(isolated_recognizer.embeddings) == 0


@pytest.mark.asyncio
async def test_mapping_functionality(isolated_recognizer):
    """Test sound-to-command mapping functionality."""
    success = await isolated_recognizer.set_mapping("test_sound", "test_command")

    assert success is True
    assert isolated_recognizer.get_mapping("test_sound") == "test_command"
    assert isolated_recognizer.get_mapping("nonexistent") is None


def test_stats_report_correctly(isolated_recognizer):
    """Test that statistics are reported correctly."""
    # Test empty stats
    stats = isolated_recognizer.get_stats()
    assert stats["total_embeddings"] == 0
    assert stats["model_ready"] is False

    # Test with data
    isolated_recognizer.embeddings = np.random.randn(5, 1024)
    isolated_recognizer.labels = ["custom1", "custom1", "esc50_breathing", "custom2", "esc50_coughing"]
    isolated_recognizer.mappings = {"custom1": "command1", "custom2": "command2"}

    stats = isolated_recognizer.get_stats()
    assert stats["total_embeddings"] == 5
    assert stats["custom_sounds"] == 2
    assert stats["esc50_samples"] == 2
    assert stats["mappings"] == 2
    assert stats["model_ready"] is True
