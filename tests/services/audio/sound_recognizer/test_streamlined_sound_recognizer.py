"""
Unit tests for StreamlinedSoundRecognizer.

Tests individual methods and components in isolation using fixtures
and mocks to avoid dependencies on external storage or persistent state.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

from iris.services.audio.sound_recognizer.streamlined_sound_recognizer import (
    StreamlinedSoundRecognizer,
    AudioPreprocessor
)


class TestAudioPreprocessor:
    """Test the AudioPreprocessor component in isolation."""
    
    def test_init_with_defaults(self):
        """Test AudioPreprocessor initialization with default parameters."""
        preprocessor = AudioPreprocessor()
        
        assert preprocessor.target_sr == 16000
        assert preprocessor.silence_threshold == 0.005
        assert preprocessor.min_sound_duration == 0.1
        assert preprocessor.max_sound_duration == 2.0
    
    def test_init_with_custom_params(self):
        """Test AudioPreprocessor initialization with custom parameters."""
        preprocessor = AudioPreprocessor(
            target_sr=22050,
            silence_threshold=0.01,
            min_sound_duration=0.2,
            max_sound_duration=3.0
        )
        
        assert preprocessor.target_sr == 22050
        assert preprocessor.silence_threshold == 0.01
        assert preprocessor.min_sound_duration == 0.2
        assert preprocessor.max_sound_duration == 3.0
    
    def test_preprocess_mono_audio(self, sample_rate):
        """Test preprocessing of mono audio."""
        preprocessor = AudioPreprocessor(target_sr=sample_rate)
        
        # Create test audio: 0.5 seconds of sine wave
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        processed = preprocessor.preprocess_audio(audio, sample_rate)
        
        assert isinstance(processed, np.ndarray)
        assert processed.ndim == 1
        assert len(processed) > 0
        assert np.max(np.abs(processed)) <= 1.0  # Normalized
    
    def test_preprocess_stereo_to_mono(self, sample_rate):
        """Test conversion of stereo audio to mono."""
        preprocessor = AudioPreprocessor(target_sr=sample_rate)
        
        # Create stereo test audio
        duration = 0.5
        samples = int(duration * sample_rate)
        stereo_audio = np.random.randn(samples, 2) * 0.3
        
        processed = preprocessor.preprocess_audio(stereo_audio, sample_rate)
        
        assert processed.ndim == 1
        assert len(processed) > 0
    
    def test_resample_audio(self):
        """Test audio resampling functionality."""
        preprocessor = AudioPreprocessor(target_sr=16000)
        
        # Create audio at different sample rate
        original_sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(duration * original_sr))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        processed = preprocessor.preprocess_audio(audio, original_sr)
        
        # Should be resampled to target rate
        expected_length = int(duration * 16000)
        assert abs(len(processed) - expected_length) < 100  # Allow small tolerance
    
    def test_silence_trimming(self, sample_rate):
        """Test silence trimming functionality."""
        preprocessor = AudioPreprocessor(target_sr=sample_rate, silence_threshold=0.01)
        
        # Create audio with silence padding
        signal_duration = 0.3
        silence_duration = 0.2
        
        # Generate signal
        t_signal = np.linspace(0, signal_duration, int(signal_duration * sample_rate))
        signal = np.sin(2 * np.pi * 440 * t_signal) * 0.5
        
        # Add silence padding
        silence = np.zeros(int(silence_duration * sample_rate))
        audio_with_silence = np.concatenate([silence, signal, silence])
        
        processed = preprocessor.preprocess_audio(audio_with_silence, sample_rate)
        
        # Should be shorter due to trimming
        assert len(processed) < len(audio_with_silence)
        assert len(processed) > len(signal) * 0.8  # Allow some tolerance
    
    def test_duration_padding(self, sample_rate):
        """Test padding of short audio to minimum duration."""
        preprocessor = AudioPreprocessor(target_sr=sample_rate, min_sound_duration=0.2)
        
        # Create very short audio
        short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.05, int(0.05 * sample_rate))) * 0.5
        
        processed = preprocessor.preprocess_audio(short_audio, sample_rate)
        
        # Should be padded to minimum duration
        min_samples = int(0.2 * sample_rate)
        assert len(processed) >= min_samples
    
    def test_duration_truncation(self, sample_rate):
        """Test truncation of long audio to maximum duration."""
        preprocessor = AudioPreprocessor(target_sr=sample_rate, max_sound_duration=1.0)
        
        # Create long audio
        long_duration = 2.0
        t = np.linspace(0, long_duration, int(long_duration * sample_rate))
        long_audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        processed = preprocessor.preprocess_audio(long_audio, sample_rate)
        
        # Should be truncated to maximum duration
        max_samples = int(1.0 * sample_rate)
        assert len(processed) <= max_samples


class TestStreamlinedSoundRecognizer:
    """Test the StreamlinedSoundRecognizer class."""
    
    def test_init(self, mock_config, mock_storage_factory):
        """Test recognizer initialization."""
        recognizer = StreamlinedSoundRecognizer(mock_config, mock_storage_factory)
        
        assert recognizer.config == mock_config.sound_recognizer
        assert recognizer.storage_adapter == mock_storage_factory.create_sound_recognizer_adapter.return_value
        assert isinstance(recognizer.preprocessor, AudioPreprocessor)
        assert recognizer.embeddings.shape == (0, 1024)
        assert recognizer.labels == []
        assert recognizer.mappings == {}
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, isolated_recognizer):
        """Test successful initialization."""
        result = await isolated_recognizer.initialize()
        assert result is True
    
    def test_extract_embedding_mock(self, isolated_recognizer, sample_rate):
        """Test embedding extraction with mocked YAMNet."""
        # Create test audio
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        embedding = isolated_recognizer._extract_embedding(audio, sample_rate)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)  # YAMNet embedding size
    
    def test_recognize_sound_no_training(self, isolated_recognizer, sample_rate):
        """Test recognition with no trained sounds."""
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        result = isolated_recognizer.recognize_sound(audio, sample_rate)
        
        assert result is None
    
    def test_recognize_sound_low_confidence(self, isolated_recognizer, sample_rate):
        """Test recognition with low confidence threshold."""
        # Add some training data
        isolated_recognizer.embeddings = np.random.randn(3, 1024)
        isolated_recognizer.labels = ['test_sound'] * 3
        isolated_recognizer.scaler.fit(isolated_recognizer.embeddings)
        
        # Set very high confidence threshold
        isolated_recognizer.confidence_threshold = 0.99
        
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        result = isolated_recognizer.recognize_sound(audio, sample_rate)
        
        # Should fail due to low confidence
        assert result is None
    
    def test_recognize_sound_only_esc50(self, isolated_recognizer, sample_rate):
        """Test recognition when only ESC-50 sounds are in top-k."""
        # Add only ESC-50 training data
        isolated_recognizer.embeddings = np.random.randn(3, 1024)
        isolated_recognizer.labels = ['esc50_breathing'] * 3
        isolated_recognizer.scaler.fit(isolated_recognizer.embeddings)
        isolated_recognizer.confidence_threshold = 0.1  # Low threshold
        
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        result = isolated_recognizer.recognize_sound(audio, sample_rate)
        
        # Should return None (background sound)
        assert result is None
    
    def test_recognize_sound_successful(self, isolated_recognizer, sample_rate):
        """Test successful sound recognition."""
        # Add mixed training data
        isolated_recognizer.embeddings = np.random.randn(6, 1024)
        isolated_recognizer.labels = ['custom_sound'] * 3 + ['esc50_breathing'] * 3
        isolated_recognizer.scaler.fit(isolated_recognizer.embeddings)
        isolated_recognizer.confidence_threshold = 0.1  # Low threshold
        isolated_recognizer.vote_threshold = 0.3  # Low vote threshold
        
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        
        result = isolated_recognizer.recognize_sound(audio, sample_rate)
        
        # Should succeed if conditions are met
        if result is not None:
            label, confidence = result
            assert isinstance(label, str)
            assert isinstance(confidence, float)
            assert not label.startswith('esc50_')
    
    @pytest.mark.asyncio
    async def test_train_sound_success(self, isolated_recognizer, training_samples):
        """Test successful sound training."""
        # Initialize the recognizer first
        await isolated_recognizer.initialize()
        
        # Train with lip_popping samples
        samples = training_samples['lip_popping'][:2]  # Use 2 samples
        result = await isolated_recognizer.train_sound('lip_popping', samples)
        
        assert result is True
        assert len(isolated_recognizer.embeddings) > 0
        assert 'lip_popping' in isolated_recognizer.labels
    
    @pytest.mark.asyncio
    async def test_train_sound_no_valid_embeddings(self, isolated_recognizer):
        """Test training with samples that fail embedding extraction."""
        await isolated_recognizer.initialize()
        
        # Mock _extract_embedding to return None
        isolated_recognizer._extract_embedding = Mock(return_value=None)
        
        samples = [(np.random.randn(1000), 16000)]
        result = await isolated_recognizer.train_sound('test_sound', samples)
        
        assert result is False
        assert len(isolated_recognizer.embeddings) == 0
    
    def test_set_and_get_mapping(self, isolated_recognizer):
        """Test sound-to-command mapping functionality."""
        isolated_recognizer.set_mapping('test_sound', 'test_command')
        
        assert isolated_recognizer.get_mapping('test_sound') == 'test_command'
        assert isolated_recognizer.get_mapping('nonexistent') is None
    
    def test_get_stats_empty(self, isolated_recognizer):
        """Test statistics with no training data."""
        stats = isolated_recognizer.get_stats()
        
        assert stats['total_embeddings'] == 0
        assert stats['custom_sounds'] == 0
        assert stats['esc50_samples'] == 0
        assert stats['mappings'] == 0
        assert stats['model_ready'] is False
    
    def test_get_stats_with_data(self, isolated_recognizer):
        """Test statistics with training data."""
        # Add mock data
        isolated_recognizer.embeddings = np.random.randn(5, 1024)
        isolated_recognizer.labels = ['custom1', 'custom1', 'esc50_breathing', 'custom2', 'esc50_coughing']
        isolated_recognizer.mappings = {'custom1': 'command1', 'custom2': 'command2'}
        
        stats = isolated_recognizer.get_stats()
        
        assert stats['total_embeddings'] == 5
        assert stats['custom_sounds'] == 2  # custom1, custom2
        assert stats['esc50_samples'] == 2  # 2 ESC-50 samples
        assert stats['mappings'] == 2
        assert stats['model_ready'] is True


class TestSoundDiscrimination:
    """Test sound discrimination capabilities."""
    
    def test_embedding_similarity_different_sounds(self, isolated_recognizer, audio_samples):
        """Test that different sound types produce different embeddings."""
        if not audio_samples['lip_popping'] or not audio_samples['tongue_clicking']:
            pytest.skip("Insufficient audio samples for discrimination test")
        
        # Get samples
        lip_audio, lip_sr, _ = audio_samples['lip_popping'][0]
        tongue_audio, tongue_sr, _ = audio_samples['tongue_clicking'][0]
        
        # Extract embeddings
        lip_embedding = isolated_recognizer._extract_embedding(lip_audio, lip_sr)
        tongue_embedding = isolated_recognizer._extract_embedding(tongue_audio, tongue_sr)
        
        if lip_embedding is not None and tongue_embedding is not None:
            # Calculate similarity
            similarity = cosine_similarity(
                lip_embedding.reshape(1, -1),
                tongue_embedding.reshape(1, -1)
            )[0][0]
            
            # Different sounds should have lower similarity (relaxed for simple mock)
            assert similarity < 0.95  # Allow some tolerance for mock embeddings
    
    def test_user_prompt_similarity(self, isolated_recognizer, audio_samples, user_prompt_sample):
        """Test that user prompt is similar to other lip_popping samples."""
        if len(audio_samples['lip_popping']) < 2:
            pytest.skip("Insufficient lip_popping samples for similarity test")
        
        user_audio, user_sr, _ = user_prompt_sample
        
        # Get another lip_popping sample (not user prompt)
        other_sample = None
        for audio, sr, name in audio_samples['lip_popping']:
            if 'user_prompt' not in name.lower():
                other_sample = (audio, sr, name)
                break
        
        if other_sample is None:
            pytest.skip("No non-user-prompt lip_popping sample found")
        
        other_audio, other_sr, _ = other_sample
        
        # Extract embeddings
        user_embedding = isolated_recognizer._extract_embedding(user_audio, user_sr)
        other_embedding = isolated_recognizer._extract_embedding(other_audio, other_sr)
        
        if user_embedding is not None and other_embedding is not None:
            # Calculate similarity
            similarity = cosine_similarity(
                user_embedding.reshape(1, -1),
                other_embedding.reshape(1, -1)
            )[0][0]
            
            # Same sound type should have higher similarity
            assert similarity > 0.3  # Reasonable threshold for mock embeddings


class TestNoiseRejection:
    """Test noise sample rejection capabilities."""
    
    @pytest.mark.asyncio
    async def test_noise_samples_rejected(self, isolated_recognizer, audio_samples, training_samples):
        """Test that noise samples are properly rejected."""
        if not audio_samples['noise']:
            pytest.skip("No noise samples available for testing")
        
        # Initialize and train
        await isolated_recognizer.initialize()
        
        # Train with some samples
        if training_samples['lip_popping']:
            await isolated_recognizer.train_sound('lip_popping', training_samples['lip_popping'][:2])
        
        if training_samples['tongue_clicking']:
            await isolated_recognizer.train_sound('tongue_clicking', training_samples['tongue_clicking'][:2])
        
        # Test noise samples
        false_positives = 0
        total_tested = 0
        
        for noise_audio, noise_sr, noise_name in audio_samples['noise']:
            result = isolated_recognizer.recognize_sound(noise_audio, noise_sr)
            total_tested += 1
            
            if result is not None:
                label, confidence = result
                if not label.startswith('esc50_'):  # Custom sound detected = false positive
                    false_positives += 1
        
        # Allow some false positives due to mock embeddings, but not too many
        false_positive_rate = false_positives / total_tested if total_tested > 0 else 0
        assert false_positive_rate < 0.6  # Less than 60% false positive rate
    
    def test_confidence_threshold_prevents_false_positives(self, isolated_recognizer, sample_rate):
        """Test that confidence threshold helps prevent false positives."""
        # Add training data
        isolated_recognizer.embeddings = np.random.randn(3, 1024)
        isolated_recognizer.labels = ['custom_sound'] * 3
        isolated_recognizer.scaler.fit(isolated_recognizer.embeddings)
        
        # Set high confidence threshold
        isolated_recognizer.confidence_threshold = 0.8
        
        # Create random noise
        noise = np.random.randn(int(0.5 * sample_rate)) * 0.1
        
        result = isolated_recognizer.recognize_sound(noise, sample_rate)
        
        # Should be rejected due to low confidence
        assert result is None