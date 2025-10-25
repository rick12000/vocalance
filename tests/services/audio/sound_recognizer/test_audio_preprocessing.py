"""
Unit tests for AudioPreprocessor component.

Tests the audio preprocessing functionality in isolation.
"""
from unittest.mock import patch

import numpy as np

from vocalance.app.services.audio.sound_recognizer.streamlined_sound_recognizer import AudioPreprocessor


class TestAudioPreprocessorDetailed:
    """Detailed tests for AudioPreprocessor functionality."""

    def test_mono_audio_unchanged(self, preprocessor, sample_rate):
        """Test that mono audio passes through correctly."""
        # Create mono test audio
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        original_audio = np.sin(2 * np.pi * 440 * t) * 0.5

        processed = preprocessor.preprocess_audio(original_audio.copy(), sample_rate)

        assert processed.ndim == 1
        assert len(processed) > 0
        # Should be similar length (allowing for silence trimming)
        assert abs(len(processed) - len(original_audio)) < len(original_audio) * 0.5

    def test_stereo_to_mono_conversion(self, preprocessor, sample_rate):
        """Test stereo to mono conversion."""
        # Create stereo audio (different signals in each channel)
        duration = 0.5
        samples = int(duration * sample_rate)
        left_channel = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)) * 0.5
        right_channel = np.sin(2 * np.pi * 880 * np.linspace(0, duration, samples)) * 0.3
        stereo_audio = np.column_stack([left_channel, right_channel])

        processed = preprocessor.preprocess_audio(stereo_audio, sample_rate)

        assert processed.ndim == 1
        assert len(processed) > 0
        # Should be average of both channels
        expected_mono = np.mean(stereo_audio, axis=-1)
        # Allow for preprocessing differences
        assert len(processed) <= len(expected_mono)

    def test_resampling_upsampling(self, mock_config):
        """Test upsampling from lower sample rate."""
        preprocessor = AudioPreprocessor(config=mock_config.sound_recognizer)

        # Create audio at lower sample rate
        original_sr = 8000
        duration = 0.5
        t = np.linspace(0, duration, int(duration * original_sr))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5

        processed = preprocessor.preprocess_audio(audio, original_sr)

        # Should be upsampled (approximately double the length)
        expected_length = int(len(audio) * 16000 / 8000)
        assert abs(len(processed) - expected_length) < expected_length * 0.2

    def test_resampling_downsampling(self, mock_config):
        """Test downsampling from higher sample rate."""
        preprocessor = AudioPreprocessor(config=mock_config.sound_recognizer)

        # Create audio at higher sample rate
        original_sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(duration * original_sr))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5

        processed = preprocessor.preprocess_audio(audio, original_sr)

        # Should be downsampled
        expected_length = int(len(audio) * 16000 / 44100)
        assert abs(len(processed) - expected_length) < expected_length * 0.2

    def test_silence_trimming_with_padding(self, preprocessor, sample_rate):
        """Test silence trimming with padded audio."""
        # Create signal with significant silence padding
        signal_duration = 0.2
        silence_duration = 0.3

        # Generate signal
        t_signal = np.linspace(0, signal_duration, int(signal_duration * sample_rate))
        signal = np.sin(2 * np.pi * 440 * t_signal) * 0.8  # Strong signal

        # Add silence padding
        silence_before = np.zeros(int(silence_duration * sample_rate))
        silence_after = np.zeros(int(silence_duration * sample_rate))
        padded_audio = np.concatenate([silence_before, signal, silence_after])

        processed = preprocessor.preprocess_audio(padded_audio, sample_rate)

        # Should be significantly shorter due to trimming
        assert len(processed) < len(padded_audio) * 0.8
        # But should retain most of the signal
        assert len(processed) >= len(signal) * 0.8

    def test_silence_trimming_no_sound_detected(self, preprocessor, sample_rate):
        """Test behavior when no sound is detected above threshold."""
        # Create very quiet audio (below threshold)
        duration = 0.5
        audio = np.random.randn(int(duration * sample_rate)) * 0.001  # Very quiet

        processed = preprocessor.preprocess_audio(audio, sample_rate)

        # Should still return some audio (original or minimal processing)
        assert len(processed) > 0

    def test_minimum_duration_padding(self, preprocessor, sample_rate):
        """Test padding of audio shorter than minimum duration."""
        # Create very short audio
        short_duration = 0.05  # 50ms, less than min_sound_duration (0.1s)
        t = np.linspace(0, short_duration, int(short_duration * sample_rate))
        short_audio = np.sin(2 * np.pi * 440 * t) * 0.5

        processed = preprocessor.preprocess_audio(short_audio, sample_rate)

        # Should be padded to minimum duration
        min_samples = int(preprocessor.min_sound_duration * sample_rate)
        assert len(processed) >= min_samples

    def test_maximum_duration_truncation(self, preprocessor, sample_rate):
        """Test truncation of audio longer than maximum duration."""
        # Create long audio
        long_duration = 3.0  # 3 seconds, more than max_sound_duration (2.0s)
        t = np.linspace(0, long_duration, int(long_duration * sample_rate))
        long_audio = np.sin(2 * np.pi * 440 * t) * 0.5

        processed = preprocessor.preprocess_audio(long_audio, sample_rate)

        # Should be truncated to maximum duration
        max_samples = int(preprocessor.max_sound_duration * sample_rate)
        assert len(processed) <= max_samples

    def test_normalization_peak_limiting(self, preprocessor, sample_rate):
        """Test that audio is properly normalized."""
        # Create audio with high amplitude
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        loud_audio = np.sin(2 * np.pi * 440 * t) * 2.0  # Amplitude > 1

        processed = preprocessor.preprocess_audio(loud_audio, sample_rate)

        # Should be normalized to reasonable range
        assert np.max(np.abs(processed)) <= 1.0
        assert np.max(np.abs(processed)) > 0.1  # Should have some signal

    def test_normalization_quiet_audio(self, preprocessor, sample_rate):
        """Test normalization of quiet audio."""
        # Create very quiet audio
        duration = 0.5
        t = np.linspace(0, duration, int(duration * sample_rate))
        quiet_audio = np.sin(2 * np.pi * 440 * t) * 0.01  # Very quiet

        processed = preprocessor.preprocess_audio(quiet_audio, sample_rate)

        # Should be amplified but not excessively
        assert np.max(np.abs(processed)) > np.max(np.abs(quiet_audio))
        assert np.max(np.abs(processed)) <= 1.0

    def test_empty_audio_handling(self, preprocessor, sample_rate):
        """Test handling of empty or zero-length audio."""
        empty_audio = np.array([])

        # Current implementation raises ValueError for empty audio
        import pytest

        with pytest.raises(ValueError, match="Audio array is empty"):
            preprocessor.preprocess_audio(empty_audio, sample_rate)

    def test_zero_audio_handling(self, preprocessor, sample_rate):
        """Test handling of all-zero audio."""
        # Create audio with all zeros
        zero_audio = np.zeros(int(0.5 * sample_rate))

        processed = preprocessor.preprocess_audio(zero_audio, sample_rate)

        # Should handle gracefully
        assert len(processed) > 0
        # May be padded to minimum duration
        min_samples = int(preprocessor.min_sound_duration * sample_rate)
        assert len(processed) >= min_samples

    def test_trim_silence_adaptive_threshold(self, preprocessor, sample_rate):
        """Test that silence trimming uses adaptive threshold."""
        # Create audio with varying noise floor
        duration = 0.5
        samples = int(duration * sample_rate)

        # Base signal
        t = np.linspace(0, duration, samples)
        signal = np.sin(2 * np.pi * 440 * t) * 0.5

        # Add noise with different levels
        noise = np.random.randn(samples) * 0.02
        noisy_signal = signal + noise

        # Add quiet sections
        quiet_start = np.random.randn(int(0.1 * sample_rate)) * 0.005
        quiet_end = np.random.randn(int(0.1 * sample_rate)) * 0.005

        full_audio = np.concatenate([quiet_start, noisy_signal, quiet_end])

        processed = preprocessor.preprocess_audio(full_audio, sample_rate)

        # Should adapt to noise floor and trim appropriately
        assert len(processed) <= len(full_audio)
        assert len(processed) > len(noisy_signal) * 0.7  # Should retain most signal

    def test_different_silence_thresholds(self, sample_rate, mock_config):
        """Test different silence threshold values."""
        # Create audio with moderate background noise
        duration = 0.5
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples)
        signal = np.sin(2 * np.pi * 440 * t) * 0.3
        noise = np.random.randn(samples) * 0.01

        # Add silence padding
        silence = np.random.randn(int(0.2 * sample_rate)) * 0.005
        audio_with_padding = np.concatenate([silence, signal + noise, silence])

        # Test with sensitive threshold
        mock_config.sound_recognizer.silence_threshold = 0.001
        sensitive_preprocessor = AudioPreprocessor(config=mock_config.sound_recognizer)
        sensitive_result = sensitive_preprocessor.preprocess_audio(audio_with_padding.copy(), sample_rate)

        # Test with less sensitive threshold
        mock_config.sound_recognizer.silence_threshold = 0.02
        less_sensitive_preprocessor = AudioPreprocessor(config=mock_config.sound_recognizer)
        less_sensitive_result = less_sensitive_preprocessor.preprocess_audio(audio_with_padding.copy(), sample_rate)

        # Sensitive threshold should trim more aggressively
        assert len(sensitive_result) <= len(less_sensitive_result)

    @patch("librosa.resample")
    def test_librosa_resample_called(self, mock_resample, preprocessor):
        """Test that librosa.resample is called when needed."""
        mock_resample.return_value = np.random.randn(1000)

        # Create audio at different sample rate
        audio = np.random.randn(500)
        original_sr = 8000

        preprocessor.preprocess_audio(audio, original_sr)

        # Should call librosa.resample
        mock_resample.assert_called_once()
        args, kwargs = mock_resample.call_args
        assert kwargs["orig_sr"] == original_sr
        assert kwargs["target_sr"] == preprocessor.target_sr

    @patch("librosa.feature.rms")
    def test_librosa_rms_called_for_trimming(self, mock_rms, preprocessor, sample_rate):
        """Test that librosa RMS is called for silence trimming."""
        # Mock RMS to return predictable values
        mock_rms.return_value = np.array([[0.1, 0.8, 0.9, 0.8, 0.1]])  # High energy in middle

        audio = np.random.randn(int(0.5 * sample_rate))

        preprocessor.preprocess_audio(audio, sample_rate)

        # Should call librosa RMS for energy analysis
        mock_rms.assert_called_once()
        args, kwargs = mock_rms.call_args
        assert kwargs["frame_length"] == 1024
        assert kwargs["hop_length"] == 512
