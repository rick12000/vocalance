"""Tests for AudioProcessor - audio normalization and adaptive noise floor estimation."""

import numpy as np
import pytest

from vocalance.app.services.audio.audio_processor import AdaptiveVADThreshold, AudioProcessor, NoiseFloorEstimate


class TestAudioProcessor:
    """Tests for AudioProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a fresh AudioProcessor for each test."""
        return AudioProcessor(sample_rate=16000, enable_normalization=True)

    @pytest.fixture
    def silence_bytes(self):
        """Create a very quiet audio chunk (simulating silence)."""
        chunk = np.random.randint(-10, 10, size=800, dtype=np.int16)
        return chunk.tobytes()

    @pytest.fixture
    def speech_bytes(self):
        """Create a louder audio chunk (simulating speech)."""
        chunk = np.random.randint(-5000, 5000, size=800, dtype=np.int16)
        return chunk.tobytes()

    @pytest.fixture
    def loud_bytes(self):
        """Create a very loud audio chunk."""
        chunk = np.random.randint(-25000, 25000, size=800, dtype=np.int16)
        return chunk.tobytes()

    def test_process_chunk_returns_tuple(self, processor, speech_bytes):
        """Test that process_chunk returns (audio_array, energy) tuple."""
        audio, energy = processor.process_chunk(speech_bytes)

        assert isinstance(audio, np.ndarray)
        assert isinstance(energy, float)
        assert len(audio) == 800  # Same number of samples
        assert 0.0 <= energy <= 1.0

    def test_calculate_energy_returns_float(self, processor, speech_bytes):
        """Test that calculate_energy returns a float."""
        energy = processor.calculate_energy(speech_bytes)

        assert isinstance(energy, float)
        assert 0.0 <= energy <= 1.0

    def test_silence_has_lower_energy_than_speech(self, processor, silence_bytes, speech_bytes):
        """Test that silence has lower energy than speech."""
        silence_energy = processor.calculate_energy(silence_bytes)
        speech_energy = processor.calculate_energy(speech_bytes)

        assert silence_energy < speech_energy

    def test_dc_offset_removal(self, processor):
        """Test that DC offset is removed from audio."""
        # Create audio with a DC offset
        chunk = np.full(800, 1000, dtype=np.int16)  # Constant value = DC offset
        audio_bytes = chunk.tobytes()

        audio, _ = processor.process_chunk(audio_bytes)

        # After DC offset removal, mean should be close to 0
        assert abs(np.mean(audio)) < 0.01

    def test_noise_floor_estimation_initial_state(self, processor):
        """Test that noise floor starts with initial estimate."""
        estimate = processor.get_noise_floor()

        assert isinstance(estimate, NoiseFloorEstimate)
        assert estimate.value == AudioProcessor.NOISE_FLOOR_INITIAL
        assert estimate.sample_count == 0
        assert estimate.is_stable is False

    def test_noise_floor_updates_with_silence_samples(self, processor, silence_bytes):
        """Test that noise floor updates as silence samples are added."""
        initial_count = processor.get_noise_floor().sample_count

        # Process silence samples (not speech)
        for _ in range(25):
            energy = processor.calculate_energy(silence_bytes)
            processor.update_noise_floor(energy, is_likely_speech=False)

        final_estimate = processor.get_noise_floor()

        assert final_estimate.sample_count > initial_count
        assert final_estimate.is_stable is True  # 25 > MIN_SAMPLES_FOR_STABLE (20)

    def test_noise_floor_ignores_speech_samples_after_bootstrap(self, processor, speech_bytes, silence_bytes):
        """Test that noise floor ignores speech samples after bootstrap period.

        During the bootstrap period (~40 chunks), ALL samples are collected to establish
        an initial noise floor. After bootstrap, only silence samples are added.
        """
        # First, complete the bootstrap period with silence samples
        for _ in range(processor.BOOTSTRAP_CHUNKS + 5):
            energy = processor.calculate_energy(silence_bytes)
            processor.update_noise_floor(energy, is_likely_speech=False)

        # Record count after bootstrap
        post_bootstrap_count = processor.get_noise_floor().sample_count

        # Process speech samples (should be ignored after bootstrap)
        for _ in range(10):
            energy = processor.calculate_energy(speech_bytes)
            processor.update_noise_floor(energy, is_likely_speech=True)

        final_count = processor.get_noise_floor().sample_count

        # No new samples should be added since we're past bootstrap and marking as speech
        assert final_count == post_bootstrap_count

    def test_get_adaptive_threshold(self, processor, silence_bytes):
        """Test that adaptive threshold is based on noise floor."""
        # Build up noise floor estimate
        for _ in range(25):
            energy = processor.calculate_energy(silence_bytes)
            processor.update_noise_floor(energy, is_likely_speech=False)

        noise_floor = processor.get_noise_floor().value
        threshold = processor.get_adaptive_threshold(base_multiplier=3.0)

        assert threshold == noise_floor * 3.0

    def test_is_above_noise_floor(self, processor, silence_bytes, speech_bytes):
        """Test is_above_noise_floor detection."""
        # Build up noise floor estimate with silence
        for _ in range(25):
            energy = processor.calculate_energy(silence_bytes)
            processor.update_noise_floor(energy, is_likely_speech=False)

        processor.calculate_energy(silence_bytes)
        speech_energy = processor.calculate_energy(speech_bytes)

        # Silence should not be significantly above noise floor
        # Speech should be significantly above noise floor
        assert processor.is_above_noise_floor(speech_energy, multiplier=2.0)

    def test_reset_clears_state(self, processor, silence_bytes):
        """Test that reset clears all state."""
        # Build up some state
        for _ in range(25):
            energy = processor.calculate_energy(silence_bytes)
            processor.update_noise_floor(energy, is_likely_speech=False)

        assert processor.get_noise_floor().sample_count > 0

        # Reset
        processor.reset()

        estimate = processor.get_noise_floor()
        assert estimate.sample_count == 0
        assert estimate.is_stable is False
        assert estimate.value == AudioProcessor.NOISE_FLOOR_INITIAL

    def test_peak_normalization_effect(self, processor, loud_bytes, silence_bytes):
        """Test that peak normalization brings different levels closer."""
        # Process a loud chunk first to calibrate
        loud_audio, loud_energy = processor.process_chunk(loud_bytes)

        # Process silence - should not be amplified excessively
        silence_audio, silence_energy = processor.process_chunk(silence_bytes)

        # The loud audio should be normalized down
        assert np.max(np.abs(loud_audio)) <= 1.0

        # Silence should remain quiet
        assert silence_energy < loud_energy


class TestAdaptiveVADThreshold:
    """Tests for AdaptiveVADThreshold class."""

    @pytest.fixture
    def threshold(self):
        """Create an AdaptiveVADThreshold for testing."""
        return AdaptiveVADThreshold(
            speech_multiplier=4.0,
            silence_multiplier=2.0,
            min_threshold=0.001,
            max_threshold=0.1,
        )

    def test_initial_thresholds(self, threshold):
        """Test that thresholds are initialized correctly."""
        assert threshold.speech_threshold > 0
        assert threshold.silence_threshold > 0
        assert threshold.silence_threshold < threshold.speech_threshold

    def test_update_adjusts_thresholds(self, threshold):
        """Test that update adjusts thresholds based on noise floor."""
        noise_floor = 0.005

        speech, silence = threshold.update(noise_floor)

        assert speech == noise_floor * 4.0  # speech_multiplier
        assert silence == noise_floor * 2.0  # silence_multiplier

    def test_thresholds_clamped_to_min(self, threshold):
        """Test that thresholds don't go below minimum."""
        very_low_noise = 0.00001  # Very low noise floor

        speech, silence = threshold.update(very_low_noise)

        assert speech >= threshold.min_threshold
        assert silence >= threshold.min_threshold * 0.5

    def test_thresholds_clamped_to_max(self, threshold):
        """Test that thresholds don't exceed maximum."""
        very_high_noise = 1.0  # Very high noise floor

        speech, silence = threshold.update(very_high_noise)

        assert speech <= threshold.max_threshold
        assert silence <= threshold.max_threshold * 0.5

    def test_is_speech_detection(self, threshold):
        """Test is_speech method."""
        threshold.update(0.01)  # Set noise floor to 0.01

        # Energy above speech threshold should be speech
        assert threshold.is_speech(0.05)  # > 0.01 * 4.0 = 0.04

        # Energy below speech threshold should not be speech
        assert not threshold.is_speech(0.02)  # < 0.04

    def test_is_silence_detection(self, threshold):
        """Test is_silence method."""
        threshold.update(0.01)  # Set noise floor to 0.01

        # Energy below silence threshold should be silence
        assert threshold.is_silence(0.01)  # < 0.01 * 2.0 = 0.02

        # Energy above silence threshold should not be silence
        assert not threshold.is_silence(0.03)  # > 0.02
