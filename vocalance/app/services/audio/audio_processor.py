import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseFloorEstimate:
    """Current noise floor estimate with metadata.

    Attributes:
        value: Estimated noise floor RMS energy (0-1 range).
        sample_count: Number of samples used in estimate.
        is_stable: True if enough samples collected for reliable estimate.
    """

    value: float
    sample_count: int
    is_stable: bool


class AudioProcessor:
    """Audio preprocessing with DC offset removal, peak normalization, and adaptive noise floor."""

    DEFAULT_TARGET_PEAK = 0.7
    MIN_PEAK_FOR_NORMALIZATION = 0.001

    NOISE_FLOOR_WINDOW_SIZE = 100
    NOISE_FLOOR_PERCENTILE = 30
    MIN_SAMPLES_FOR_STABLE = 20
    NOISE_FLOOR_INITIAL = 0.002
    BOOTSTRAP_CHUNKS = 40

    def __init__(
        self,
        sample_rate: int = 16000,
        enable_normalization: bool = True,
        target_peak: float = DEFAULT_TARGET_PEAK,
    ) -> None:
        self.sample_rate = sample_rate
        self.enable_normalization = enable_normalization
        self.target_peak = target_peak

        self._energy_history: deque = deque(maxlen=self.NOISE_FLOOR_WINDOW_SIZE)
        self._noise_floor = self.NOISE_FLOOR_INITIAL
        self._noise_floor_stable = False
        self._bootstrap_count = 0

        self._recent_peaks: deque = deque(maxlen=50)
        self._calibrated_gain = 1.0

        logger.debug(
            f"AudioProcessor initialized: sample_rate={sample_rate}, "
            f"normalization={'enabled' if enable_normalization else 'disabled'}, "
            f"target_peak={target_peak}"
        )

    def process_chunk(self, audio_bytes: bytes) -> tuple[np.ndarray, float]:
        """Process audio chunk: DC offset removal, peak normalization, RMS energy calculation.

        Returns:
            Tuple of (processed audio array, RMS energy).
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = self._remove_dc_offset(audio)
        if self.enable_normalization:
            audio = self._normalize_peak(audio)
        energy = self._calculate_rms_energy(audio)
        return audio, energy

    def calculate_energy(self, audio_bytes: bytes) -> float:
        """Calculate RMS energy of audio chunk with preprocessing.

        Convenience method when only energy is needed.

        Args:
            audio_bytes: Raw 16-bit PCM audio data.

        Returns:
            RMS energy in [0, 1] range.
        """
        _, energy = self.process_chunk(audio_bytes)
        return energy

    def update_noise_floor(self, energy: float, is_likely_speech: bool) -> NoiseFloorEstimate:
        """Update noise floor estimate with new energy sample.

        Uses a rolling window approach that:
        - Bootstrap period: collects ALL samples for first ~1.2s regardless of classification
        - After bootstrap: only considers samples that are likely silence
        - Uses 30th percentile for more aggressive noise rejection
        - Continuously adapts to changing environments

        Args:
            energy: RMS energy of current chunk.
            is_likely_speech: Whether this chunk likely contains speech.

        Returns:
            Current noise floor estimate with metadata.
        """
        self._bootstrap_count += 1

        # During bootstrap: add ALL samples to get initial noise floor estimate
        # This fixes the chicken-and-egg problem where threshold depends on noise floor
        # but noise floor depends on correct speech/silence classification
        if self._bootstrap_count <= self.BOOTSTRAP_CHUNKS:
            self._energy_history.append(energy)
        elif not is_likely_speech:
            # After bootstrap: only add silence samples
            self._energy_history.append(energy)

        # Update noise floor estimate if we have samples
        if len(self._energy_history) >= self.MIN_SAMPLES_FOR_STABLE:
            self._noise_floor = float(np.percentile(list(self._energy_history), self.NOISE_FLOOR_PERCENTILE))
            self._noise_floor_stable = True
        elif len(self._energy_history) > 0:
            # Use available samples but mark as unstable
            self._noise_floor = float(np.percentile(list(self._energy_history), self.NOISE_FLOOR_PERCENTILE))
            self._noise_floor_stable = False

        return NoiseFloorEstimate(
            value=self._noise_floor,
            sample_count=len(self._energy_history),
            is_stable=self._noise_floor_stable,
        )

    def get_noise_floor(self) -> NoiseFloorEstimate:
        """Get current noise floor estimate without updating it.

        Returns:
            NoiseFloorEstimate containing value, sample count, and stability flag.
        """
        return NoiseFloorEstimate(
            value=self._noise_floor,
            sample_count=len(self._energy_history),
            is_stable=self._noise_floor_stable,
        )

    def get_adaptive_threshold(self, base_multiplier: float = 3.0) -> float:
        """Calculate adaptive energy threshold based on noise floor.

        Args:
            base_multiplier: Multiplier applied to noise floor.

        Returns:
            Adaptive energy threshold.
        """
        return self._noise_floor * base_multiplier

    def is_above_noise_floor(self, energy: float, multiplier: float = 2.0) -> bool:
        """Check if energy is significantly above noise floor.

        Args:
            energy: RMS energy value to check.
            multiplier: How many times above noise floor to trigger.

        Returns:
            True if energy exceeds noise_floor * multiplier.
        """
        return energy > self._noise_floor * multiplier

    def reset(self) -> None:
        """Reset noise floor estimation and bootstrap state.

        Call when switching microphones or after environment change.
        """
        self._energy_history.clear()
        self._noise_floor = self.NOISE_FLOOR_INITIAL
        self._noise_floor_stable = False
        self._bootstrap_count = 0
        self._recent_peaks.clear()
        self._calibrated_gain = 1.0
        logger.debug("AudioProcessor reset")

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio signal.

        Some audio interfaces introduce DC bias that affects energy calculations.
        This subtracts the mean to center the signal around zero.

        Args:
            audio: Audio array in [-1, 1] range.

        Returns:
            DC-corrected audio array.
        """
        return audio - np.mean(audio)

    def _normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target peak level with attack/release asymmetry.

        This makes energy calculations consistent across microphones with
        different output levels. Uses asymmetric smoothing to avoid AGC pumping:
        - Fast attack: quickly reduce gain for loud sounds (prevents clipping)
        - Slow release: gradually increase gain after loud sounds (prevents noise amplification)

        Args:
            audio: Audio array in [-1, 1] range.

        Returns:
            Peak-normalized audio array.
        """
        peak = np.max(np.abs(audio))

        # Track recent peaks for smooth gain adjustment
        if peak > self.MIN_PEAK_FOR_NORMALIZATION:
            self._recent_peaks.append(peak)

        # Calculate smooth gain based on recent peaks
        if len(self._recent_peaks) > 0:
            # Use 90th percentile of recent peaks to avoid over-amplifying
            reference_peak = float(np.percentile(list(self._recent_peaks), 90))
            if reference_peak > self.MIN_PEAK_FOR_NORMALIZATION:
                target_gain = self.target_peak / reference_peak
                # Asymmetric smoothing to prevent AGC pumping artifacts
                if target_gain < self._calibrated_gain:
                    # Fast attack: respond quickly to loud sounds
                    self._calibrated_gain = 0.7 * self._calibrated_gain + 0.3 * target_gain
                else:
                    # Slow release: gradually increase gain after loud sounds
                    self._calibrated_gain = 0.98 * self._calibrated_gain + 0.02 * target_gain
                # Clamp gain to reasonable range
                self._calibrated_gain = max(0.5, min(5.0, self._calibrated_gain))

        return audio * self._calibrated_gain

    def _calculate_rms_energy(self, audio: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) energy.

        Args:
            audio: Audio array in any dtype.

        Returns:
            RMS energy value.
        """
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


class AdaptiveVADThreshold:
    """Adaptive VAD threshold that adjusts based on noise floor."""

    def __init__(
        self,
        speech_multiplier: float = 4.0,
        silence_multiplier: float = 2.0,
        min_threshold: float = 0.0003,
        max_threshold: float = 0.1,
    ) -> None:
        """Initialize adaptive threshold with multipliers and bounds.

        Args:
            speech_multiplier: Noise floor multiplier for speech detection.
            silence_multiplier: Noise floor multiplier for silence detection.
            min_threshold: Minimum allowed threshold (prevents over-sensitivity).
            max_threshold: Maximum allowed threshold (prevents under-sensitivity).
        """
        self.speech_multiplier = speech_multiplier
        self.silence_multiplier = silence_multiplier
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self._speech_threshold = min_threshold * speech_multiplier
        self._silence_threshold = min_threshold * silence_multiplier

    def update(self, noise_floor: float) -> tuple[float, float]:
        """Update thresholds based on noise floor.

        Args:
            noise_floor: Current noise floor estimate.

        Returns:
            Tuple of (speech_threshold, silence_threshold).
        """
        speech = noise_floor * self.speech_multiplier
        silence = noise_floor * self.silence_multiplier

        self._speech_threshold = max(self.min_threshold, min(self.max_threshold, speech))
        self._silence_threshold = max(self.min_threshold * 0.5, min(self.max_threshold * 0.5, silence))

        return self._speech_threshold, self._silence_threshold

    @property
    def speech_threshold(self) -> float:
        """Current speech detection threshold."""
        return self._speech_threshold

    @property
    def silence_threshold(self) -> float:
        """Current silence detection threshold."""
        return self._silence_threshold

    def is_speech(self, energy: float) -> bool:
        """Check if energy exceeds speech threshold."""
        return energy > self._speech_threshold

    def is_silence(self, energy: float) -> bool:
        """Check if energy is below silence threshold."""
        return energy < self._silence_threshold
