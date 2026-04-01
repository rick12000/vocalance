from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseFloorEstimate:
    """RMS noise floor estimate from recent chunks."""

    value: float
    sample_count: int
    is_stable: bool


class AudioProcessor:
    """DC removal, optional peak normalization, RMS energy, rolling noise floor."""

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

    def process_chunk(self, audio_bytes: bytes) -> tuple[np.ndarray, float]:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = self._remove_dc_offset(audio)
        if self.enable_normalization:
            audio = self._normalize_peak(audio)
        energy = self._calculate_rms_energy(audio)
        return audio, energy

    def calculate_energy(self, audio_bytes: bytes) -> float:
        return self.process_chunk(audio_bytes)[1]

    def update_noise_floor(self, energy: float, is_likely_speech: bool) -> NoiseFloorEstimate:
        self._bootstrap_count += 1

        if self._bootstrap_count <= self.BOOTSTRAP_CHUNKS:
            self._energy_history.append(energy)
        elif not is_likely_speech:
            self._energy_history.append(energy)

        if len(self._energy_history) >= self.MIN_SAMPLES_FOR_STABLE:
            self._noise_floor = float(np.percentile(list(self._energy_history), self.NOISE_FLOOR_PERCENTILE))
            self._noise_floor_stable = True
        elif len(self._energy_history) > 0:
            self._noise_floor = float(np.percentile(list(self._energy_history), self.NOISE_FLOOR_PERCENTILE))
            self._noise_floor_stable = False

        return NoiseFloorEstimate(
            value=self._noise_floor,
            sample_count=len(self._energy_history),
            is_stable=self._noise_floor_stable,
        )

    def get_noise_floor(self) -> NoiseFloorEstimate:
        return NoiseFloorEstimate(
            value=self._noise_floor,
            sample_count=len(self._energy_history),
            is_stable=self._noise_floor_stable,
        )

    def get_adaptive_threshold(self, base_multiplier: float = 3.0) -> float:
        return self._noise_floor * base_multiplier

    def is_above_noise_floor(self, energy: float, multiplier: float = 2.0) -> bool:
        return energy > self._noise_floor * multiplier

    def reset(self) -> None:
        self._energy_history.clear()
        self._noise_floor = self.NOISE_FLOOR_INITIAL
        self._noise_floor_stable = False
        self._bootstrap_count = 0
        self._recent_peaks.clear()
        self._calibrated_gain = 1.0

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        return audio - np.mean(audio)

    def _normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))

        if peak > self.MIN_PEAK_FOR_NORMALIZATION:
            self._recent_peaks.append(peak)

        if len(self._recent_peaks) > 0:
            reference_peak = float(np.percentile(list(self._recent_peaks), 90))
            if reference_peak > self.MIN_PEAK_FOR_NORMALIZATION:
                target_gain = self.target_peak / reference_peak
                if target_gain < self._calibrated_gain:
                    self._calibrated_gain = 0.7 * self._calibrated_gain + 0.3 * target_gain
                else:
                    self._calibrated_gain = 0.98 * self._calibrated_gain + 0.02 * target_gain
                self._calibrated_gain = max(0.5, min(5.0, self._calibrated_gain))

        return audio * self._calibrated_gain

    def _calculate_rms_energy(self, audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


class AdaptiveVADThreshold:
    """Speech / silence thresholds derived from the noise floor."""

    def __init__(
        self,
        speech_multiplier: float = 4.0,
        silence_multiplier: float = 2.0,
        min_threshold: float = 0.0003,
        max_threshold: float = 0.1,
    ) -> None:
        self.speech_multiplier = speech_multiplier
        self.silence_multiplier = silence_multiplier
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self._speech_threshold = min_threshold * speech_multiplier
        self._silence_threshold = min_threshold * silence_multiplier

    def update(self, noise_floor: float) -> tuple[float, float]:
        speech = noise_floor * self.speech_multiplier
        silence = noise_floor * self.silence_multiplier

        self._speech_threshold = max(self.min_threshold, min(self.max_threshold, speech))
        self._silence_threshold = max(self.min_threshold * 0.5, min(self.max_threshold * 0.5, silence))

        return self._speech_threshold, self._silence_threshold

    @property
    def speech_threshold(self) -> float:
        return self._speech_threshold

    @property
    def silence_threshold(self) -> float:
        return self._silence_threshold

    def is_speech(self, energy: float) -> bool:
        return energy > self._speech_threshold

    def is_silence(self, energy: float) -> bool:
        return energy < self._silence_threshold
