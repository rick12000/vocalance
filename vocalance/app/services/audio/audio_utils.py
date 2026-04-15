from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Union

import numpy as np
from pydantic import BaseModel, ConfigDict


@dataclass
class NoiseFloorEstimate:
    value: float
    sample_count: int
    is_stable: bool


class AudioProcessor:
    """DC removal, optional peak normalization, RMS energy, and rolling noise-floor estimate."""

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

        self.energy_history: deque[float] = deque(maxlen=self.NOISE_FLOOR_WINDOW_SIZE)
        self.noise_floor = self.NOISE_FLOOR_INITIAL
        self.noise_floor_stable = False
        self.bootstrap_count = 0

        self.recent_peaks: deque[float] = deque(maxlen=50)
        self.calibrated_gain = 1.0

    def process_chunk(self, audio_bytes: bytes) -> tuple[np.ndarray, float]:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = self.remove_dc_offset(audio)
        if self.enable_normalization:
            audio = self.normalize_peak(audio)
        energy = self.calculate_rms_energy(audio)
        return audio, energy

    def update_noise_floor(self, energy: float, is_likely_speech: bool) -> NoiseFloorEstimate:
        self.bootstrap_count += 1

        if self.bootstrap_count <= self.BOOTSTRAP_CHUNKS:
            self.energy_history.append(energy)
        elif not is_likely_speech:
            self.energy_history.append(energy)

        if len(self.energy_history) >= self.MIN_SAMPLES_FOR_STABLE:
            self.noise_floor = float(np.percentile(list(self.energy_history), self.NOISE_FLOOR_PERCENTILE))
            self.noise_floor_stable = True
        elif len(self.energy_history) > 0:
            self.noise_floor = float(np.percentile(list(self.energy_history), self.NOISE_FLOOR_PERCENTILE))
            self.noise_floor_stable = False

        return NoiseFloorEstimate(
            value=self.noise_floor,
            sample_count=len(self.energy_history),
            is_stable=self.noise_floor_stable,
        )

    def get_noise_floor(self) -> NoiseFloorEstimate:
        return NoiseFloorEstimate(
            value=self.noise_floor,
            sample_count=len(self.energy_history),
            is_stable=self.noise_floor_stable,
        )

    def get_adaptive_threshold(self, base_multiplier: float = 3.0) -> float:
        return self.noise_floor * base_multiplier

    def is_above_noise_floor(self, energy: float, multiplier: float = 2.0) -> bool:
        return energy > self.noise_floor * multiplier

    def reset(self) -> None:
        self.energy_history.clear()
        self.noise_floor = self.NOISE_FLOOR_INITIAL
        self.noise_floor_stable = False
        self.bootstrap_count = 0
        self.recent_peaks.clear()
        self.calibrated_gain = 1.0

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        return audio - np.mean(audio)

    def normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))

        if peak > self.MIN_PEAK_FOR_NORMALIZATION:
            self.recent_peaks.append(float(peak))

        if len(self.recent_peaks) > 0:
            reference_peak = float(np.percentile(list(self.recent_peaks), 90))
            if reference_peak > self.MIN_PEAK_FOR_NORMALIZATION:
                target_gain = self.target_peak / reference_peak
                if target_gain < self.calibrated_gain:
                    self.calibrated_gain = 0.7 * self.calibrated_gain + 0.3 * target_gain
                else:
                    self.calibrated_gain = 0.98 * self.calibrated_gain + 0.02 * target_gain
                self.calibrated_gain = max(0.5, min(5.0, self.calibrated_gain))

        return audio * self.calibrated_gain

    def calculate_rms_energy(self, audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


class AdaptiveVADThreshold:
    """Derives speech and silence energy thresholds from the noise floor."""

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

        self.speech_threshold_value = min_threshold * speech_multiplier
        self.silence_threshold_value = min_threshold * silence_multiplier

    def update(self, noise_floor: float) -> tuple[float, float]:
        speech = noise_floor * self.speech_multiplier
        silence = noise_floor * self.silence_multiplier

        self.speech_threshold_value = max(self.min_threshold, min(self.max_threshold, speech))
        self.silence_threshold_value = max(self.min_threshold * 0.5, min(self.max_threshold * 0.5, silence))

        return self.speech_threshold_value, self.silence_threshold_value

    @property
    def speech_threshold(self) -> float:
        return self.speech_threshold_value

    @property
    def silence_threshold(self) -> float:
        return self.silence_threshold_value

    def is_speech(self, energy: float) -> bool:
        return energy > self.speech_threshold_value

    def is_silence(self, energy: float) -> bool:
        return energy < self.silence_threshold_value


class Onset(BaseModel):
    model_config = ConfigDict(frozen=True)
    ts: float


class Clip(BaseModel):
    model_config = ConfigDict(frozen=True)
    pcm_bytes: bytes
    sample_rate: int


SegmentHit = Union[Onset, Clip]


@dataclass
class SegmentConfig:
    speech_multiplier: float
    silence_multiplier: float
    min_threshold: float
    max_threshold: float
    silent_chunks_for_end: int
    pre_roll_chunks: int
    min_duration_chunks: int
    max_duration_chunks: int
    min_peak_ratio: float = 0.0
    emit_onset: bool = False


class UtteranceSegmenter:
    """Stateful VAD over mono PCM chunks; yields ``Onset`` or ``Clip`` hits."""

    def __init__(self, segment_config: SegmentConfig, analyzer: AudioProcessor, sample_rate: int) -> None:
        self.config = segment_config
        self.analyzer = analyzer
        self.sample_rate = sample_rate
        self.energy_gate = AdaptiveVADThreshold(
            speech_multiplier=segment_config.speech_multiplier,
            silence_multiplier=segment_config.silence_multiplier,
            min_threshold=segment_config.min_threshold,
            max_threshold=segment_config.max_threshold,
        )
        self.pre_roll: list[np.ndarray] = []
        self.segment_buffer: list[np.ndarray] = []
        self.capturing = False
        self.silence_streak = 0
        self.onset_pending = True
        self.peak_energy = 0.0
        self.state_lock = threading.Lock()

    def feed_pcm_chunk(self, pcm_bytes: bytes, ts: float, skip_scoring: bool = False) -> list[SegmentHit]:
        float_chunk, rms_energy = self.analyzer.process_chunk(pcm_bytes)
        with self.state_lock:
            if skip_scoring:
                self.reset_state()
                return []
            return self.advance_from_chunk(float_chunk, rms_energy, ts)

    def advance_from_chunk(self, float_chunk: np.ndarray, rms_energy: float, ts: float) -> list[SegmentHit]:
        hits: list[SegmentHit] = []
        likely_speech = rms_energy > self.energy_gate.speech_threshold
        noise = self.analyzer.update_noise_floor(rms_energy, likely_speech)
        if noise.is_stable:
            self.energy_gate.update(noise.value)

        if not self.capturing:
            self.pre_roll.append(float_chunk)
            if len(self.pre_roll) > self.config.pre_roll_chunks:
                self.pre_roll.pop(0)
            if self.energy_gate.is_speech(rms_energy):
                self.capturing = True
                self.peak_energy = rms_energy
                self.segment_buffer.extend(self.pre_roll)
                self.segment_buffer.append(float_chunk)
                self.silence_streak = 0
                if self.config.emit_onset and self.onset_pending:
                    hits.append(Onset(ts=ts))
                    self.onset_pending = False
        else:
            self.segment_buffer.append(float_chunk)
            if rms_energy > self.peak_energy:
                self.peak_energy = rms_energy
            if self.energy_gate.is_silence(rms_energy):
                self.silence_streak += 1
                if self.silence_streak >= self.config.silent_chunks_for_end:
                    hits.extend(self.finalize_clip_if_ready())
            else:
                self.silence_streak = 0
            if len(self.segment_buffer) >= self.config.max_duration_chunks:
                hits.extend(self.finalize_clip_if_ready())
        return hits

    def finalize_clip_if_ready(self) -> list[Clip]:
        if len(self.segment_buffer) < self.config.min_duration_chunks:
            self.reset_state()
            return []
        limits = self.config
        if limits.min_peak_ratio > 0 and self.peak_energy < self.energy_gate.speech_threshold * limits.min_peak_ratio:
            self.reset_state()
            return []
        raw = np.concatenate(self.segment_buffer)
        pcm_bytes = (np.clip(raw, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        self.reset_state()
        return [Clip(pcm_bytes=pcm_bytes, sample_rate=self.sample_rate)]

    def reset_state(self) -> None:
        self.segment_buffer.clear()
        self.pre_roll.clear()
        self.capturing = False
        self.silence_streak = 0
        self.onset_pending = True
        self.peak_energy = 0.0

    def set_silence_tail(self, chunks: int) -> None:
        with self.state_lock:
            self.config.silent_chunks_for_end = chunks
