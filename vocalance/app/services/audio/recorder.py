import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from vocalance.app.config.app_config import GlobalAppConfig


class AudioRecorder:
    def __init__(
        self,
        app_config: GlobalAppConfig,
        mode: str = "command",
        on_audio_segment: Optional[Callable[[bytes], None]] = None,
        on_audio_detected: Optional[Callable[[], None]] = None,
    ) -> None:
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{mode}")
        self.app_config = app_config
        self.mode = mode
        self.on_audio_segment = on_audio_segment
        self.on_audio_detected = on_audio_detected

        if mode == "command":
            self.chunk_size = app_config.audio.command_chunk_size
            self.energy_threshold = app_config.vad.command_energy_threshold
            self.silent_chunks_for_end = app_config.vad.command_silent_chunks_for_end
            self.max_duration = app_config.vad.command_max_recording_duration
            self.pre_roll_chunks = app_config.vad.command_pre_roll_buffers
            self.min_duration = app_config.vad.command_min_recording_duration
            self.adaptive_margin_multiplier = app_config.vad.command_adaptive_margin_multiplier
        else:
            self.chunk_size = app_config.audio.chunk_size
            self.energy_threshold = app_config.vad.dictation_energy_threshold
            self.silent_chunks_for_end = app_config.vad.dictation_silent_chunks_for_end
            self.max_duration = app_config.vad.dictation_max_recording_duration
            self.pre_roll_chunks = app_config.vad.dictation_pre_roll_buffers
            self.min_duration = app_config.vad.dictation_min_recording_duration
            self.adaptive_margin_multiplier = app_config.vad.dictation_adaptive_margin_multiplier

        self.sample_rate = app_config.audio.sample_rate
        self.device = getattr(app_config.audio, "device", None)
        self.silence_threshold = self.energy_threshold * app_config.vad.silence_threshold_multiplier
        self._is_recording: bool = False
        self._is_active: bool = True
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._noise_floor: float = app_config.vad.noise_floor_initial_value
        self._noise_samples: list[float] = []
        self._max_noise_samples: int = app_config.vad.max_noise_samples

        self.logger.debug(
            f"AudioRecorder initialized for {mode} mode: chunk_size={self.chunk_size}samples, "
            f"silent_chunks_for_end={self.silent_chunks_for_end}"
        )

    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        if audio_chunk.dtype == np.int16:
            return np.sqrt(np.mean((audio_chunk.astype(np.float32) / 32768.0) ** 2))
        return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

    def _update_noise_floor(self, energy: float) -> None:
        if len(self._noise_samples) < self._max_noise_samples:
            self._noise_samples.append(energy)

            if len(self._noise_samples) == self._max_noise_samples:
                self._noise_floor = np.percentile(self._noise_samples, self.app_config.vad.noise_floor_percentile)
                adaptive_threshold = self._noise_floor * self.adaptive_margin_multiplier

                if adaptive_threshold > self.energy_threshold * self.app_config.vad.adaptive_threshold_max_multiplier:
                    old_threshold = self.energy_threshold
                    self.energy_threshold = adaptive_threshold
                    self.silence_threshold = self.energy_threshold * self.app_config.vad.adaptive_silence_threshold_multiplier
                    self.logger.debug(f"Adapted thresholds: {old_threshold:.6f} -> {self.energy_threshold:.6f}")

    def _recording_thread(self) -> None:
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, blocksize=self.chunk_size, channels=1, dtype="int16", device=self.device
            )
            self._stream.start()
            self.logger.debug(f"{self.mode} recording started")

            while True:
                with self._lock:
                    if not self._is_recording:
                        break
                    is_active = self._is_active

                if not is_active:
                    time.sleep(0.1)
                    continue

                audio_buffer = []
                pre_roll_buffer = []
                speech_detected = False

                while True:
                    with self._lock:
                        if not self._is_recording or not self._is_active:
                            break

                    if speech_detected:
                        break

                    try:
                        data, _ = self._stream.read(self.chunk_size)
                        energy = self._calculate_energy(data)

                        pre_roll_buffer.append(data)
                        if len(pre_roll_buffer) > self.pre_roll_chunks:
                            pre_roll_buffer.pop(0)

                        if energy <= self.energy_threshold and len(self._noise_samples) < self._max_noise_samples:
                            self._update_noise_floor(energy)

                        if energy > self.energy_threshold:
                            speech_detected = True
                            audio_buffer.extend(pre_roll_buffer)

                            if self.mode == "command" and self.on_audio_detected:
                                self.on_audio_detected()

                    except (OSError, RuntimeError) as e:
                        self.logger.error(f"Audio device error: {e}")
                        return
                    except Exception as e:
                        self.logger.exception(f"Unexpected error reading audio: {e}")
                        raise

                if not speech_detected:
                    continue

                silent_chunks_count = 0
                recording_start = time.time()
                chunks_collected = 0

                while True:
                    with self._lock:
                        if not self._is_recording or not self._is_active:
                            break

                    try:
                        data, _ = self._stream.read(self.chunk_size)
                        energy = self._calculate_energy(data)
                        audio_buffer.append(data)
                        chunks_collected += 1

                        if energy < self.silence_threshold:
                            silent_chunks_count += 1
                            if silent_chunks_count >= self.silent_chunks_for_end:
                                self.logger.debug(f"Silence detected: {silent_chunks_count} consecutive silent chunks")
                                break
                        else:
                            silent_chunks_count = 0

                        if time.time() - recording_start > self.max_duration:
                            self.logger.debug(f"Max duration reached: {self.max_duration}s")
                            break

                    except (OSError, RuntimeError) as e:
                        self.logger.error(f"Audio device error during recording: {e}")
                        return
                    except Exception as e:
                        self.logger.exception(f"Unexpected error during recording: {e}")
                        raise

                if audio_buffer and self.on_audio_segment:
                    audio_data = np.concatenate(audio_buffer)
                    duration = len(audio_data) / self.sample_rate

                    if duration >= self.min_duration:
                        audio_bytes = audio_data.tobytes()
                        self.logger.info(f"Segment captured: {duration:.3f}s, {chunks_collected} chunks")
                        self.on_audio_segment(audio_bytes)

        except Exception as e:
            self.logger.error(f"Recording thread error: {e}", exc_info=True)
        finally:
            self._cleanup_stream()

    def _cleanup_stream(self) -> None:
        if self._stream:
            try:
                if hasattr(self._stream, "active") and self._stream.active:
                    self._stream.stop()
                    self.logger.debug(f"{self.mode} stream stopped")

                self._stream.close()
                self.logger.debug(f"{self.mode} stream closed")

            except Exception as e:
                self.logger.error(f"Error cleaning up {self.mode} audio stream: {e}", exc_info=True)
            finally:
                self._stream = None
                self.logger.info(f"{self.mode} audio stream cleanup completed")

    def start(self) -> None:
        with self._lock:
            if self._is_recording:
                return
            self._is_recording = True
            self._thread = threading.Thread(target=self._recording_thread, daemon=False)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._is_recording:
                return
            self._is_recording = False

        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                self.logger.error(f"{self.mode} recording thread did not terminate after 5s timeout")

        self._cleanup_stream()

    def set_active(self, active: bool) -> None:
        with self._lock:
            self._is_active = active

    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    def is_active(self) -> bool:
        with self._lock:
            return self._is_active
