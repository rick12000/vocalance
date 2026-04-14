"""VAD-based audio listeners: PCM from AudioService → segment events on the bus.

``VoiceActivityDetector`` implements the shared VAD algorithm (pre-roll, onset,
accumulate, silence-end, finalize) with per-instance configuration.  Both the
command and sound pipelines are driven by the same class.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import (
    AudioDetectedEvent,
    CommandAudioSegmentReadyEvent,
    ProcessAudioChunkForSoundRecognitionEvent,
)
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.audio_processor import AdaptiveVADThreshold, AudioProcessor

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Threshold and duration parameters for one VAD instance."""

    speech_multiplier: float
    silence_multiplier: float
    min_threshold: float
    max_threshold: float
    silent_chunks_for_end: int
    pre_roll_chunks: int
    min_duration_chunks: int
    max_duration_chunks: int
    # Optional minimum peak energy ratio vs speech threshold (0 = disabled)
    min_peak_ratio: float = 0.0
    # Whether to emit AudioDetectedEvent on first onset
    emit_onset_event: bool = False


class VoiceActivityDetector:
    """Parameterized VAD: consumes PCM, emits one event per detected segment.

    The caller supplies ``on_segment(audio_bytes, sample_rate)`` and optionally
    ``on_onset(timestamp)`` which are called on the asyncio loop via
    ``asyncio.create_task``.  The VAD itself runs synchronously inside the audio
    callback (already on the loop via ``call_soon_threadsafe``).
    """

    def __init__(
        self,
        config: VADConfig,
        audio_processor: AudioProcessor,
        sample_rate: int,
        on_segment: Callable[[bytes, int], "asyncio.Task"],
        on_onset: Optional[Callable[[float], "asyncio.Task"]] = None,
    ) -> None:
        self._config = config
        self._processor = audio_processor
        self._sample_rate = sample_rate
        self._on_segment = on_segment
        self._on_onset = on_onset

        self._threshold = AdaptiveVADThreshold(
            speech_multiplier=config.speech_multiplier,
            silence_multiplier=config.silence_multiplier,
            min_threshold=config.min_threshold,
            max_threshold=config.max_threshold,
        )

        self._pre_roll: list[np.ndarray] = []
        self._buffer: list[np.ndarray] = []
        self._is_recording = False
        self._silent_count = 0
        self._first_onset = True
        self._peak_energy = 0.0
        self._lock = threading.Lock()

    def process_chunk(self, audio_bytes: bytes, timestamp: float, skip: bool = False) -> None:
        """Called on the asyncio thread with each raw PCM chunk."""
        try:
            chunk, energy = self._processor.process_chunk(audio_bytes)
        except Exception as e:
            logger.error("VAD preprocess error: %s", e, exc_info=True)
            return

        with self._lock:
            if skip:
                self._reset()
                return
            try:
                self._process(chunk, energy, timestamp)
            except Exception as e:
                logger.error("VAD error: %s", e, exc_info=True)
                self._reset()

    def _process(self, chunk: np.ndarray, energy: float, timestamp: float) -> None:
        is_likely_speech = energy > self._threshold.speech_threshold
        noise = self._processor.update_noise_floor(energy, is_likely_speech)
        if noise.is_stable:
            self._threshold.update(noise.value)

        if not self._is_recording:
            self._pre_roll.append(chunk)
            if len(self._pre_roll) > self._config.pre_roll_chunks:
                self._pre_roll.pop(0)

            if self._threshold.is_speech(energy):
                self._is_recording = True
                self._peak_energy = energy
                self._buffer.extend(self._pre_roll)
                self._buffer.append(chunk)
                self._silent_count = 0

                if self._config.emit_onset_event and self._first_onset and self._on_onset:
                    asyncio.create_task(self._on_onset(timestamp))
                    self._first_onset = False
        else:
            self._buffer.append(chunk)
            if energy > self._peak_energy:
                self._peak_energy = energy

            if self._threshold.is_silence(energy):
                self._silent_count += 1
                if self._silent_count >= self._config.silent_chunks_for_end:
                    self._finalize()
            else:
                self._silent_count = 0

            if len(self._buffer) >= self._config.max_duration_chunks:
                self._finalize()

    def _finalize(self) -> None:
        if len(self._buffer) < self._config.min_duration_chunks:
            self._reset()
            return

        cfg = self._config
        if cfg.min_peak_ratio > 0 and self._peak_energy < self._threshold.speech_threshold * cfg.min_peak_ratio:
            self._reset()
            return

        audio = np.concatenate(self._buffer)
        audio_bytes = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        duration = len(audio) / self._sample_rate
        logger.debug("VAD segment: %.3fs, %d chunks, %d bytes", duration, len(self._buffer), len(audio_bytes))

        asyncio.create_task(self._on_segment(audio_bytes, self._sample_rate))
        self._reset()

    def _reset(self) -> None:
        self._buffer.clear()
        self._pre_roll.clear()
        self._is_recording = False
        self._silent_count = 0
        self._first_onset = True
        self._peak_energy = 0.0

    def update_silent_chunks_threshold(self, chunks: int) -> None:
        with self._lock:
            self._config.silent_chunks_for_end = chunks

    @property
    def speech_threshold(self) -> float:
        return self._threshold.speech_threshold


# ---------------------------------------------------------------------------
# Concrete listener classes — thin wrappers that produce typed events
# ---------------------------------------------------------------------------


class CommandAudioListener:
    """Command VAD: emits ``AudioDetectedEvent`` on onset, ``CommandAudioSegmentReadyEvent`` on end."""

    def __init__(
        self, event_bus: EventBus, config: GlobalAppConfig, shared_audio_processor: Optional[AudioProcessor] = None
    ) -> None:
        self._event_bus = event_bus
        processor = shared_audio_processor or AudioProcessor(
            sample_rate=config.audio.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )
        vad_cfg = VADConfig(
            speech_multiplier=config.vad.command_adaptive_margin_multiplier,
            silence_multiplier=config.vad.command_adaptive_margin_multiplier * config.vad.silence_threshold_multiplier,
            min_threshold=config.vad.command_energy_threshold,
            max_threshold=0.1,
            silent_chunks_for_end=config.vad.command_silent_chunks_for_end,
            pre_roll_chunks=config.vad.command_pre_roll_buffers,
            min_duration_chunks=int(config.vad.command_min_recording_duration / 0.03),
            max_duration_chunks=int(config.vad.command_max_recording_duration / 0.03),
            emit_onset_event=True,
        )
        self._vad = VoiceActivityDetector(
            config=vad_cfg,
            audio_processor=processor,
            sample_rate=config.audio.sample_rate,
            on_segment=self._emit_segment,
            on_onset=self._emit_onset,
        )

    async def _emit_onset(self, timestamp: float) -> None:
        await self._event_bus.publish(AudioDetectedEvent(timestamp=timestamp))

    async def _emit_segment(self, audio_bytes: bytes, sample_rate: int) -> None:
        await self._event_bus.publish(CommandAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=sample_rate))

    def process_audio_chunk(self, audio_bytes: bytes, timestamp: float) -> None:
        self._vad.process_chunk(audio_bytes, timestamp)

    def update_silent_chunks_threshold(self, chunks: int) -> None:
        self._vad.update_silent_chunks_threshold(chunks)

    def set_main_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        pass

    # Properties for test introspection
    @property
    def pre_roll_chunks(self) -> int:
        return self._vad._config.pre_roll_chunks

    @property
    def silent_chunks_for_end(self) -> int:
        return self._vad._config.silent_chunks_for_end

    @property
    def max_duration_chunks(self) -> int:
        return self._vad._config.max_duration_chunks

    @property
    def _vad_lock(self) -> threading.Lock:
        return self._vad._lock

    @property
    def _is_recording(self) -> bool:
        return self._vad._is_recording


class SoundAudioListener:
    """Sound VAD: emits ``ProcessAudioChunkForSoundRecognitionEvent``; suppressed during dictation."""

    def __init__(
        self, event_bus: EventBus, config: GlobalAppConfig, shared_audio_processor: Optional[AudioProcessor] = None
    ) -> None:
        self._event_bus = event_bus
        self._dictation_active = False
        self._lock = threading.Lock()
        processor = shared_audio_processor or AudioProcessor(
            sample_rate=config.audio.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )
        vad_cfg = VADConfig(
            speech_multiplier=config.vad.sound_adaptive_margin_multiplier,
            silence_multiplier=config.vad.sound_adaptive_margin_multiplier * config.vad.silence_threshold_multiplier,
            min_threshold=config.vad.sound_energy_threshold,
            max_threshold=0.15,
            silent_chunks_for_end=5,
            pre_roll_chunks=5,
            min_duration_chunks=5,
            max_duration_chunks=34,
            min_peak_ratio=1.5,
        )
        self._vad = VoiceActivityDetector(
            config=vad_cfg,
            audio_processor=processor,
            sample_rate=config.audio.sample_rate,
            on_segment=self._emit_segment,
        )

    def setup_subscriptions(self) -> None:
        self._event_bus.subscribe(DictationModeDisableOthersEvent, self._handle_dictation_mode_change)

    async def _emit_segment(self, audio_bytes: bytes, sample_rate: int) -> None:
        await self._event_bus.publish(ProcessAudioChunkForSoundRecognitionEvent(audio_chunk=audio_bytes, sample_rate=sample_rate))

    def process_audio_chunk(self, audio_bytes: bytes, timestamp: float) -> None:
        with self._lock:
            skip = self._dictation_active
        self._vad.process_chunk(audio_bytes, timestamp, skip=skip)

    async def _handle_dictation_mode_change(self, event: DictationModeDisableOthersEvent) -> None:
        with self._lock:
            self._dictation_active = event.dictation_mode_active

    def set_main_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        pass

    # Properties for test introspection
    @property
    def silent_chunks_for_end(self) -> int:
        return self._vad._config.silent_chunks_for_end

    @property
    def max_duration_chunks(self) -> int:
        return self._vad._config.max_duration_chunks

    @property
    def _vad_lock(self) -> threading.Lock:
        return self._vad._lock

    @property
    def _is_recording(self) -> bool:
        return self._vad._is_recording
