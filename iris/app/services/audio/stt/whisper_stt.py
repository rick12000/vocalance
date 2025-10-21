import asyncio
import gc
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel

from iris.app.config.app_config import GlobalAppConfig

logger = logging.getLogger(__name__)


class WhisperSpeechToText:
    """Faster-Whisper STT engine optimized for dictation accuracy with normalization."""

    def __init__(self, model_name: str, device: str, sample_rate: int, config: GlobalAppConfig) -> None:
        self._model_name = model_name
        self._device = device
        self._sample_rate = sample_rate
        self._config = config
        self._model = None
        self._model_lock = asyncio.Lock()

        self._beam_size = config.stt.whisper_beam_size if hasattr(config.stt, "whisper_beam_size") else 5
        self._temperature = config.stt.whisper_temperature if hasattr(config.stt, "whisper_temperature") else 0.0
        self._no_speech_threshold = (
            config.stt.whisper_no_speech_threshold if hasattr(config.stt, "whisper_no_speech_threshold") else 0.6
        )
        self._compute_type = "int8"

        self._max_retries = config.stt.whisper_max_retries
        self._retry_delay_seconds = config.stt.whisper_retry_delay_seconds
        self._download_root = os.path.join(config.storage.user_data_root, "whisper_models")
        os.makedirs(self._download_root, exist_ok=True)
        logger.info(f"Whisper models directory: {self._download_root}")

        self._load_model_with_retry()
        self._warm_up_model()

        logger.info(f"Initialized faster-whisper: {model_name}, device: {device}")

    def _load_model_with_retry(self) -> None:
        """Load Whisper model with retry logic and permanent storage."""
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(f"Loading faster-whisper model: {self._model_name} (attempt {attempt}/{self._max_retries})")
                self._model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                    cpu_threads=4,
                    num_workers=1,
                    download_root=self._download_root,
                )
                logger.info("Faster-whisper model loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load model (attempt {attempt}/{self._max_retries}): {e}", exc_info=True)

                if attempt < self._max_retries:
                    logger.info(f"Retrying in {self._retry_delay_seconds} seconds...")
                    time.sleep(self._retry_delay_seconds)
                else:
                    logger.error(f"Failed to load Whisper model after {self._max_retries} attempts")
                    raise RuntimeError(f"Failed to load Whisper model after {self._max_retries} attempts")

    def _warm_up_model(self) -> None:
        try:
            logger.info("Warming up faster-whisper model...")
            dummy_audio = np.zeros(16000, dtype=np.float32)
            segments, _ = self._model.transcribe(dummy_audio, beam_size=1, vad_filter=False)
            list(segments)
            logger.info("Faster-whisper model warmed up successfully")
        except Exception as e:
            logger.warning(f"Failed to warm up model: {e}")

    def _get_transcription_options(self, audio_duration: float) -> Dict[str, Any]:
        options = {
            "language": "en",
            "beam_size": self._beam_size,
            "temperature": self._temperature,
            "no_speech_threshold": self._no_speech_threshold,
            "condition_on_previous_text": False,
            "word_timestamps": False,
            "vad_filter": False,
        }

        if audio_duration < 5.0:
            options["beam_size"] = max(1, self._beam_size - 1)

        return options

    def _prepare_audio(self, audio_bytes: bytes) -> np.ndarray:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np

    def _extract_text_from_segments(self, segments: List[Any]) -> Tuple[str, float]:
        all_text_parts = []
        total_confidence = 0.0
        segment_count = 0

        for segment in segments:
            segment_text = segment.text.strip()
            if not segment_text:
                continue

            segment_count += 1
            all_text_parts.append(segment_text)

            if hasattr(segment, "avg_logprob") and segment.avg_logprob is not None:
                confidence = min(1.0, max(0.0, (segment.avg_logprob + 1.0) / 1.0))
                total_confidence += confidence
            else:
                total_confidence += 0.8

        combined_text = " ".join(all_text_parts).strip()
        avg_confidence = total_confidence / max(1, segment_count)

        return combined_text, avg_confidence

    def recognize_sync(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        if sample_rate and sample_rate != self._sample_rate:
            logger.warning(f"Sample rate mismatch. Expected {self._sample_rate}, got {sample_rate}")

        if not audio_bytes or not self._model:
            return ""

        duration_sec = len(audio_bytes) / (self._sample_rate * 2)
        if duration_sec < 0.3:
            return ""

        recognition_start = time.time()

        audio_np = self._prepare_audio(audio_bytes)
        options = self._get_transcription_options(duration_sec)
        segments, info = self._model.transcribe(audio_np, **options)
        recognized_text, avg_confidence = self._extract_text_from_segments(segments)

        recognition_time = time.time() - recognition_start

        if not recognized_text:
            return ""

        recognized_text = self._normalize_text(recognized_text)

        if recognized_text:
            logger.info(
                f"Whisper recognized: '{recognized_text}' (confidence: {avg_confidence:.3f}, time: {recognition_time:.3f}s)"
            )

        return recognized_text

    async def recognize(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        async with self._model_lock:
            return await asyncio.to_thread(self.recognize_sync, audio_bytes, sample_rate)

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^(um|uh|like|so)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+(um|uh|like|so)$", "", text, flags=re.IGNORECASE)

        words = text.split()
        if len(words) > 1:
            result = [words[0]]
            for word in words[1:]:
                if word.lower() != result[-1].lower():
                    result.append(word)
            text = " ".join(result)

        return text.strip()

    async def reset_context(self) -> None:
        async with self._model_lock:
            if hasattr(self, "_text_cache"):
                self._text_cache.clear()
            logger.debug("Whisper context and cache reset")

    async def shutdown(self) -> None:
        logger.info("Shutting down WhisperSpeechToText")

        async with self._model_lock:
            if hasattr(self, "_model") and self._model is not None:
                if hasattr(self._model, "unload"):
                    self._model.unload()
                del self._model
                self._model = None
                logger.info("Whisper model deleted")

            if hasattr(self, "_noise_samples") and self._noise_samples is not None:
                self._noise_samples.clear()
                self._noise_samples = None

            if hasattr(self, "_last_result"):
                self._last_result = None

        gc.collect()
        logger.info("WhisperSpeechToText shutdown complete")
