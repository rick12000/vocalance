import asyncio
import gc
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel

from vocalance.app.config.app_config import GlobalAppConfig

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Faster-Whisper STT engine optimized for dictation accuracy with normalization.

    Wraps faster-whisper library for high-accuracy transcription with retry logic,
    model download management, and extensive text normalization. Uses dynamic beam
    sizing and advanced transcription options to balance speed and accuracy for
    dictation mode.

    Attributes:
        _model: Loaded WhisperModel instance.
        _model_lock: Asyncio lock ensuring thread-safe recognition.
        _beam_size: Beam search width for decoding.
        _temperature: Sampling temperature for generation.
        _no_speech_threshold: Threshold for no-speech detection.
    """

    def __init__(self, model_name: str, device: str, sample_rate: int, config: GlobalAppConfig) -> None:
        """Initialize Whisper STT engine with model and configuration.

        Downloads model if necessary, performs warm-up inference, and configures
        transcription parameters from global config.

        Args:
            model_name: Whisper model identifier (e.g., "tiny", "base", "small").
            device: Compute device ("cpu" or "cuda").
            sample_rate: Audio sample rate in Hz.
            config: Global application configuration.
        """
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
        
        # Quality thresholds to prevent hallucinations at the source
        # compression_ratio_threshold: Detects repetitive/garbled output (hallucinations have high compression)
        # logprob_threshold: Filters low-confidence predictions (hallucinations have low log probability)
        self._compression_ratio_threshold = (
            config.stt.whisper_compression_ratio_threshold 
            if hasattr(config.stt, "whisper_compression_ratio_threshold") 
            else 2.4  # faster-whisper default, reject segments with compression_ratio > 2.4
        )
        self._logprob_threshold = (
            config.stt.whisper_logprob_threshold
            if hasattr(config.stt, "whisper_logprob_threshold")
            else -1.0  # Reject segments with avg_logprob < -1.0 (low confidence)
        )
        
        self._compute_type = "int8"

        self._max_retries = config.stt.whisper_max_retries
        self._retry_delay_seconds = config.stt.whisper_retry_delay_seconds
        self._download_root = os.path.join(config.storage.user_data_root, "whisper_models")
        os.makedirs(self._download_root, exist_ok=True)
        logger.debug(f"Whisper models directory: {self._download_root}")

        self._load_model_with_retry()
        self._warm_up_model()

        logger.debug(f"Initialized faster-whisper: {model_name}, device: {device}")

    def _load_model_with_retry(self) -> None:
        """Load Whisper model with retry logic and permanent storage.

        Downloads model to user_data_root if not present, retries on failure with
        exponential backoff configured via max_retries and retry_delay_seconds.

        Raises:
            RuntimeError: If model loading fails after all retry attempts.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(f"Loading faster-whisper model: {self._model_name} (attempt {attempt}/{self._max_retries})")
                self._model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                    cpu_threads=4,
                    num_workers=1,
                    download_root=self._download_root,
                )
                logger.debug("Faster-whisper model loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load model (attempt {attempt}/{self._max_retries}): {e}", exc_info=True)

                if attempt < self._max_retries:
                    logger.debug(f"Retrying in {self._retry_delay_seconds} seconds...")
                    time.sleep(self._retry_delay_seconds)
                else:
                    logger.error(f"Failed to load Whisper model after {self._max_retries} attempts")
                    raise RuntimeError(f"Failed to load Whisper model after {self._max_retries} attempts")

    def _warm_up_model(self) -> None:
        """Warm up model with dummy inference to optimize first real transcription.

        Runs a quick transcription on silent audio to load model into memory and
        optimize inference latency for subsequent calls.
        """
        try:
            logger.debug("Warming up faster-whisper model...")
            dummy_audio = np.zeros(16000, dtype=np.float32)
            segments, _ = self._model.transcribe(dummy_audio, beam_size=1, vad_filter=False)
            list(segments)
            logger.debug("Faster-whisper model warmed up successfully")
        except Exception as e:
            logger.warning(f"Failed to warm up model: {e}")

    def _get_transcription_options(self, audio_duration: float) -> Dict[str, Any]:
        """Get transcription options dynamically adjusted for audio duration.

        Args:
            audio_duration: Duration of audio segment in seconds.

        Returns:
            Dictionary of transcription options for faster-whisper.
        """
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
        """Convert raw audio bytes to float32 numpy array normalized to [-1, 1].

        Args:
            audio_bytes: Raw 16-bit PCM audio data.

        Returns:
            Float32 numpy array normalized for Whisper input.
        """
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np

    def _extract_text_from_segments(self, segments: List[Any]) -> Tuple[str, float]:
        """Extract and combine text from Whisper segments with confidence scoring.

        Args:
            segments: List of Whisper segment objects from transcription.

        Returns:
            Tuple of (combined text, average confidence score).
        """
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
        """Synchronous speech recognition with Whisper and text normalization.

        Args:
            audio_bytes: Raw audio data to transcribe.
            sample_rate: Optional sample rate override.

        Returns:
            Normalized recognized text string, or empty string if no speech detected.
        """
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
        """Async speech recognition wrapper with thread-safe execution.

        Args:
            audio_bytes: Raw audio data to transcribe.
            sample_rate: Optional sample rate override.

        Returns:
            Normalized recognized text string.
        """
        async with self._model_lock:
            return await asyncio.to_thread(self.recognize_sync, audio_bytes, sample_rate)

    def recognize_streaming_sync(
        self, audio_bytes: bytes, context_segments: Optional[List[str]] = None, sample_rate: Optional[int] = None
    ) -> Tuple[List[dict], float]:
        """Synchronous streaming speech recognition with segment timestamps.

        Designed for continuous streaming transcription where predictions are made
        on overlapping audio chunks. Returns segment list with timestamps for
        accurate offset tracking (WhisperLive approach).

        Args:
            audio_bytes: Raw audio data to transcribe.
            context_segments: List of recent transcription texts for context (last 5-10 segments).
            sample_rate: Optional sample rate override.

        Returns:
            Tuple of (segments_list, confidence_score) where segments_list contains:
            [{"text": str, "start": float, "end": float, "completed": bool}, ...]
        """
        if sample_rate and sample_rate != self._sample_rate:
            logger.warning(f"Sample rate mismatch. Expected {self._sample_rate}, got {sample_rate}")

        if not audio_bytes or not self._model:
            return [], 0.0

        duration_sec = len(audio_bytes) / (self._sample_rate * 2)
        if duration_sec < 0.3:
            return [], 0.0

        recognition_start = time.time()

        # Build initial_prompt from context segments
        initial_prompt = None
        if context_segments and len(context_segments) > 0:
            # Use last 3-5 segments for context, limited to ~200 chars to avoid token limits
            recent_context = " ".join(context_segments[-5:])
            if len(recent_context) > 200:
                recent_context = recent_context[-200:]
            initial_prompt = recent_context.strip()

        audio_np = self._prepare_audio(audio_bytes)
        options = self._get_transcription_options(duration_sec)

        # Add initial_prompt for context
        if initial_prompt:
            options["initial_prompt"] = initial_prompt
            options["condition_on_previous_text"] = True

        segments_iter, info = self._model.transcribe(audio_np, **options)

        # Convert segments iterator to list and extract segment info with timestamps
        segment_list = []
        confidence_scores = []

        for seg in segments_iter:
            # Filter out segments with high no_speech probability (silence/noise)
            # WhisperLive uses 0.45 threshold - segments above this are likely silence
            no_speech_prob = seg.no_speech_prob if hasattr(seg, "no_speech_prob") else 0.0
            if no_speech_prob > self._no_speech_threshold:
                logger.debug(f"Skipping segment with high no_speech_prob: {no_speech_prob:.3f}")
                continue

            # CRITICAL: Filter hallucinations at source using Whisper's quality metrics
            # Check avg_logprob - hallucinations typically have very low log probability
            avg_logprob = seg.avg_logprob if hasattr(seg, "avg_logprob") else 0.0
            if avg_logprob < self._logprob_threshold:
                logger.debug(f"Skipping low-confidence segment: avg_logprob={avg_logprob:.3f} < {self._logprob_threshold}")
                continue

            # Check compression_ratio - hallucinations often have high compression ratio
            # (the model is repeating itself or outputting gibberish)
            compression_ratio = seg.compression_ratio if hasattr(seg, "compression_ratio") else 0.0
            if compression_ratio > self._compression_ratio_threshold:
                logger.debug(f"Skipping repetitive segment: compression_ratio={compression_ratio:.2f} > {self._compression_ratio_threshold}")
                continue

            text = seg.text.strip()
            if not text:
                continue

            # Apply light normalization
            text = self._normalize_text_streaming(text)
            if not text:
                continue

            segment_list.append(
                {
                    "text": text,
                    "start": seg.start,
                    "end": seg.end,
                    "completed": False,  # Will mark all but last as completed
                    "no_speech_prob": no_speech_prob,
                    "avg_logprob": avg_logprob,
                    "compression_ratio": compression_ratio,
                }
            )

            # Track confidence (use inverse of no_speech_prob as proxy)
            confidence_scores.append(1.0 - no_speech_prob)

        recognition_time = time.time() - recognition_start

        if not segment_list:
            return [], 0.0

        # Mark all segments except the last as completed
        # Last segment might be incomplete (word cutoff at end of audio)
        for seg in segment_list[:-1]:
            seg["completed"] = True

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        total_text = " ".join([seg["text"] for seg in segment_list])
        logger.debug(
            f"Whisper streaming: {len(segment_list)} segments, '{total_text[:50]}...' "
            f"(conf: {avg_confidence:.3f}, time: {recognition_time:.3f}s)"
        )

        return segment_list, avg_confidence

    async def recognize_streaming(
        self, audio_bytes: bytes, context_segments: Optional[List[str]] = None, sample_rate: Optional[int] = None
    ) -> Tuple[List[dict], float]:
        """Async streaming speech recognition with segment timestamps.

        Args:
            audio_bytes: Raw audio data to transcribe.
            context_segments: List of recent transcription texts for context.
            sample_rate: Optional sample rate override.

        Returns:
            Tuple of (segments_list, confidence_score) where segments_list contains:
            [{"text": str, "start": float, "end": float, "completed": bool}, ...]
        """
        async with self._model_lock:
            return await asyncio.to_thread(self.recognize_streaming_sync, audio_bytes, context_segments, sample_rate)

    def _normalize_text_streaming(self, text: str) -> str:
        """Lighter normalization for streaming mode to preserve detail.

        Unlike regular normalization, this preserves more filler words and
        only removes egregious duplicates, since partial predictions benefit
        from seeing the model's full output.

        Args:
            text: Raw transcribed text.

        Returns:
            Lightly normalized text string.
        """
        if not text:
            return ""

        text = text.strip()
        if not text:
            return ""

        # Only collapse excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove consecutive duplicate words (but keep single occurrences)
        words = text.split()
        if len(words) > 1:
            result = [words[0]]
            for word in words[1:]:
                if word.lower() != result[-1].lower():
                    result.append(word)
            text = " ".join(result)

        return text.strip()

    def _normalize_text(self, text: str) -> str:
        """Normalize transcribed text by removing filler words and duplicates.

        Removes leading/trailing fillers (um, uh, like, so), collapses whitespace,
        and removes consecutive duplicate words for cleaner dictation output.

        Args:
            text: Raw transcribed text.

        Returns:
            Normalized text string.
        """
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
        """Reset Whisper context and text cache for fresh transcription state."""
        async with self._model_lock:
            if hasattr(self, "_text_cache"):
                self._text_cache.clear()
            logger.debug("Whisper context and cache reset")

    async def shutdown(self) -> None:
        """Shutdown Whisper engine and release all model resources.

        Unloads model if supported, deletes references to noise samples and cached
        results, and runs garbage collection to free memory immediately.
        """
        logger.info("Shutting down WhisperSTT")

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
        logger.info("WhisperSTT shutdown complete")
