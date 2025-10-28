import asyncio
import gc
import json
import logging
from typing import Optional

import vosk

from vocalance.app.config.app_config import GlobalAppConfig

logger = logging.getLogger(__name__)


class VoskSTT:
    """Vosk STT engine optimized for fast, offline command recognition.

    Wraps the Vosk speech recognition library with async support and thread-safe
    recognition. Uses Kaldi recognizer for real-time speech-to-text with minimal
    latency, ideal for command mode where speed is critical.

    Attributes:
        _model: Loaded Vosk language model.
        _recognizer: Kaldi recognizer instance for speech processing.
        _recognizer_lock: Asyncio lock ensuring thread-safe recognition.
    """

    def __init__(self, model_path: str, sample_rate: int, config: GlobalAppConfig) -> None:
        """Initialize Vosk STT engine with model and configuration.

        Args:
            model_path: Path to Vosk model directory.
            sample_rate: Audio sample rate in Hz.
            config: Global application configuration.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self._sample_rate = sample_rate
        self._model_path = model_path

        self._model = vosk.Model(model_path)
        self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
        self._recognizer_lock = asyncio.Lock()

        self.logger.debug(f"VoskSTT initialized: model={model_path}, sample_rate={sample_rate}")

    def recognize_sync(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        """Synchronous speech recognition using Vosk.

        Args:
            audio_bytes: Raw audio data to transcribe.
            sample_rate: Optional sample rate (uses configured rate if None).

        Returns:
            Recognized text string, or empty string if no speech detected.
        """
        if not audio_bytes:
            return ""

        self._recognizer.Reset()

        self._recognizer.AcceptWaveform(audio_bytes)
        result = json.loads(self._recognizer.FinalResult())
        recognized_text = result.get("text", "")

        return recognized_text

    async def recognize(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        """Async speech recognition wrapper with thread-safe execution.

        Args:
            audio_bytes: Raw audio data to transcribe.
            sample_rate: Optional sample rate override.

        Returns:
            Recognized text string.
        """
        async with self._recognizer_lock:
            return await asyncio.to_thread(self.recognize_sync, audio_bytes, sample_rate)

    async def shutdown(self) -> None:
        """Shutdown Vosk engine and release model resources.

        Deletes recognizer and model references, then runs garbage collection
        to free memory immediately.
        """
        logger.debug("Shutting down VoskSTT")

        async with self._recognizer_lock:
            if hasattr(self, "_recognizer") and self._recognizer is not None:
                del self._recognizer
                self._recognizer = None
                logger.debug("Vosk recognizer deleted")

            if hasattr(self, "_model") and self._model is not None:
                del self._model
                self._model = None
                logger.debug("Vosk model deleted")

        gc.collect()
        logger.debug("VoskSTT shutdown complete")
