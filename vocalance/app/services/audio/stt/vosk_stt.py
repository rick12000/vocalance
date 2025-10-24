import asyncio
import gc
import json
import logging
from typing import Optional

import vosk

from vocalance.app.config.app_config import GlobalAppConfig

logger = logging.getLogger(__name__)


class EnhancedVoskSTT:
    """Vosk STT engine optimized for fast command recognition."""

    def __init__(self, model_path: str, sample_rate: int, config: GlobalAppConfig) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self._sample_rate = sample_rate
        self._model_path = model_path

        self._model = vosk.Model(model_path)
        self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
        self._recognizer_lock = asyncio.Lock()

        self.logger.info(f"VoskSTT initialized: model={model_path}, sample_rate={sample_rate}")

    def recognize_sync(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        if not audio_bytes:
            return ""

        self._recognizer.Reset()

        self._recognizer.AcceptWaveform(audio_bytes)
        result = json.loads(self._recognizer.FinalResult())
        recognized_text = result.get("text", "")

        return recognized_text

    async def recognize(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        async with self._recognizer_lock:
            return await asyncio.to_thread(self.recognize_sync, audio_bytes, sample_rate)

    async def shutdown(self) -> None:
        logger.info("Shutting down EnhancedVoskSTT")

        async with self._recognizer_lock:
            if hasattr(self, "_recognizer") and self._recognizer is not None:
                del self._recognizer
                self._recognizer = None
                logger.info("Vosk recognizer deleted")

            if hasattr(self, "_model") and self._model is not None:
                del self._model
                self._model = None
                logger.info("Vosk model deleted")

        gc.collect()
        logger.info("EnhancedVoskSTT shutdown complete")
