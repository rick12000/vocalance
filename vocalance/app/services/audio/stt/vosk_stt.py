from __future__ import annotations

import asyncio
import gc
import json
from typing import Optional

import vosk

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.lifecycle.worker import run_blocking


class VoskSTT:
    """Offline command recognition via Vosk (Kaldi recognizer, async-wrapped)."""

    def __init__(self, model_path: str, sample_rate: int, config: GlobalAppConfig) -> None:
        self.config = config
        self._sample_rate = sample_rate
        self._model_path = model_path

        self._model = vosk.Model(model_path)
        self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
        self._recognizer_lock = asyncio.Lock()

    def recognize_sync(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        """Run Vosk on one chunk synchronously; returns final text or empty string."""
        if not audio_bytes:
            return ""

        self._recognizer.Reset()
        self._recognizer.AcceptWaveform(audio_bytes)
        result = json.loads(self._recognizer.FinalResult())
        return result.get("text", "")

    async def recognize(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        """Thread-off ``recognize_sync`` behind the internal recognizer lock."""
        async with self._recognizer_lock:
            return await run_blocking(self.recognize_sync, audio_bytes, sample_rate, name="vosk-recognize")

    async def shutdown(self) -> None:
        """Release the Kaldi recognizer and model under lock, then collect."""
        async with self._recognizer_lock:
            if getattr(self, "_recognizer", None) is not None:
                del self._recognizer
                self._recognizer = None
            if getattr(self, "_model", None) is not None:
                del self._model
                self._model = None
        gc.collect()
