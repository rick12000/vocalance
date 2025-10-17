import gc
import json
import logging
import threading
from typing import Optional

import vosk

from iris.app.config.app_config import GlobalAppConfig
from iris.app.services.audio.stt.stt_utils import DuplicateTextFilter

logger = logging.getLogger(__name__)


class EnhancedVoskSTT:
    """Vosk STT engine optimized for fast command recognition with duplicate filtering."""

    def __init__(self, model_path: str, sample_rate: int, config: GlobalAppConfig) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self._sample_rate = sample_rate
        self._model_path = model_path

        try:
            self._model = vosk.Model(model_path)
            self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
            self._recognizer_lock = threading.Lock()

            self.logger.info(f"VoskSTT initialized: model={model_path}, sample_rate={sample_rate}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vosk: {e}")
            raise

        self._duplicate_filter = DuplicateTextFilter(cache_size=5, duplicate_threshold_ms=300)

    def recognize(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        if not audio_bytes:
            return ""

        with self._recognizer_lock:
            self._recognizer.Reset()

            try:
                self._recognizer.AcceptWaveform(audio_bytes)
                result = json.loads(self._recognizer.FinalResult())
                recognized_text = result.get("text", "")

                if not recognized_text:
                    return ""

                # Check for duplicates
                if self._duplicate_filter.is_duplicate(recognized_text):
                    return ""

                return recognized_text

            except Exception as e:
                logger.error(f"Recognition error: {e}")
                return ""

    async def shutdown(self) -> None:
        try:
            logger.info("Shutting down EnhancedVoskSTT")

            with self._recognizer_lock:
                # Clear Vosk recognizer first (holds references to model)
                if hasattr(self, "_recognizer") and self._recognizer is not None:
                    # Reset recognizer state before deletion
                    try:
                        self._recognizer.Reset()
                    except Exception:
                        pass
                    del self._recognizer
                    self._recognizer = None
                    logger.info("Vosk recognizer deleted")

                # Clear Vosk model (releases memory)
                if hasattr(self, "_model") and self._model is not None:
                    del self._model
                    self._model = None
                    logger.info("Vosk model deleted")

                # Clear duplicate filter
                if hasattr(self, "_duplicate_filter") and self._duplicate_filter is not None:
                    del self._duplicate_filter
                    self._duplicate_filter = None

            # Force garbage collection for Vosk C++ objects
            gc.collect()

            logger.info("EnhancedVoskSTT shutdown complete")

        except Exception as e:
            logger.error(f"Error during EnhancedVoskSTT shutdown: {e}", exc_info=True)
