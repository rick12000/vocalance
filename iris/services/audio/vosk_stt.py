"""
Enhanced Vosk STT Service

Provides optimized Vosk STT functionality for command recognition.
"""

import logging
import json
import time
import threading
from typing import Optional
import vosk

from iris.config.app_config import GlobalAppConfig
from iris.services.audio.stt_utils import DuplicateTextFilter

logger = logging.getLogger(__name__)

class EnhancedVoskSTT:
    """Enhanced Vosk STT with duplicate filtering"""
    
    def __init__(self, model_path: str, sample_rate: int, config: GlobalAppConfig):
        """Initialize Vosk STT"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Core settings
        self._sample_rate = sample_rate
        self._model_path = model_path
        
        # Initialize Vosk components
        try:
            self._model = vosk.Model(model_path)
            self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
            self._recognizer_lock = threading.Lock()
            
            self.logger.info(f"VoskSTT initialized: model={model_path}, sample_rate={sample_rate}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vosk: {e}")
            raise
        
        # Lightweight duplicate filtering
        self._duplicate_filter = DuplicateTextFilter(cache_size=5, duplicate_threshold_ms=300)
        
    def recognize(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        """
        Recognize speech using Vosk
        
        Args:
            audio_bytes: Raw audio bytes to process
            sample_rate: Sample rate of the audio
            
        Returns:
            Recognized text string
        """
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
    
    def recognize_streaming(self, audio_bytes: bytes, is_final: bool = False) -> Optional[str]:
        """
        Streaming recognition for real-time partial results
        
        Args:
            audio_bytes: Raw audio bytes to process
            is_final: Whether this is the final audio chunk
            
        Returns:
            Partial or final recognized text
        """
        if not audio_bytes:
            return None
        
        with self._recognizer_lock:
            try:
                if self._recognizer.AcceptWaveform(audio_bytes):
                    # Final result available
                    result = json.loads(self._recognizer.Result())
                    recognized_text = result.get("text", "")
                else:
                    # Partial result
                    partial_result = json.loads(self._recognizer.PartialResult())
                    recognized_text = partial_result.get("partial", "")
                
                if recognized_text:
                    normalized_text = recognized_text.lower().strip()
                    if not self._is_nonsense(normalized_text):
                        return normalized_text
                
                return None
                
            except Exception as e:
                logger.error(f"Streaming recognition error: {e}")
                return None 