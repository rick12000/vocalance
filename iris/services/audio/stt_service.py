"""
Streamlined Speech-to-Text Service

Provides efficient STT processing with mode-aware handling for command and dictation modes.
"""

import logging
import asyncio
import time
import threading
from typing import Optional, Dict, Any
from enum import Enum

from iris.event_bus import EventBus
from iris.config.app_config import GlobalAppConfig

from iris.services.audio.vosk_stt import EnhancedVoskSTT
from iris.services.audio.whisper_stt import WhisperSpeechToText
from iris.services.audio.stt_utils import DuplicateTextFilter
from iris.events.stt_events import CommandTextRecognizedEvent, DictationTextRecognizedEvent, STTProcessingStartedEvent, STTProcessingCompletedEvent
from iris.events.dictation_events import DictationModeDisableOthersEvent
from iris.events.core_events import ProcessAudioChunkForSoundRecognitionEvent, CommandAudioSegmentReadyEvent, DictationAudioSegmentReadyEvent

logger = logging.getLogger(__name__)

class STTMode(Enum):
    """STT processing modes"""
    COMMAND = "command"
    DICTATION = "dictation"

class StreamlinedSpeechToTextService:
    """Streamlined STT service with mode-aware processing"""
    
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig):
        self.event_bus = event_bus
        self.config = config
        self.stt_config = config.stt
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Current processing mode
        self._dictation_active = False
        self._processing_lock = threading.RLock()
        
        # STT engines
        self.vosk_engine = None
        self.whisper_engine = None
        self._engines_initialized = False
        
        # Duplicate detection
        self._duplicate_filter = DuplicateTextFilter(cache_size=5, duplicate_threshold_ms=1000)
        
        # Smart timeout management
        self._smart_timeout_manager = None
        
        # Amber trigger words
        self._amber_words = {"amber", "stop", "end"}
        
        logger.info(f"StreamlinedSpeechToTextService initialized - initial dictation_active: {self._dictation_active}")

    def initialize_engines(self):
        """Initialize STT engines at startup"""
        if self._engines_initialized:
            return
            
        try:
            logger.info("🚀 Initializing STT engines...")
            
            # Initialize Vosk engine
            logger.info("📖 Loading Vosk STT engine...")
            self.vosk_engine = EnhancedVoskSTT(
                model_path=self.config.model_paths.vosk_model,
                sample_rate=self.stt_config.sample_rate,
                config=self.config
            )
            
            # Initialize Whisper engine
            logger.info("🎤 Loading Whisper STT engine...")
            self.whisper_engine = WhisperSpeechToText(
                model_name=self.stt_config.whisper_model,
                device=self.stt_config.whisper_device,
                sample_rate=self.stt_config.sample_rate,
                config=self.stt_config
            )
            
            self._engines_initialized = True
            logger.info("✅ All STT engines initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize STT engines: {e}", exc_info=True)
            raise

    def setup_subscriptions(self):
        """Setup event subscriptions"""
        self.event_bus.subscribe(CommandAudioSegmentReadyEvent, self._handle_command_audio_segment)
        self.event_bus.subscribe(DictationAudioSegmentReadyEvent, self._handle_dictation_audio_segment)
        self.event_bus.subscribe(DictationModeDisableOthersEvent, self._handle_dictation_mode_change)
        
        logger.info("STT service event subscriptions configured")

    async def _publish_recognition_result(self, text: str, processing_time: float, engine: str, mode: STTMode):
        """Publish recognition results using mode-specific events"""
        
        if mode == STTMode.DICTATION:
            event = DictationTextRecognizedEvent(
                text=text,
                processing_time_ms=processing_time,
                engine=engine,
                mode=mode.value
            )
        else:
            event = CommandTextRecognizedEvent(
                text=text,
                processing_time_ms=processing_time,
                engine=engine,
                mode=mode.value
            )
        await self.event_bus.publish(event)
        logger.info(f"Published {type(event).__name__}: '{text}' from {engine}")

        # Update duplicate tracking
        self._last_recognized_text = text
        self._last_text_time = time.time()
        
    async def _publish_sound_recognition_event(self, audio_bytes: bytes, sample_rate: int):
        """Publish audio chunk for sound recognition"""
        try:
            sound_event = ProcessAudioChunkForSoundRecognitionEvent(
                audio_chunk=audio_bytes,
                sample_rate=sample_rate
            )
            await self.event_bus.publish(sound_event)
            logger.debug(f"Published sound recognition event for {len(audio_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error publishing sound recognition event: {e}")
    
    async def _handle_command_audio_segment(self, event_data: CommandAudioSegmentReadyEvent):
        """Process command audio"""
        try:
            if not self._engines_initialized:
                logger.error("STT engines not initialized")
                return

            # Debug logging for dictation mode state
            logger.debug(f"Processing command audio segment - dictation_active: {self._dictation_active}")

            # If in dictation mode, only check for amber trigger words
            if self._dictation_active:
                logger.debug("In dictation mode - checking for amber trigger words only")
                vosk_result = self.vosk_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
                logger.debug(f"Vosk result during dictation: '{vosk_result}'")
                
                if self._is_amber_trigger(vosk_result):
                    logger.info(f"Stop word '{vosk_result}' detected during dictation")
                    await self._publish_recognition_result(vosk_result, 0, "vosk", STTMode.COMMAND)
                else:
                    logger.debug(f"No amber trigger detected in: '{vosk_result}' - ignoring during dictation")
                return
            
            # Fast command processing (only when NOT in dictation mode)
            logger.debug("Processing command audio in normal mode")
            
            # Publish processing started event for UI progress
            await self.event_bus.publish(
                STTProcessingStartedEvent(
                    engine="vosk",
                    mode=STTMode.COMMAND.value,
                    audio_size_bytes=len(event_data.audio_bytes)
                )
            )
            processing_start = time.time()
            recognized_text = self.vosk_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
            processing_time = (time.time() - processing_start) * 1000
            
            if recognized_text and recognized_text.strip():
                if not self._duplicate_filter.is_duplicate(recognized_text):
                    await self._publish_recognition_result(recognized_text, processing_time, "vosk", STTMode.COMMAND) 
            else:
                # No speech detected - try sound recognition
                await self._publish_sound_recognition_event(event_data.audio_bytes, event_data.sample_rate)
            
            # Publish processing completed event for UI progress
            await self.event_bus.publish(
                STTProcessingCompletedEvent(
                    engine="vosk",
                    mode=STTMode.COMMAND.value,
                    processing_time_ms=processing_time,
                    text_length=len(recognized_text) if recognized_text else 0
                )
            )
        except Exception as e:
            logger.error(f"Error processing command audio: {e}")

    async def _handle_dictation_audio_segment(self, event_data: DictationAudioSegmentReadyEvent):
        """Process dictation audio"""
        try:
            if not self._engines_initialized:
                logger.error("STT engines not initialized")
                return

            # Publish processing started event
            await self.event_bus.publish(
                STTProcessingStartedEvent(
                    engine="whisper",
                    mode=STTMode.DICTATION.value,
                    audio_size_bytes=len(event_data.audio_bytes)
                )
            )
            processing_start = time.time()
            recognized_text = self.whisper_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
            processing_time = (time.time() - processing_start) * 1000
            
            if recognized_text and recognized_text.strip():
                if not self._duplicate_filter.is_duplicate(recognized_text):
                    await self._publish_recognition_result(recognized_text, processing_time, "whisper", STTMode.DICTATION)

            # Publish processing completed event
            await self.event_bus.publish(
                STTProcessingCompletedEvent(
                    engine="whisper",
                    mode=STTMode.DICTATION.value,
                    processing_time_ms=processing_time,
                    text_length=len(recognized_text) if recognized_text else 0
                )
            )

        except Exception as e:
            logger.error(f"Error processing dictation audio: {e}")
    
    def _is_amber_trigger(self, text: Optional[str]) -> bool:
        """Check if text contains amber trigger words"""
        if not text:
            return False
        return any(word in text.lower().strip() for word in self._amber_words)
    
    async def _handle_dictation_mode_change(self, event_data: DictationModeDisableOthersEvent):
        """Handle dictation mode changes"""
        with self._processing_lock:
            old_state = self._dictation_active
            self._dictation_active = event_data.dictation_mode_active
            logger.info(f"STT service dictation mode changed: {old_state} -> {self._dictation_active} (event: {event_data.dictation_mode_active})")
            
            # Additional debug info
            if self._dictation_active:
                logger.info("STT service now in DICTATION mode - command audio will only check for amber triggers")
            else:
                logger.info("STT service now in COMMAND mode - normal command processing enabled")

    def get_status(self) -> Dict[str, Any]:
        """Get current STT service status"""
        return {
            "dictation_active": self._dictation_active,
            "engines_initialized": self._engines_initialized,
            "vosk_initialized": self.vosk_engine is not None,
            "whisper_initialized": self.whisper_engine is not None,
            "last_recognized_text": self._last_recognized_text[:50] + "..." if len(self._last_recognized_text) > 50 else self._last_recognized_text
        }
    
    def connect_to_audio_service(self, audio_service):
        """Connect to audio service for streaming recognition"""
        try:
            if self.vosk_engine and hasattr(audio_service, 'set_streaming_stt_engine'):
                audio_service.set_streaming_stt_engine(self.vosk_engine)
                logger.info("Connected STT service to audio service for streaming")
        except Exception as e:
            logger.error(f"Failed to connect to audio service: {e}")

    async def shutdown(self):
        """Shutdown STT service"""
        try:
            logger.info("Shutting down STT service")
            self.vosk_engine = None
            self.whisper_engine = None
            logger.info("STT service shutdown complete")
        except Exception as e:
            logger.error(f"Error during STT service shutdown: {e}")
