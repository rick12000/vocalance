import gc
import logging
import threading
import time
from enum import Enum
from typing import Optional

from iris.app.config.app_config import GlobalAppConfig
from iris.app.event_bus import EventBus
from iris.app.events.command_management_events import CommandMappingsUpdatedEvent
from iris.app.events.core_events import (
    CommandAudioSegmentReadyEvent,
    CommandTextRecognizedEvent,
    DictationAudioSegmentReadyEvent,
    DictationTextRecognizedEvent,
    ProcessAudioChunkForSoundRecognitionEvent,
    STTProcessingCompletedEvent,
    STTProcessingStartedEvent,
)
from iris.app.events.dictation_events import DictationModeDisableOthersEvent
from iris.app.services.audio.smart_timeout_manager import SmartTimeoutManager
from iris.app.services.audio.stt_utils import DuplicateTextFilter
from iris.app.services.audio.vosk_stt import EnhancedVoskSTT
from iris.app.services.audio.whisper_stt import WhisperSpeechToText

logger = logging.getLogger(__name__)


class STTMode(Enum):
    COMMAND = "command"
    DICTATION = "dictation"


class SpeechToTextService:
    """STT service with dual engines (Vosk for commands, Whisper for dictation).

    Manages mode-aware audio processing with amber trigger detection during dictation
    and smart timeout management for command ambiguity resolution.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        self.event_bus = event_bus
        self.config = config
        self.stt_config = config.stt
        self.logger = logging.getLogger(self.__class__.__name__)
        self._dictation_active: bool = False
        self._processing_lock = threading.RLock()
        self.vosk_engine: Optional[EnhancedVoskSTT] = None
        self.whisper_engine: Optional[WhisperSpeechToText] = None
        self._engines_initialized: bool = False
        self._duplicate_filter = DuplicateTextFilter(cache_size=5, duplicate_threshold_ms=1000)
        self._smart_timeout_manager = SmartTimeoutManager(app_config=config)
        self._amber_words = {"amber", "stop", "end"}

        logger.info(f"SpeechToTextService initialized - initial dictation_active: {self._dictation_active}")

    def initialize_engines(self) -> None:
        if self._engines_initialized:
            return

        try:
            logger.info("Initializing STT engines...")

            # Initialize Vosk engine
            logger.info("Loading Vosk STT engine...")
            self.vosk_engine = EnhancedVoskSTT(
                model_path=self.config.asset_paths.get_vosk_model_path(),
                sample_rate=self.stt_config.sample_rate,
                config=self.config,
            )

            # Initialize Whisper engine
            logger.info("Loading Whisper STT engine...")
            self.whisper_engine = WhisperSpeechToText(
                model_name=self.stt_config.whisper_model,
                device=self.stt_config.whisper_device,
                sample_rate=self.stt_config.sample_rate,
                config=self.stt_config,
            )

            self._engines_initialized = True
            logger.info("All STT engines initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize STT engines: {e}", exc_info=True)
            raise

    def setup_subscriptions(self) -> None:
        self.event_bus.subscribe(event_type=CommandAudioSegmentReadyEvent, handler=self._handle_command_audio_segment)
        self.event_bus.subscribe(event_type=DictationAudioSegmentReadyEvent, handler=self._handle_dictation_audio_segment)
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)
        self.event_bus.subscribe(event_type=CommandMappingsUpdatedEvent, handler=self._handle_command_mappings_updated)

        logger.info("STT service event subscriptions configured")

    async def _publish_recognition_result(self, text: str, processing_time: float, engine: str, mode: STTMode) -> None:
        if mode == STTMode.DICTATION:
            event = DictationTextRecognizedEvent(text=text, processing_time_ms=processing_time, engine=engine, mode=mode.value)
        else:
            event = CommandTextRecognizedEvent(text=text, processing_time_ms=processing_time, engine=engine, mode=mode.value)
        await self.event_bus.publish(event)
        logger.info(f"Published {type(event).__name__}: '{text}' from {engine}")

        self._last_recognized_text = text
        self._last_text_time = time.time()

    async def _publish_sound_recognition_event(self, audio_bytes: bytes, sample_rate: int) -> None:
        try:
            sound_event = ProcessAudioChunkForSoundRecognitionEvent(audio_chunk=audio_bytes, sample_rate=sample_rate)
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
                STTProcessingStartedEvent(engine="vosk", mode=STTMode.COMMAND.value, audio_size_bytes=len(event_data.audio_bytes))
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
                    text_length=len(recognized_text) if recognized_text else 0,
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
                    engine="whisper", mode=STTMode.DICTATION.value, audio_size_bytes=len(event_data.audio_bytes)
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
                    text_length=len(recognized_text) if recognized_text else 0,
                )
            )

        except Exception as e:
            logger.error(f"Error processing dictation audio: {e}")

    def _is_amber_trigger(self, text: Optional[str]) -> bool:
        if not text:
            return False
        return any(word in text.lower().strip() for word in self._amber_words)

    async def _handle_dictation_mode_change(self, event_data: DictationModeDisableOthersEvent) -> None:
        with self._processing_lock:
            old_state = self._dictation_active
            self._dictation_active = event_data.dictation_mode_active
            logger.info(
                f"STT service dictation mode changed: {old_state} -> {self._dictation_active} (event: {event_data.dictation_mode_active})"
            )

            # Additional debug info
            if self._dictation_active:
                logger.info("STT service now in DICTATION mode - command audio will only check for amber triggers")
            else:
                logger.info("STT service now in COMMAND mode - normal command processing enabled")

    async def _handle_command_mappings_updated(self, event_data: CommandMappingsUpdatedEvent):
        """Handle command mappings updates to refresh smart timeout manager"""
        try:
            if event_data.updated_mappings and self._smart_timeout_manager:
                # Convert mappings list to simple dict for smart timeout manager
                command_map = {cmd.command_key: None for cmd in event_data.updated_mappings}
                self._smart_timeout_manager.update_command_action_map(command_map)
                logger.info(f"Updated smart timeout manager with {len(command_map)} command mappings")
        except Exception as e:
            logger.error(f"Error handling command mappings update: {e}")

    def update_command_action_map(self, command_action_map) -> None:
        if self._smart_timeout_manager:
            self._smart_timeout_manager.update_command_action_map(command_action_map)
            logger.info(f"Updated smart timeout manager with {len(command_action_map)} commands")

    async def shutdown(self) -> None:
        try:
            logger.info("Shutting down STT service")

            # Properly shutdown Vosk engine
            if hasattr(self, "vosk_engine") and self.vosk_engine is not None:
                await self.vosk_engine.shutdown()
                del self.vosk_engine
                self.vosk_engine = None

            # Properly shutdown Whisper engine
            if hasattr(self, "whisper_engine") and self.whisper_engine is not None:
                await self.whisper_engine.shutdown()
                del self.whisper_engine
                self.whisper_engine = None

            # Clear duplicate filter
            if hasattr(self, "_duplicate_filter") and self._duplicate_filter is not None:
                del self._duplicate_filter
                self._duplicate_filter = None

            # Clear smart timeout manager
            if hasattr(self, "_smart_timeout_manager") and self._smart_timeout_manager is not None:
                del self._smart_timeout_manager
                self._smart_timeout_manager = None

            # Force garbage collection
            gc.collect()

            logger.info("STT service shutdown complete")
        except Exception as e:
            logger.error(f"Error during STT service shutdown: {e}", exc_info=True)
