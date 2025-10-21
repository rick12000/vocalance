import asyncio
import gc
import logging
import threading
import time
from enum import Enum
from typing import Optional

from iris.app.config.app_config import GlobalAppConfig
from iris.app.event_bus import EventBus
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
from iris.app.services.audio.stt.stt_utils import DuplicateTextFilter
from iris.app.services.audio.stt.vosk_stt import EnhancedVoskSTT
from iris.app.services.audio.stt.whisper_stt import WhisperSpeechToText

logger = logging.getLogger(__name__)


class STTMode(Enum):
    COMMAND = "command"
    DICTATION = "dictation"


class SpeechToTextService:
    """STT service with dual engines (Vosk for commands, Whisper for dictation).

    Manages mode-aware audio processing with amber trigger detection during dictation.
    All operations are async-safe and use asyncio primitives for thread safety.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        self.event_bus = event_bus
        self.config = config
        self.stt_config = config.stt
        self.logger = logging.getLogger(self.__class__.__name__)
        self._dictation_active: bool = False
        self._state_lock = asyncio.Lock()
        self.vosk_engine: Optional[EnhancedVoskSTT] = None
        self.whisper_engine: Optional[WhisperSpeechToText] = None
        self._engines_initialized: bool = False
        self._duplicate_filter = DuplicateTextFilter(cache_size=5, duplicate_threshold_ms=1000)
        self._stop_trigger = config.dictation.stop_trigger

        logger.info(f"SpeechToTextService initialized - initial dictation_active: {self._dictation_active}")

    async def initialize_engines(self, shutdown_coordinator=None) -> None:
        """
        Initialize Vosk and Whisper engines.

        Args:
            shutdown_coordinator: Optional coordinator to check for cancellation
        """
        if self._engines_initialized:
            return

        logger.info("Initializing STT engines...")

        logger.info("Loading Vosk STT engine...")
        self.vosk_engine = EnhancedVoskSTT(
            model_path=self.config.asset_paths.get_vosk_model_path(),
            sample_rate=self.stt_config.sample_rate,
            config=self.config,
        )

        logger.info("Loading Whisper STT engine...")

        # Start whisper download in daemon thread
        whisper_result = [None]  # Mutable container for thread result
        whisper_error = [None]

        def load_whisper():
            try:
                whisper_result[0] = WhisperSpeechToText(
                    model_name=self.stt_config.whisper_model,
                    device=self.stt_config.whisper_device,
                    sample_rate=self.stt_config.sample_rate,
                    config=self.config,
                )
            except Exception as e:
                whisper_error[0] = e

        load_thread = threading.Thread(target=load_whisper, daemon=True, name="WhisperDownload")
        load_thread.start()

        # Poll for completion or cancellation
        while load_thread.is_alive():
            if shutdown_coordinator and shutdown_coordinator.is_shutdown_requested():
                logger.info("Whisper download cancelled - abandoning thread")
                raise asyncio.CancelledError("Whisper download cancelled")

            await asyncio.sleep(0.1)  # Check every 100ms

        # Check result
        if whisper_error[0]:
            raise whisper_error[0]

        if whisper_result[0] is None:
            raise RuntimeError("Whisper initialization failed")

        self.whisper_engine = whisper_result[0]
        self._engines_initialized = True
        logger.info("All STT engines initialized successfully")

    def setup_subscriptions(self) -> None:
        self.event_bus.subscribe(event_type=CommandAudioSegmentReadyEvent, handler=self._handle_command_audio_segment)
        self.event_bus.subscribe(event_type=DictationAudioSegmentReadyEvent, handler=self._handle_dictation_audio_segment)
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)

        logger.info("STT service event subscriptions configured")

    async def _publish_recognition_result(self, text: str, processing_time: float, engine: str, mode: STTMode) -> None:
        if mode == STTMode.DICTATION:
            event = DictationTextRecognizedEvent(text=text, processing_time_ms=processing_time, engine=engine, mode=mode.value)
        else:
            event = CommandTextRecognizedEvent(text=text, processing_time_ms=processing_time, engine=engine, mode=mode.value)
        await self.event_bus.publish(event)
        logger.info(f"Published {type(event).__name__}: '{text}' from {engine}")

    async def _publish_sound_recognition_event(self, audio_bytes: bytes, sample_rate: int) -> None:
        sound_event = ProcessAudioChunkForSoundRecognitionEvent(audio_chunk=audio_bytes, sample_rate=sample_rate)
        await self.event_bus.publish(sound_event)
        logger.debug(f"Published sound recognition event for {len(audio_bytes)} bytes")

    async def _handle_command_audio_segment(self, event_data: CommandAudioSegmentReadyEvent):
        """Process command audio"""
        if not self._engines_initialized:
            logger.error("STT engines not initialized")
            return

        async with self._state_lock:
            is_dictation_active = self._dictation_active

        logger.debug(f"Processing command audio segment - dictation_active: {is_dictation_active}")

        if is_dictation_active:
            logger.debug("In dictation mode - checking for stop trigger only")
            vosk_result = await self.vosk_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
            logger.debug(f"Vosk result during dictation: '{vosk_result}'")

            if self._is_stop_trigger(vosk_result):
                logger.info(f"Stop word '{vosk_result}' detected during dictation")
                await self._publish_recognition_result(vosk_result, 0, "vosk", STTMode.COMMAND)
            else:
                logger.debug(f"No stop trigger detected in: '{vosk_result}' - ignoring during dictation")
            return

        logger.debug("Processing command audio in normal mode")

        await self.event_bus.publish(
            STTProcessingStartedEvent(engine="vosk", mode=STTMode.COMMAND.value, audio_size_bytes=len(event_data.audio_bytes))
        )
        processing_start = time.time()
        recognized_text = await self.vosk_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
        processing_time = (time.time() - processing_start) * 1000

        if recognized_text and recognized_text.strip():
            if not await self._duplicate_filter.is_duplicate(recognized_text):
                await self._publish_recognition_result(recognized_text, processing_time, "vosk", STTMode.COMMAND)
        else:
            await self._publish_sound_recognition_event(event_data.audio_bytes, event_data.sample_rate)

        await self.event_bus.publish(
            STTProcessingCompletedEvent(
                engine="vosk",
                mode=STTMode.COMMAND.value,
                processing_time_ms=processing_time,
                text_length=len(recognized_text) if recognized_text else 0,
            )
        )

    async def _handle_dictation_audio_segment(self, event_data: DictationAudioSegmentReadyEvent):
        """Process dictation audio"""
        if not self._engines_initialized:
            logger.error("STT engines not initialized")
            return

        await self.event_bus.publish(
            STTProcessingStartedEvent(engine="whisper", mode=STTMode.DICTATION.value, audio_size_bytes=len(event_data.audio_bytes))
        )
        processing_start = time.time()
        recognized_text = await self.whisper_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
        processing_time = (time.time() - processing_start) * 1000

        if recognized_text and recognized_text.strip():
            if not await self._duplicate_filter.is_duplicate(recognized_text):
                await self._publish_recognition_result(recognized_text, processing_time, "whisper", STTMode.DICTATION)

        await self.event_bus.publish(
            STTProcessingCompletedEvent(
                engine="whisper",
                mode=STTMode.DICTATION.value,
                processing_time_ms=processing_time,
                text_length=len(recognized_text) if recognized_text else 0,
            )
        )

    def _is_stop_trigger(self, text: Optional[str]) -> bool:
        if not text:
            return False
        return self._stop_trigger in text.lower().strip()

    async def _handle_dictation_mode_change(self, event_data: DictationModeDisableOthersEvent) -> None:
        async with self._state_lock:
            old_state = self._dictation_active
            self._dictation_active = event_data.dictation_mode_active
            logger.info(
                f"STT service dictation mode changed: {old_state} -> {self._dictation_active} (event: {event_data.dictation_mode_active})"
            )

            if self._dictation_active:
                logger.info("STT service now in DICTATION mode - command audio will only check for stop trigger")
            else:
                logger.info("STT service now in COMMAND mode - normal command processing enabled")

    async def shutdown(self) -> None:
        logger.info("Shutting down STT service")

        if hasattr(self, "vosk_engine") and self.vosk_engine is not None:
            await self.vosk_engine.shutdown()
            del self.vosk_engine
            self.vosk_engine = None

        if hasattr(self, "whisper_engine") and self.whisper_engine is not None:
            await self.whisper_engine.shutdown()
            del self.whisper_engine
            self.whisper_engine = None

        if hasattr(self, "_duplicate_filter") and self._duplicate_filter is not None:
            del self._duplicate_filter
            self._duplicate_filter = None

        gc.collect()
        logger.info("STT service shutdown complete")
