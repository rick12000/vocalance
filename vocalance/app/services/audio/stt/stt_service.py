import asyncio
import gc
import logging
import threading
import time
from enum import Enum
from typing import Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import (
    CommandAudioSegmentReadyEvent,
    CommandTextRecognizedEvent,
    DictationAudioSegmentReadyEvent,
    DictationTextRecognizedEvent,
    ProcessAudioChunkForSoundRecognitionEvent,
    STTProcessingCompletedEvent,
    STTProcessingStartedEvent,
)
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.audio.stt.stt_utils import DuplicateTextFilter
from vocalance.app.services.audio.stt.vosk_stt import VoskSTT
from vocalance.app.services.audio.stt.whisper_stt import WhisperSTT

logger = logging.getLogger(__name__)


class STTMode(Enum):
    COMMAND = "command"
    DICTATION = "dictation"


class SpeechToTextService:
    """Dual-engine speech-to-text service with mode-specific processing.

    Orchestrates two STT engines: Vosk (fast, offline) for command mode and Whisper
    (accurate, model-based) for dictation mode. Manages mode transitions, duplicate
    text filtering, stop trigger detection during dictation, and publishes recognized
    text events. Processes audio segments from CommandAudioListener and DictationAudioListener
    with proper synchronization and resource optimization.

    Attributes:
        vosk_engine: VoskSTT for fast command recognition.
        whisper_engine: WhisperSTT for accurate dictation transcription.
        _dictation_active: Flag indicating if dictation mode is currently active.
        _duplicate_filter: Filters duplicate transcription results.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        """Initialize STT service with configuration and event bus.

        Args:
            event_bus: EventBus for subscribing to audio events and publishing text.
            config: Global application configuration with STT and audio settings.
        """
        self.event_bus = event_bus
        self.config = config
        self.stt_config = config.stt
        self.logger = logging.getLogger(self.__class__.__name__)
        self._dictation_active: bool = False
        self._state_lock = asyncio.Lock()
        self.vosk_engine: Optional[VoskSTT] = None
        self.whisper_engine: Optional[WhisperSTT] = None
        self._engines_initialized: bool = False
        self._duplicate_filter = DuplicateTextFilter(cache_size=5, duplicate_threshold_ms=1000)
        self._stop_trigger = config.dictation.stop_trigger

        logger.debug(f"SpeechToTextService initialized - initial dictation_active: {self._dictation_active}")

    async def initialize_engines(self, shutdown_coordinator=None) -> None:
        """Initialize Vosk and Whisper STT engines with cancellation support.

        Loads Vosk engine synchronously, then downloads and initializes Whisper model
        in a background thread with periodic cancellation checks. Supports early
        termination if shutdown is requested during model download.

        Args:
            shutdown_coordinator: Optional coordinator for checking shutdown requests during init.

        Raises:
            asyncio.CancelledError: If shutdown is requested during Whisper download.
            Exception: If engine initialization fails.
        """
        if self._engines_initialized:
            return

        logger.debug("Initializing STT engines...")

        logger.debug("Loading Vosk STT engine...")
        self.vosk_engine = VoskSTT(
            model_path=self.config.asset_paths.get_vosk_model_path(),
            sample_rate=self.stt_config.sample_rate,
            config=self.config,
        )

        logger.debug("Loading Whisper STT engine...")

        # Start whisper download in daemon thread
        whisper_result = [None]  # Mutable container for thread result
        whisper_error = [None]

        def load_whisper():
            try:
                whisper_result[0] = WhisperSTT(
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
        """Setup event subscriptions for STT service.

        Subscribes to command audio, dictation audio, and dictation mode change events
        for processing audio segments and switching engine behavior.
        """
        self.event_bus.subscribe(event_type=CommandAudioSegmentReadyEvent, handler=self._handle_command_audio_segment)
        self.event_bus.subscribe(event_type=DictationAudioSegmentReadyEvent, handler=self._handle_dictation_audio_segment)
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)

        logger.info("STT service event subscriptions configured")

    async def _publish_recognition_result(self, text: str, processing_time: float, engine: str, mode: STTMode) -> None:
        """Publish recognized text event based on mode.

        Args:
            text: Recognized text from STT engine.
            processing_time: Processing time in milliseconds.
            engine: Engine name (vosk or whisper).
            mode: STTMode enum indicating command or dictation.
        """
        if mode == STTMode.DICTATION:
            event = DictationTextRecognizedEvent(text=text, processing_time_ms=processing_time, engine=engine, mode=mode.value)
        else:
            event = CommandTextRecognizedEvent(text=text, processing_time_ms=processing_time, engine=engine, mode=mode.value)
        await self.event_bus.publish(event)
        logger.info(f"Published {type(event).__name__}: '{text}' from {engine}")

    async def _publish_sound_recognition_event(self, audio_bytes: bytes, sample_rate: int) -> None:
        """Publish audio to sound recognition when no speech is detected.

        Args:
            audio_bytes: Raw audio data to process for sound recognition.
            sample_rate: Sample rate of the audio.
        """
        sound_event = ProcessAudioChunkForSoundRecognitionEvent(audio_chunk=audio_bytes, sample_rate=sample_rate)
        await self.event_bus.publish(sound_event)
        logger.debug(f"Published sound recognition event for {len(audio_bytes)} bytes")

    async def _handle_command_audio_segment(self, event_data: CommandAudioSegmentReadyEvent) -> None:
        """Process command audio with mode-aware behavior.

        In command mode: runs full Vosk recognition and publishes results or forwards
        to sound recognition. In dictation mode: only checks for stop trigger words
        and ignores other commands to avoid interrupting dictation flow.

        Args:
            event_data: Event containing command audio segment and sample rate.
        """
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

    async def _handle_dictation_audio_segment(self, event_data: DictationAudioSegmentReadyEvent) -> None:
        """Process dictation audio using Whisper engine for accuracy.

        Uses Whisper STT for high-accuracy transcription of longer dictation segments.
        Filters duplicates and publishes dictation text events with processing metrics.

        **Optimization**: Skips Whisper processing if not in dictation mode to save resources.
        DictationAudioListener emits events continuously, but Whisper (expensive) only runs
        when dictation mode is active.

        Args:
            event_data: Event containing dictation audio segment and sample rate.
        """
        if not self._engines_initialized:
            logger.error("STT engines not initialized")
            return

        # Check if dictation mode is active - skip expensive Whisper processing if not
        async with self._state_lock:
            is_dictation_active = self._dictation_active

        if not is_dictation_active:
            logger.debug(
                f"Skipping Whisper processing for dictation segment ({len(event_data.audio_bytes)} bytes) "
                "- not in dictation mode"
            )
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
        """Check if recognized text contains the configured stop trigger word.

        Args:
            text: Recognized text to check.

        Returns:
            True if stop trigger is detected (case-insensitive), False otherwise.
        """
        if not text:
            return False
        return self._stop_trigger in text.lower().strip()

    async def _handle_dictation_mode_change(self, event_data: DictationModeDisableOthersEvent) -> None:
        """Handle dictation mode state changes for mode-aware processing.

        Updates internal dictation_active flag to control command audio processing
        behavior: when active, only stop triggers are detected; when inactive, full
        command recognition is performed.

        Args:
            event_data: Event containing dictation mode activation state.
        """
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
        """Shutdown STT service and release all engine resources.

        Shuts down both Vosk and Whisper engines, releases duplicate filter,
        explicitly deletes references to enable garbage collection, and runs
        gc.collect() to free memory immediately.
        """
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
