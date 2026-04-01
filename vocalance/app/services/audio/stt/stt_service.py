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
    STTProcessingCompletedEvent,
    STTProcessingStartedEvent,
)
from vocalance.app.events.dictation_events import (
    DictationModeDisableOthersEvent,
    DictationModifierId,
    DictationModifierPhraseEvent,
    DictationStopWordDetectedEvent,
)
from vocalance.app.services.audio.stt.moonshine_stt import MoonshineSTT
from vocalance.app.services.audio.stt.vosk_stt import VoskSTT

logger = logging.getLogger(__name__)


class STTMode(Enum):
    COMMAND = "command"


class SpeechToTextService:
    """Dual-engine speech-to-text service with mode-specific processing.

    Vosk handles command segments (including stop-word checks during dictation).
    Dictation text is produced by Moonshine streaming sessions owned by DictationCoordinator.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
    ) -> None:
        """Store config, bus, and placeholders for Vosk and Moonshine engines."""
        self.event_bus = event_bus
        self.config = config
        self.stt_config = config.stt
        self.logger = logging.getLogger(self.__class__.__name__)
        self._dictation_active: bool = False
        self._current_dictation_mode: str = "inactive"
        self._state_lock = asyncio.Lock()
        self.vosk_engine: Optional[VoskSTT] = None
        self.moonshine_engine: Optional[MoonshineSTT] = None
        self._engines_initialized: bool = False
        self._stop_trigger = config.dictation.stop_trigger
        d = config.dictation
        mod_pairs: list[tuple[str, DictationModifierId]] = [
            (d.modifier_upper_phrase.lower(), "upper"),
            (d.modifier_capitals_phrase.lower(), "capitals"),
            (d.modifier_camel_phrase.lower(), "camel"),
            (d.modifier_snake_phrase.lower(), "snake"),
            (d.modifier_spelling_phrase.lower(), "spelling"),
        ]
        self._modifier_phrases = sorted(mod_pairs, key=lambda x: -len(x[0]))

        logger.debug("SpeechToTextService initialized - initial dictation_active: %s", self._dictation_active)

    async def initialize_engines(self, shutdown_coordinator=None) -> None:
        if self._engines_initialized:
            return

        logger.debug("Initializing STT engines...")

        self.vosk_engine = VoskSTT(
            model_path=self.config.asset_paths.get_vosk_model_path(),
            sample_rate=self.stt_config.sample_rate,
            config=self.config,
        )

        logger.debug("Loading Moonshine STT engine...")

        moonshine_result = [None]
        moonshine_error = [None]

        def load_moonshine():
            try:
                moonshine_result[0] = MoonshineSTT(
                    sample_rate=self.stt_config.sample_rate,
                    config=self.config,
                )
            except Exception as e:
                moonshine_error[0] = e

        load_thread = threading.Thread(target=load_moonshine, daemon=True, name="MoonshineDownload")
        load_thread.start()

        while load_thread.is_alive():
            if shutdown_coordinator and shutdown_coordinator.is_shutdown_requested():
                logger.info("Moonshine download cancelled - abandoning thread")
                raise asyncio.CancelledError("Moonshine download cancelled")

            await asyncio.sleep(0.1)

        if moonshine_error[0]:
            raise moonshine_error[0]

        if moonshine_result[0] is None:
            raise RuntimeError("Moonshine initialization failed")

        self.moonshine_engine = moonshine_result[0]
        self._engines_initialized = True
        logger.info("All STT engines initialized successfully")

    def setup_subscriptions(self) -> None:
        """Subscribe to command segments and dictation mode toggles."""
        self.event_bus.subscribe(event_type=CommandAudioSegmentReadyEvent, handler=self._handle_command_audio_segment)
        self.event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)

        logger.info("STT service event subscriptions configured")

    async def _publish_recognition_result(self, text: str, processing_time: float, engine: str) -> None:
        event = CommandTextRecognizedEvent(
            text=text, processing_time_ms=processing_time, engine=engine, mode=STTMode.COMMAND.value
        )
        await self.event_bus.publish(event)
        logger.info("Published %s: '%s' from %s", type(event).__name__, text, engine)

    async def _handle_command_audio_segment(self, event_data: CommandAudioSegmentReadyEvent) -> None:
        """Spawn async processing so the bus handler returns quickly."""
        asyncio.create_task(self._process_command_audio_segment(event_data))

    async def _process_command_audio_segment(self, event_data: CommandAudioSegmentReadyEvent) -> None:
        """Run Vosk on the segment: full commands when idle, stop-word-only while dictation is active."""
        if not self._engines_initialized:
            logger.error("STT engines not initialized")
            return

        async with self._state_lock:
            is_dictation_active = self._dictation_active

        logger.debug("Processing command audio segment - dictation_active: %s", is_dictation_active)

        if is_dictation_active:
            logger.debug("In dictation mode - checking for stop trigger only")
            vosk_result = await self.vosk_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
            logger.debug("Vosk result during dictation: '%s'", vosk_result)

            if self._is_stop_trigger(vosk_result):
                logger.info("Stop word '%s' detected during dictation", vosk_result)
                await self._publish_stop_word_detected_event()
                await self._publish_recognition_result(vosk_result, 0, "vosk")
            else:
                mod = self._match_modifier_phrase(vosk_result)
                if mod:
                    logger.info("Dictation modifier phrase detected: %s in '%s'", mod, vosk_result)
                    await self.event_bus.publish(
                        DictationModifierPhraseEvent(modifier_id=mod, raw_recognized_text=vosk_result or "")
                    )
                else:
                    logger.debug("No stop trigger or modifier in: '%s' - ignoring during dictation", vosk_result)
            return

        logger.debug("Processing command audio in normal mode")

        await self.event_bus.publish(
            STTProcessingStartedEvent(engine="vosk", mode=STTMode.COMMAND.value, audio_size_bytes=len(event_data.audio_bytes))
        )
        processing_start = time.time()
        recognized_text = await self.vosk_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
        processing_time = (time.time() - processing_start) * 1000

        if recognized_text and recognized_text.strip():
            await self._publish_recognition_result(recognized_text, processing_time, "vosk")

        await self.event_bus.publish(
            STTProcessingCompletedEvent(
                engine="vosk",
                mode=STTMode.COMMAND.value,
                processing_time_ms=processing_time,
                text_length=len(recognized_text) if recognized_text else 0,
            )
        )

    def _is_stop_trigger(self, text: Optional[str]) -> bool:
        if not text:
            return False
        return self._stop_trigger in text.lower().strip()

    def _match_modifier_phrase(self, text: Optional[str]) -> Optional[DictationModifierId]:
        """Return the configured modifier id if ``text`` contains that phrase (substring match, longest first)."""
        if not text:
            return None
        t = text.lower().strip()
        for phrase, mid in self._modifier_phrases:
            if phrase and phrase in t:
                return mid
        return None

    async def _publish_stop_word_detected_event(self) -> None:
        """Notify listeners that the stop word was heard (when a dictation mode is active)."""
        async with self._state_lock:
            current_mode = self._current_dictation_mode

        if current_mode and current_mode != "inactive":
            event = DictationStopWordDetectedEvent(mode=current_mode)
            await self.event_bus.publish(event)
            logger.info("Published DictationStopWordDetectedEvent for mode: %s", current_mode)

    async def _handle_dictation_mode_change(self, event_data: DictationModeDisableOthersEvent) -> None:
        async with self._state_lock:
            old_state = self._dictation_active
            self._dictation_active = event_data.dictation_mode_active
            self._current_dictation_mode = event_data.dictation_mode
            logger.info(
                "STT service dictation mode changed: %s -> %s (mode: %s)",
                old_state,
                self._dictation_active,
                self._current_dictation_mode,
            )

            if self._dictation_active:
                logger.info(
                    "STT service now in DICTATION mode (%s) - command audio will only check for stop trigger",
                    self._current_dictation_mode,
                )
            else:
                logger.info("STT service now in COMMAND mode - normal command processing enabled")

    async def shutdown(self) -> None:
        """Shut down Vosk and Moonshine engines."""
        logger.info("Shutting down STT service")

        if hasattr(self, "vosk_engine") and self.vosk_engine is not None:
            await self.vosk_engine.shutdown()
            del self.vosk_engine
            self.vosk_engine = None

        if hasattr(self, "moonshine_engine") and self.moonshine_engine is not None:
            await self.moonshine_engine.shutdown()
            del self.moonshine_engine
            self.moonshine_engine = None

        gc.collect()
        logger.info("STT service shutdown complete")
