import asyncio
import gc
import logging
import time
from typing import Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import CommandAudioSegmentReadyEvent, CommandTextRecognizedEvent
from vocalance.app.events.dictation_events import (
    DictationModeDisableOthersEvent,
    DictationModifierId,
    DictationModifierPhraseEvent,
    DictationStopWordDetectedEvent,
)
from vocalance.app.services.audio.stt.moonshine_stt import MoonshineSTT
from vocalance.app.services.audio.stt.vosk_stt import VoskSTT
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class SpeechToTextService(Service):
    """Dual-engine STT: Vosk for commands, Moonshine for dictation (owned by DictationCoordinator)."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        self._event_bus = event_bus
        self._config = config
        self._dictation_active: bool = False
        self._current_dictation_mode: str = "inactive"
        self.vosk_engine: Optional[VoskSTT] = None
        self.moonshine_engine: Optional[MoonshineSTT] = None
        self._stop_trigger = config.dictation.stop_trigger

        d = config.dictation
        mod_pairs: list[tuple[str, DictationModifierId]] = [
            (d.modifier_upper_phrase.lower(), "upper"),
            (d.modifier_capitals_phrase.lower(), "capitals"),
            (d.modifier_camel_phrase.lower(), "camel"),
            (d.modifier_snake_phrase.lower(), "snake"),
            (d.modifier_spelling_phrase.lower(), "spelling"),
            (d.modifier_kebab_phrase.lower(), "kebab"),
            (d.modifier_diminish_phrase.lower(), "diminish"),
            (d.modifier_strip_phrase.lower(), "strip"),
        ]
        self._modifier_phrases = sorted(mod_pairs, key=lambda x: -len(x[0]))

        event_bus.subscribe(CommandAudioSegmentReadyEvent, self._handle_command_audio_segment)
        event_bus.subscribe(DictationModeDisableOthersEvent, self._handle_dictation_mode_change)

    async def initialize(self) -> bool:
        stt_cfg = self._config.stt

        def _load_engines() -> tuple:
            vosk = VoskSTT(
                model_path=self._config.asset_paths.get_vosk_model_path(),
                sample_rate=stt_cfg.sample_rate,
                config=self._config,
            )
            moonshine = MoonshineSTT(sample_rate=stt_cfg.sample_rate, config=self._config)
            return vosk, moonshine

        self.vosk_engine, self.moonshine_engine = await asyncio.to_thread(_load_engines)
        logger.info("STT engines initialized")
        return True

    async def _handle_command_audio_segment(self, event: CommandAudioSegmentReadyEvent) -> None:
        if self._dictation_active:
            vosk_result = await self.vosk_engine.recognize(event.audio_bytes, event.sample_rate)
            if self._is_stop_trigger(vosk_result):
                if self._current_dictation_mode != "inactive":
                    await self._event_bus.publish(DictationStopWordDetectedEvent(mode=self._current_dictation_mode))
                await self._event_bus.publish(
                    CommandTextRecognizedEvent(text=vosk_result, processing_time_ms=0, engine="vosk", mode="command")
                )
            else:
                mod = self._match_modifier_phrase(vosk_result)
                if mod:
                    await self._event_bus.publish(
                        DictationModifierPhraseEvent(modifier_id=mod, raw_recognized_text=vosk_result or "")
                    )
            return

        start = time.time()
        recognized_text = await self.vosk_engine.recognize(event.audio_bytes, event.sample_rate)
        processing_time = (time.time() - start) * 1000
        if recognized_text and recognized_text.strip():
            await self._event_bus.publish(
                CommandTextRecognizedEvent(text=recognized_text, processing_time_ms=processing_time, engine="vosk", mode="command")
            )

    def _is_stop_trigger(self, text: Optional[str]) -> bool:
        return bool(text) and self._stop_trigger in text.lower().strip()

    def _match_modifier_phrase(self, text: Optional[str]) -> Optional[DictationModifierId]:
        if not text:
            return None
        t = text.lower().strip()
        for phrase, mid in self._modifier_phrases:
            if phrase and phrase in t:
                return mid
        return None

    async def _handle_dictation_mode_change(self, event: DictationModeDisableOthersEvent) -> None:
        self._dictation_active = event.dictation_mode_active
        self._current_dictation_mode = event.dictation_mode

    async def shutdown(self) -> None:
        self._event_bus.unsubscribe(CommandAudioSegmentReadyEvent, self._handle_command_audio_segment)
        self._event_bus.unsubscribe(DictationModeDisableOthersEvent, self._handle_dictation_mode_change)
        if self.vosk_engine is not None:
            await self.vosk_engine.shutdown()
            self.vosk_engine = None
        if self.moonshine_engine is not None:
            await self.moonshine_engine.shutdown()
            self.moonshine_engine = None
        gc.collect()
