from __future__ import annotations

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
from vocalance.app.lifecycle.worker import run_blocking
from vocalance.app.services.base_service import Service
from vocalance.app.services.command_flow.speech_recognition.vosk_engine import VoskEngine

logger = logging.getLogger(__name__)


class CommandSpeechService(Service):
    """Owns the Vosk engine and handles command-mode + dictation side-channel speech recognition.

    In normal command mode, every :class:`CommandAudioSegmentReadyEvent` is
    transcribed and re-published as a :class:`CommandTextRecognizedEvent`.
    During dictation, the same engine watches for the stop trigger and any
    modifier phrases (so the dictation coordinator can react without a
    separate transcription path).
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus)
        self._config = config
        self._dictation_active: bool = False
        self._current_dictation_mode: str = "inactive"
        self.vosk_engine: Optional[VoskEngine] = None
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

        self.subscribe(CommandAudioSegmentReadyEvent, self._handle_command_audio_segment)
        self.subscribe(DictationModeDisableOthersEvent, self._handle_dictation_mode_change)

    async def initialize(self) -> bool:
        """Load Vosk on a daemon worker thread."""
        cfg = self._config

        def _load_vosk() -> VoskEngine:
            return VoskEngine(
                model_path=cfg.asset_paths.get_vosk_model_path(),
                sample_rate=cfg.stt.sample_rate,
                config=cfg,
            )

        self.vosk_engine = await run_blocking(_load_vosk, name="vosk-load")
        logger.info("Vosk engine initialized")
        return True

    async def _handle_command_audio_segment(self, event: CommandAudioSegmentReadyEvent) -> None:
        if self._dictation_active:
            vosk_result = await self.vosk_engine.recognize(event.audio_bytes, event.sample_rate)
            if self._is_stop_trigger(vosk_result):
                if self._current_dictation_mode != "inactive":
                    await self.event_bus.publish(DictationStopWordDetectedEvent(mode=self._current_dictation_mode))
                await self.event_bus.publish(
                    CommandTextRecognizedEvent(text=vosk_result, processing_time_ms=0, engine="vosk", mode="command")
                )
            else:
                mod = self._match_modifier_phrase(vosk_result)
                if mod:
                    await self.event_bus.publish(
                        DictationModifierPhraseEvent(modifier_id=mod, raw_recognized_text=vosk_result or "")
                    )
            return

        start = time.time()
        recognized_text = await self.vosk_engine.recognize(event.audio_bytes, event.sample_rate)
        processing_time = (time.time() - start) * 1000
        if recognized_text and recognized_text.strip():
            await self.event_bus.publish(
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
        if self.vosk_engine is not None:
            await self.vosk_engine.shutdown()
            self.vosk_engine = None
        gc.collect()
        await super().shutdown()
