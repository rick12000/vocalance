import logging

import numpy as np
from PySide6.QtCore import QTimer

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioChunkCapturedEvent
from vocalance.app.events.dictation_events import (
    DictationModifierStateChangedEvent,
    DictationPausedStateEvent,
    DictationSessionEvent,
    DictationStatusChangedEvent,
    DictationStopWordDetectedEvent,
    FinalDictationTextEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingFailedEvent,
    LLMProcessingReadyEvent,
    LLMProcessingStartedEvent,
    LLMTokenGeneratedEvent,
    PartialDictationTextEvent,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController
from vocalance.app.ui.features.dictation.popup_view import QtDictationPopupView


class QtDictationPopupController(QtBaseController):
    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger(self.__class__.__name__))
        self.popup_view = QtDictationPopupView()
        self.llm_stream_session_id: str | None = None
        self.subscribe(DictationStatusChangedEvent, self.on_dictation_status_changed)
        self.subscribe(DictationSessionEvent, self.on_dictation_session)
        self.subscribe(PartialDictationTextEvent, self.on_partial_text)
        self.subscribe(FinalDictationTextEvent, self.on_final_text)
        self.subscribe(LLMProcessingStartedEvent, self.on_llm_started)
        self.subscribe(LLMProcessingCompletedEvent, self.on_llm_completed)
        self.subscribe(LLMProcessingFailedEvent, self.on_llm_failed)
        self.subscribe(LLMTokenGeneratedEvent, self.on_llm_token)
        self.subscribe(DictationStopWordDetectedEvent, self.on_stop_word_detected)
        self.subscribe(DictationPausedStateEvent, self.on_paused_state_changed)
        self.subscribe(DictationModifierStateChangedEvent, self.on_modifier_state_changed)
        self.subscribe(AudioChunkCapturedEvent, self.on_audio_chunk_captured)

    def on_dictation_status_changed(self, status_changed: DictationStatusChangedEvent) -> None:
        if status_changed.is_active and status_changed.show_ui:
            if status_changed.mode not in ("smart", "visual", "hidden", "amend"):
                self.popup_view.show_simple_listening()
        elif not status_changed.is_active:
            self.popup_view.hide_popup()

    def on_dictation_session(self, session: DictationSessionEvent) -> None:
        mode, state = session.mode, session.state
        if state == "started":
            if mode == "amend":
                self.popup_view.show_amend_dictation()
            elif mode == "smart":
                self.popup_view.show_smart_dictation()
            elif mode == "visual":
                self.popup_view.show_visual_dictation()
            elif mode == "hidden":
                self.popup_view.show_simple_listening()
        elif state == "stopped":
            if mode in ("smart", "amend"):
                self.popup_view.show_llm_processing()
            elif mode in ("visual", "hidden"):
                self.popup_view.hide_popup()

    def on_partial_text(self, partial: PartialDictationTextEvent) -> None:
        if partial.text:
            self.popup_view.display_partial_text(partial.text, partial.segment_id)

    def on_final_text(self, final: FinalDictationTextEvent) -> None:
        if final.text:
            self.popup_view.display_final_text(final.text, final.segment_id)

    async def on_llm_started(self, started: LLMProcessingStartedEvent) -> None:
        self.popup_view.update_llm_status("Processing...")
        session_id = started.session_id or "default"
        self.llm_stream_session_id = session_id
        await self.event_bus.publish(LLMProcessingReadyEvent(session_id=session_id))

    def on_llm_token(self, token_event: LLMTokenGeneratedEvent) -> None:
        if self.llm_stream_session_id is None or token_event.session_id != self.llm_stream_session_id:
            return
        if token_event.token:
            self.popup_view.append_llm_token(token_event.token)

    def on_llm_failed(self, failed: LLMProcessingFailedEvent) -> None:
        self.llm_stream_session_id = None

    def on_llm_completed(self, llm_completion: LLMProcessingCompletedEvent) -> None:
        self.llm_stream_session_id = None
        self.popup_view.update_llm_status("Complete!")
        QTimer.singleShot(1500, self.popup_view.hide_popup)

    def on_modifier_state_changed(self, modifier_state: DictationModifierStateChangedEvent) -> None:
        if modifier_state.active and modifier_state.display_label:
            self.popup_view.set_modifier_banner(modifier_state.display_label, True)
        else:
            self.popup_view.set_modifier_banner("", False)

    def on_stop_word_detected(self, stop_word: DictationStopWordDetectedEvent) -> None:
        if stop_word.mode in ("hidden", "visual", "smart", "amend"):
            self.popup_view.set_border_orange()

    def on_paused_state_changed(self, event: DictationPausedStateEvent) -> None:
        if event.is_paused:
            self.popup_view.set_border_yellow()
        else:
            self.popup_view.reset_border_color()

    def on_audio_chunk_captured(self, event: AudioChunkCapturedEvent) -> None:
        if not self.popup_view.isVisible() or self.popup_view.current_mode != "simple":
            return
        audio_data = np.frombuffer(event.pcm_bytes, dtype=np.int16)
        if len(audio_data) == 0:
            return
        rms = float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))
        normalized_level = min(1.0, rms / 5000.0)
        self.popup_view.update_audio_level(normalized_level)

    def shutdown(self) -> None:
        self.popup_view.hide_popup()
        super().shutdown()
