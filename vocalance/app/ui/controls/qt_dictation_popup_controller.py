import logging

from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import (
    DictationModifierStateChangedEvent,
    DictationSessionEvent,
    DictationStatusChangedEvent,
    DictationStopWordDetectedEvent,
    FinalDictationTextEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingReadyEvent,
    LLMProcessingStartedEvent,
    LLMTokenGeneratedEvent,
    PartialDictationTextEvent,
)
from vocalance.app.ui.views.qt_dictation_popup_view import QtDictationPopupView


class QtDictationPopupController:
    """Controller for dictation popup window.

    Manages:
    - Popup visibility based on dictation mode
    - Real-time dictation text updates
    - LLM output streaming
    - Event subscriptions for dictation lifecycle
    """

    def __init__(self, event_bus: EventBus) -> None:
        """Initialize dictation popup controller.

        Args:
            event_bus: Event bus for pub/sub.
        """
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create popup view
        self.popup_view = QtDictationPopupView()

        # Subscribe to dictation events
        self._subscribe_to_events()

        self.logger.debug("QtDictationPopupController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to dictation-related events."""
        try:
            self.event_bus.subscribe(DictationStatusChangedEvent, self._on_dictation_status_changed)

            # Smart/Visual/Hidden dictation mode lifecycle
            self.event_bus.subscribe(DictationSessionEvent, self._on_dictation_session)

            # Streaming dictation text events (partial/final for visual/smart modes)
            self.event_bus.subscribe(PartialDictationTextEvent, self._on_partial_text)
            self.event_bus.subscribe(FinalDictationTextEvent, self._on_final_text)

            # LLM processing events (for smart mode)
            self.event_bus.subscribe(LLMProcessingStartedEvent, self._on_llm_started)
            self.event_bus.subscribe(LLMProcessingCompletedEvent, self._on_llm_completed)
            self.event_bus.subscribe(LLMTokenGeneratedEvent, self._on_llm_token)

            # Stop word detection event
            self.event_bus.subscribe(DictationStopWordDetectedEvent, self._on_stop_word_detected)
            self.event_bus.subscribe(DictationModifierStateChangedEvent, self._on_modifier_state_changed)

            self.logger.debug("Dictation popup event subscriptions configured")
        except Exception as e:
            self.logger.error(f"Error setting up dictation popup subscriptions: {e}", exc_info=True)

    # Public API

    def show_simple_listening(self) -> None:
        """Show simple listening mode."""
        try:
            self.popup_view.show_simple_listening()
            self.logger.debug("Showing simple listening mode")
        except Exception as e:
            self.logger.error(f"Error showing simple listening: {e}", exc_info=True)

    def show_smart_dictation(self) -> None:
        """Show smart dictation mode (dictation + LLM)."""
        try:
            self.popup_view.show_smart_dictation()
            self.logger.debug("Showing smart dictation mode")
        except Exception as e:
            self.logger.error(f"Error showing smart dictation: {e}", exc_info=True)

    def show_visual_dictation(self) -> None:
        """Show visual dictation mode."""
        try:
            self.popup_view.show_visual_dictation()
            self.logger.debug("Showing visual dictation mode")
        except Exception as e:
            self.logger.error(f"Error showing visual dictation: {e}", exc_info=True)

    def hide_popup(self) -> None:
        """Hide the dictation popup."""
        try:
            self.popup_view.hide_popup()
            self.logger.debug("Dictation popup hidden")
        except Exception as e:
            self.logger.error(f"Error hiding popup: {e}", exc_info=True)

    def on_dictation_text_recognized(self, text: str) -> None:
        """Handle recognized dictation text.

        Args:
            text: Recognized text to append
        """
        try:
            self.popup_view.append_dictation_text(text)
            self.logger.debug(f"Appended dictation text: {text[:30]}...")
        except Exception as e:
            self.logger.error(f"Error appending dictation text: {e}", exc_info=True)

    def on_llm_token(self, token: str) -> None:
        """Handle LLM output token.

        Args:
            token: Token to append to LLM output
        """
        try:
            self.popup_view.append_llm_token(token)
        except Exception as e:
            self.logger.error(f"Error appending LLM token: {e}", exc_info=True)

    def on_llm_status_changed(self, status: str) -> None:
        """Handle LLM status change.

        Args:
            status: New status text
        """
        try:
            self.popup_view.update_llm_status(status)
            self.logger.debug(f"LLM status updated: {status}")
        except Exception as e:
            self.logger.error(f"Error updating LLM status: {e}", exc_info=True)

    # Event handlers

    def _on_dictation_status_changed(self, status_changed: DictationStatusChangedEvent) -> None:
        """Show or hide the simple listening popup for standard and type dictation."""
        try:
            is_active = status_changed.is_active
            mode = status_changed.mode
            show_ui = status_changed.show_ui

            self.logger.debug(f"DictationStatusChanged: is_active={is_active}, mode={mode}, show_ui={show_ui}")

            if is_active and show_ui:
                if mode not in ("smart", "visual", "hidden", "amend"):
                    self.show_simple_listening()
            elif not is_active:
                self.hide_popup()
        except Exception as e:
            self.logger.error(f"Error handling dictation status change: {e}", exc_info=True)

    def _on_dictation_session(self, session: DictationSessionEvent) -> None:
        """Handle dictation session start/stop events."""
        mode = session.mode
        state = session.state

        if state == "started":
            if mode == "amend":
                self.popup_view.show_amend_dictation()
            elif mode == "smart":
                self.show_smart_dictation()
            elif mode == "visual":
                self.show_visual_dictation()
            elif mode == "hidden":
                self.show_simple_listening()
                self.logger.debug("Hidden dictation started - showing simple listening popup")
        elif state == "stopped":
            if mode in ("smart", "amend"):
                self.popup_view.show_llm_processing()
                self.logger.debug("Dual-pane dictation stopped — LLM processing UI")
            elif mode == "visual":
                self.hide_popup()
            elif mode == "hidden":
                self.hide_popup()
                self.logger.debug("Hidden dictation stopped - hiding popup")

    def _on_partial_text(self, partial: PartialDictationTextEvent) -> None:
        """Handle partial dictation text event (gray/tentative text)."""
        try:
            text = partial.text
            segment_id = partial.segment_id
            self.logger.info(f"PARTIAL TEXT EVENT: text='{text}', segment_id={segment_id}")
            if text:
                # Call display_partial_text, not append_dictation_text!
                self.popup_view.display_partial_text(text, segment_id)
                self.logger.debug(f"Displayed partial text: '{text[:30]}...'")
        except Exception as e:
            self.logger.error(f"Error handling partial text event: {e}", exc_info=True)

    def _on_final_text(self, final: FinalDictationTextEvent) -> None:
        """Handle final dictation text event (white/stable text)."""
        try:
            text = final.text
            segment_id = final.segment_id
            self.logger.info(f"FINAL TEXT EVENT: text='{text}', segment_id={segment_id}")
            if text:
                # Call display_final_text, not append_dictation_text!
                self.popup_view.display_final_text(text, segment_id)
                self.logger.debug(f"Displayed final text: '{text[:30]}...'")
        except Exception as e:
            self.logger.error(f"Error handling final text event: {e}", exc_info=True)

    async def _on_llm_started(self, started: LLMProcessingStartedEvent) -> None:
        """Publish LLMProcessingReadyEvent once the popup can accept tokens."""
        self.on_llm_status_changed("Processing...")

        session_id = started.session_id or "default"
        ready_event = LLMProcessingReadyEvent(session_id=session_id)
        await self.event_bus.publish(ready_event)
        self.logger.debug(f"Published LLMProcessingReadyEvent for session {session_id}")

    def _on_llm_token(self, token_event: LLMTokenGeneratedEvent) -> None:
        """Append one token to the LLM output pane."""
        try:
            token = token_event.token
            if token:
                self.on_llm_token(token)
        except Exception as e:
            self.logger.error(f"Error handling LLM token: {e}", exc_info=True)

    def _on_llm_completed(self, _llm_completion: LLMProcessingCompletedEvent) -> None:
        """Show completion briefly, then hide the popup."""
        self.on_llm_status_changed("Complete!")
        # Hide popup after brief delay to show completion
        # Use QTimer instead of asyncio.sleep to avoid blocking event loop
        from PySide6.QtCore import QTimer

        QTimer.singleShot(1500, self.hide_popup)  # 1500ms = 1.5s
        self.logger.debug("Scheduled popup hide after 1.5s delay")

    def _on_modifier_state_changed(self, modifier_state: DictationModifierStateChangedEvent) -> None:
        """Forward modifier state to the popup view (chip appears only in smart, amend, and visual layouts)."""
        try:
            if modifier_state.active and modifier_state.display_label:
                self.popup_view.set_modifier_banner(modifier_state.display_label, True)
            else:
                self.popup_view.set_modifier_banner("", False)
        except Exception as e:
            self.logger.error("Error handling modifier state: %s", e, exc_info=True)

    def _on_stop_word_detected(self, stop_word: DictationStopWordDetectedEvent) -> None:
        """Set an orange border when the stop phrase is detected in supported modes."""
        try:
            mode = stop_word.mode
            self.logger.info(f"Stop word detected in {mode} mode - changing border to orange")

            # Only change border for modes that use the streaming popup (not simple listening)
            if mode in ("hidden", "visual", "smart", "amend"):
                self.popup_view.set_border_orange()
                self.logger.debug(f"Border color changed to orange for {mode} mode")
            else:
                self.logger.debug("Stop word in mode %s — border unchanged", mode)
        except Exception as e:
            self.logger.error(f"Error handling stop word detection: {e}", exc_info=True)

    def feed_audio_chunk_for_level_meter(self, audio_chunk: bytes) -> None:
        """Drive the simple-mode level meter from raw PCM (called from the audio VAD worker thread)."""
        if not self.popup_view.isVisible() or self.popup_view.current_mode != "simple":
            return

        try:
            import numpy as np

            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

            if len(audio_data) == 0:
                return

            rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))
            max_ref = 5000.0
            normalized_level = min(1.0, rms / max_ref)

            self.popup_view.update_audio_level(normalized_level)

        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up controller resources."""
        try:
            self.popup_view.hide_popup()
            self.logger.debug("Dictation popup controller cleaned up")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
