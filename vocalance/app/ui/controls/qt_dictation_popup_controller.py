"""Qt-based dictation popup controller.

Manages the dictation popup view and coordinates with dictation coordinator service.
"""

import asyncio
import logging

from vocalance.app.event_bus import EventBus
from vocalance.app.ui.views.qt_dictation_popup_view import QtDictationPopupView


class QtDictationPopupController:
    """Controller for dictation popup window.

    Manages:
    - Popup visibility based on dictation mode
    - Real-time dictation text updates
    - LLM output streaming
    - Event subscriptions for dictation lifecycle
    """

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
    ):
        """Initialize dictation popup controller.

        Args:
            event_bus: Event bus for pub/sub
            event_loop: Asyncio event loop
        """
        self.event_bus = event_bus
        self.event_loop = event_loop
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create popup view
        self.popup_view = QtDictationPopupView()

        # Subscribe to dictation events
        self._subscribe_to_events()

        self.logger.debug("QtDictationPopupController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to dictation-related events."""
        try:
            from vocalance.app.events.core_events import AudioChunkEvent
            from vocalance.app.events.dictation_events import (
                DictationStatusChangedEvent,
                FinalDictationTextEvent,
                HiddenDictationStartedEvent,
                HiddenDictationStoppedEvent,
                LLMProcessingCompletedEvent,
                LLMProcessingStartedEvent,
                LLMTokenGeneratedEvent,
                PartialDictationTextEvent,
                SmartDictationStartedEvent,
                SmartDictationStoppedEvent,
                SmartDictationTextDisplayEvent,
                VisualDictationStartedEvent,
                VisualDictationStoppedEvent,
            )

            # Standard/Type dictation modes (critical for showing simple listening popup)
            self.event_bus.subscribe(DictationStatusChangedEvent, self._on_dictation_status_changed)

            # Audio visualization
            self.event_bus.subscribe(AudioChunkEvent, self._on_audio_chunk)

            # Smart/Visual/Hidden dictation mode lifecycle
            self.event_bus.subscribe(SmartDictationStartedEvent, self._on_smart_started)
            self.event_bus.subscribe(SmartDictationStoppedEvent, self._on_smart_stopped)
            self.event_bus.subscribe(VisualDictationStartedEvent, self._on_visual_started)
            self.event_bus.subscribe(VisualDictationStoppedEvent, self._on_visual_stopped)
            self.event_bus.subscribe(HiddenDictationStartedEvent, self._on_hidden_started)
            self.event_bus.subscribe(HiddenDictationStoppedEvent, self._on_hidden_stopped)

            # Streaming dictation text events (partial/final for visual/smart modes)
            self.event_bus.subscribe(PartialDictationTextEvent, self._on_partial_text)
            self.event_bus.subscribe(FinalDictationTextEvent, self._on_final_text)

            # Smart dictation text display events
            self.event_bus.subscribe(SmartDictationTextDisplayEvent, self._on_smart_dictation_text)

            # LLM processing events (for smart mode)
            self.event_bus.subscribe(LLMProcessingStartedEvent, self._on_llm_started)
            self.event_bus.subscribe(LLMProcessingCompletedEvent, self._on_llm_completed)
            self.event_bus.subscribe(LLMTokenGeneratedEvent, self._on_llm_token)

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

    async def _on_dictation_status_changed(self, event) -> None:
        """Handle dictation status changes for standard/type modes.

        This is the critical event that was missing - without it, standard and type
        dictation modes don't show the popup at all.
        """
        try:
            is_active = getattr(event, "is_active", False)
            mode = getattr(event, "mode", "inactive")
            show_ui = getattr(event, "show_ui", False)

            self.logger.debug(f"DictationStatusChanged: is_active={is_active}, mode={mode}, show_ui={show_ui}")

            if is_active and show_ui:
                # Show popup for standard/type modes (not smart/visual/hidden - they have their own events)
                if mode not in ("smart", "visual", "hidden"):
                    self.show_simple_listening()
            else:
                # Hide popup when dictation becomes inactive
                if not is_active:
                    self.hide_popup()
        except Exception as e:
            self.logger.error(f"Error handling dictation status change: {e}", exc_info=True)

    async def _on_smart_started(self, event) -> None:
        """Handle smart dictation started event."""
        self.show_smart_dictation()

    async def _on_smart_stopped(self, event) -> None:
        """Handle smart dictation stopped event - switch to LLM processing mode."""
        # Switch to LLM processing mode (keeps window open, shows "AI Output" label)
        self.popup_view.show_llm_processing()
        self.logger.debug("Smart dictation stopped - switched to LLM processing mode")

    async def _on_visual_started(self, event) -> None:
        """Handle visual dictation started event."""
        self.show_visual_dictation()

    async def _on_visual_stopped(self, event) -> None:
        """Handle visual dictation stopped event."""
        self.hide_popup()

    async def _on_hidden_started(self, event) -> None:
        """Handle hidden dictation started event.

        Shows simple listening popup (sound wave only) for hidden mode,
        since hidden mode doesn't display streaming text.
        """
        self.show_simple_listening()
        self.logger.debug("Hidden dictation started - showing simple listening popup")

    async def _on_hidden_stopped(self, event) -> None:
        """Handle hidden dictation stopped event."""
        self.hide_popup()
        self.logger.debug("Hidden dictation stopped - hiding popup")

    async def _on_partial_text(self, event) -> None:
        """Handle partial dictation text event (gray/tentative text)."""
        try:
            text = getattr(event, "text", "")
            segment_id = getattr(event, "segment_id", "")
            self.logger.info(f"PARTIAL TEXT EVENT: text='{text}', segment_id={segment_id}")
            if text:
                # Call display_partial_text, not append_dictation_text!
                self.popup_view.display_partial_text(text, segment_id)
                self.logger.debug(f"Displayed partial text: '{text[:30]}...'")
        except Exception as e:
            self.logger.error(f"Error handling partial text event: {e}", exc_info=True)

    async def _on_final_text(self, event) -> None:
        """Handle final dictation text event (white/stable text)."""
        try:
            text = getattr(event, "text", "")
            segment_id = getattr(event, "segment_id", "")
            self.logger.info(f"FINAL TEXT EVENT: text='{text}', segment_id={segment_id}")
            if text:
                # Call display_final_text, not append_dictation_text!
                self.popup_view.display_final_text(text, segment_id)
                self.logger.debug(f"Displayed final text: '{text[:30]}...'")
        except Exception as e:
            self.logger.error(f"Error handling final text event: {e}", exc_info=True)

    async def _on_smart_dictation_text(self, event) -> None:
        """Handle smart dictation text display event."""
        try:
            text = getattr(event, "text", "")
            self.logger.info(f"SMART DICTATION TEXT EVENT: text='{text}'")
            if text:
                self.on_dictation_text_recognized(text)
                self.logger.debug(f"Smart dictation text appended: '{text[:30]}...'")
        except Exception as e:
            self.logger.error(f"Error handling smart dictation text: {e}", exc_info=True)

    async def _on_llm_started(self, event) -> None:
        """Handle LLM processing started event and signal UI ready."""
        self.on_llm_status_changed("Processing...")

        # Critical: Signal that UI is ready to receive LLM tokens
        # This tells the service to start streaming tokens
        from vocalance.app.events.dictation_events import LLMProcessingReadyEvent

        session_id = getattr(event, "session_id", None) or "default"
        ready_event = LLMProcessingReadyEvent(session_id=session_id)
        await self.event_bus.publish(ready_event)
        self.logger.debug(f"Published LLMProcessingReadyEvent for session {session_id}")

    async def _on_llm_token(self, event) -> None:
        """Handle LLM token generated event."""
        try:
            token = getattr(event, "token", "")
            if token:
                self.on_llm_token(token)
        except Exception as e:
            self.logger.error(f"Error handling LLM token: {e}", exc_info=True)

    async def _on_llm_completed(self, event) -> None:
        """Handle LLM processing completed event."""
        self.on_llm_status_changed("Complete!")
        # Hide popup after brief delay to show completion
        # Use QTimer instead of asyncio.sleep to avoid blocking event loop
        from PySide6.QtCore import QTimer

        QTimer.singleShot(1500, self.hide_popup)  # 1500ms = 1.5s
        self.logger.debug("Scheduled popup hide after 1.5s delay")

    async def _on_audio_chunk(self, event) -> None:
        """Handle audio chunk for visualization."""
        # Optimization: Only process if popup is visible and in simple mode
        if not self.popup_view.isVisible() or self.popup_view.current_mode != "simple":
            return

        try:
            import numpy as np

            # Convert bytes to numpy array (assuming int16)
            audio_data = np.frombuffer(event.audio_chunk, dtype=np.int16)

            if len(audio_data) == 0:
                return

            # Calculate RMS
            # Convert to float for calculation to avoid overflow
            rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))

            # Normalize
            # 16-bit audio max amplitude is 32768
            # Normal speech might be around 1000-5000 RMS
            # Use 5000 as reference max for visualization to make it sensitive
            max_ref = 5000.0
            normalized_level = min(1.0, rms / max_ref)

            self.popup_view.update_audio_level(normalized_level)

        except Exception:
            # Fail silently for performance in high-frequency callback
            pass

    def cleanup(self) -> None:
        """Clean up controller resources."""
        try:
            self.popup_view.hide_popup()
            self.logger.debug("Dictation popup controller cleaned up")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
