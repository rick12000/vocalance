import asyncio
from enum import Enum

from vocalance.app.events.dictation_events import (
    DictationStatusChangedEvent,
    FinalDictationTextEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingFailedEvent,
    LLMProcessingReadyEvent,
    LLMProcessingStartedEvent,
    LLMTokenGeneratedEvent,
    PartialDictationTextEvent,
    SmartDictationRemoveCharactersEvent,
    SmartDictationStartedEvent,
    SmartDictationStoppedEvent,
    SmartDictationTextDisplayEvent,
    VisualDictationStartedEvent,
    VisualDictationStoppedEvent,
)
from vocalance.app.ui.controls.base_control import BaseController
from vocalance.app.ui.utils.ui_thread_utils import schedule_ui_update


class DictationPopupMode(Enum):
    """Modes for the dictation popup window."""

    HIDDEN = "hidden"
    SIMPLE_LISTENING = "simple_listening"
    SMART_DICTATION = "smart_dictation"
    VISUAL_DICTATION = "visual_dictation"
    LLM_PROCESSING = "llm_processing"


class DictationPopupController(BaseController):
    """Simplified controller for dictation popup"""

    def __init__(self, event_bus, event_loop, logger):
        super().__init__(event_bus, event_loop, logger, "DictationPopupController")

        self.current_mode = DictationPopupMode.HIDDEN

        self.subscribe_to_events(
            [
                (DictationStatusChangedEvent, self._handle_dictation_status_changed),
                (SmartDictationStartedEvent, self._handle_smart_dictation_started),
                (SmartDictationStoppedEvent, self._handle_smart_dictation_stopped),
                (VisualDictationStartedEvent, self._handle_visual_dictation_started),
                (VisualDictationStoppedEvent, self._handle_visual_dictation_stopped),
                (LLMProcessingStartedEvent, self._handle_llm_processing_started),
                (LLMProcessingCompletedEvent, self._handle_llm_processing_completed),
                (LLMProcessingFailedEvent, self._handle_llm_processing_failed),
                (LLMTokenGeneratedEvent, self._handle_llm_token_generated),
                (SmartDictationTextDisplayEvent, self._handle_smart_dictation_text_display),
                (SmartDictationRemoveCharactersEvent, self._handle_smart_dictation_remove_characters),
                (PartialDictationTextEvent, self._handle_partial_dictation_text),
                (FinalDictationTextEvent, self._handle_final_dictation_text),
            ]
        )

    async def _handle_dictation_status_changed(self, event_data) -> None:
        """Handle normal dictation status changes (not smart or visual dictation) - marshalled to UI thread"""
        if event_data.show_ui and event_data.is_active:
            if event_data.mode not in ("smart", "visual"):
                self.current_mode = DictationPopupMode.SIMPLE_LISTENING
                if self.view_callback:
                    schedule_ui_update(self.view_callback.show_simple_listening, event_data.mode, event_data.stop_command)
        else:
            if not event_data.is_active and (
                self.current_mode == DictationPopupMode.SIMPLE_LISTENING
                or self.current_mode
                in [DictationPopupMode.SMART_DICTATION, DictationPopupMode.VISUAL_DICTATION, DictationPopupMode.LLM_PROCESSING]
            ):
                if self.view_callback:
                    schedule_ui_update(self.view_callback.hide_popup)
                self.current_mode = DictationPopupMode.HIDDEN

    async def _handle_smart_dictation_started(self, event_data) -> None:
        """Show smart dictation popup - marshalled to UI thread"""
        self.current_mode = DictationPopupMode.SMART_DICTATION
        if self.view_callback:
            schedule_ui_update(self.view_callback.show_smart_dictation)

    async def _handle_smart_dictation_stopped(self, event_data) -> None:
        """Switch to LLM processing mode - marshalled to UI thread"""
        self.current_mode = DictationPopupMode.LLM_PROCESSING
        if self.view_callback:
            schedule_ui_update(self.view_callback.show_llm_processing)

    async def _handle_llm_processing_started(self, event_data) -> None:
        """Signal UI is ready for LLM processing - marshalled to UI thread"""
        if self.view_callback:
            schedule_ui_update(self.view_callback.update_llm_status, "Processing...")

        # Signal ready immediately
        session_id = getattr(event_data, "session_id", None) or "default"
        ready_event = LLMProcessingReadyEvent(session_id=session_id)
        await self.event_bus.publish(ready_event)

    async def _handle_llm_processing_completed(self, event_data) -> None:
        """Hide popup after brief completion display - marshalled to UI thread"""
        if self.view_callback:
            schedule_ui_update(self.view_callback.update_llm_status, "Complete!")
            await asyncio.sleep(1.5)  # Brief display
            schedule_ui_update(self.view_callback.hide_popup)
        self.current_mode = DictationPopupMode.HIDDEN

    async def _handle_llm_processing_failed(self, event_data) -> None:
        """Hide popup on failure - marshalled to UI thread"""
        if self.view_callback:
            schedule_ui_update(self.view_callback.hide_popup)
        self.current_mode = DictationPopupMode.HIDDEN

    async def _handle_llm_token_generated(self, event_data) -> None:
        """Display LLM tokens - marshalled to UI thread"""
        token = getattr(event_data, "token", "")
        self.logger.debug(f"CONTROLLER: Received token event: '{token}'")
        if token and self.view_callback:
            self.logger.debug("CONTROLLER: Scheduling UI update for token")
            schedule_ui_update(self.view_callback.append_llm_token, token)
        else:
            self.logger.warning("CONTROLLER: No token or no view_callback")

    async def _handle_smart_dictation_text_display(self, event_data) -> None:
        """Display dictation text - marshalled to UI thread"""
        text = getattr(event_data, "text", "")
        if text and self.view_callback:
            schedule_ui_update(self.view_callback.append_dictation_text, text)

    async def _handle_smart_dictation_remove_characters(self, event_data) -> None:
        """Remove characters from dictation text - marshalled to UI thread"""
        count = getattr(event_data, "count", 0)
        if count > 0 and self.view_callback:
            schedule_ui_update(self.view_callback.remove_dictation_characters, count)

    async def _handle_visual_dictation_started(self, event_data) -> None:
        """Show visual dictation popup - marshalled to UI thread"""
        self.current_mode = DictationPopupMode.VISUAL_DICTATION
        if self.view_callback:
            schedule_ui_update(self.view_callback.show_visual_dictation)

    async def _handle_visual_dictation_stopped(self, event_data) -> None:
        """Hide popup immediately after visual dictation stops - marshalled to UI thread"""
        if self.view_callback:
            schedule_ui_update(self.view_callback.hide_popup)
        self.current_mode = DictationPopupMode.HIDDEN

    async def _handle_partial_dictation_text(self, event_data) -> None:
        """Display partial (unstable) dictation text in gray - marshalled to UI thread"""
        text = getattr(event_data, "text", "")
        segment_id = getattr(event_data, "segment_id", "")
        if text and self.view_callback:
            schedule_ui_update(self.view_callback.display_partial_text, text, segment_id)

    async def _handle_final_dictation_text(self, event_data) -> None:
        """Display final (stable) dictation text in white - marshalled to UI thread"""
        text = getattr(event_data, "text", "")
        segment_id = getattr(event_data, "segment_id", "")
        if text and self.view_callback:
            schedule_ui_update(self.view_callback.display_final_text, text, segment_id)

    def cleanup(self) -> None:
        """Clean up resources"""
        super().cleanup()
