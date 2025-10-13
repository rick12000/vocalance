import logging
import asyncio
from typing import Optional
from enum import Enum
from iris.app.ui.controls.base_control import BaseController
from iris.app.ui.utils.ui_thread_utils import schedule_ui_update
from iris.app.events.dictation_events import (
    DictationStatusChangedEvent,
    LLMProcessingStartedEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingFailedEvent,
    LLMTokenGeneratedEvent,
    SmartDictationStartedEvent,
    SmartDictationStoppedEvent,
    SmartDictationTextDisplayEvent,
    LLMProcessingReadyEvent
)


class DictationPopupMode(Enum):
    """Modes for the dictation popup window."""
    HIDDEN = "hidden"
    SIMPLE_LISTENING = "simple_listening"
    SMART_DICTATION = "smart_dictation"
    LLM_PROCESSING = "llm_processing"


class DictationPopupController(BaseController):
    """Simplified controller for dictation popup"""
    
    def __init__(self, event_bus, event_loop, logger):
        super().__init__(event_bus, event_loop, logger, "DictationPopupController")
        
        self.current_mode = DictationPopupMode.HIDDEN
        
        self.subscribe_to_events([
            (DictationStatusChangedEvent, self._handle_dictation_status_changed),
            (SmartDictationStartedEvent, self._handle_smart_dictation_started),
            (SmartDictationStoppedEvent, self._handle_smart_dictation_stopped),
            (LLMProcessingStartedEvent, self._handle_llm_processing_started),
            (LLMProcessingCompletedEvent, self._handle_llm_processing_completed),
            (LLMProcessingFailedEvent, self._handle_llm_processing_failed),
            (LLMTokenGeneratedEvent, self._handle_llm_token_generated),
            (SmartDictationTextDisplayEvent, self._handle_smart_dictation_text_display),
        ])

    async def _handle_dictation_status_changed(self, event_data) -> None:
        """Handle normal dictation status changes (not smart dictation) - marshalled to UI thread"""
        if event_data.show_ui and event_data.is_active:
            # Only handle non-smart dictation modes
            if event_data.mode != "smart":
                self.current_mode = DictationPopupMode.SIMPLE_LISTENING
                if self.view_callback:
                    schedule_ui_update(self.view_callback.show_simple_listening, event_data.mode, event_data.stop_command)
        else:
            # Hide popup for non-smart modes
            if not event_data.is_active and self.current_mode == DictationPopupMode.SIMPLE_LISTENING:
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
        session_id = getattr(event_data, 'session_id', None) or 'default'
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
        token = getattr(event_data, 'token', '')
        self.logger.debug(f"CONTROLLER: Received token event: '{token}'")
        if token and self.view_callback:
            self.logger.debug(f"CONTROLLER: Scheduling UI update for token")
            schedule_ui_update(self.view_callback.append_llm_token, token)
        else:
            self.logger.warning(f"CONTROLLER: No token or no view_callback")

    async def _handle_smart_dictation_text_display(self, event_data) -> None:
        """Display dictation text - marshalled to UI thread"""
        text = getattr(event_data, 'text', '')
        if text and self.view_callback:
            schedule_ui_update(self.view_callback.append_dictation_text, text)

    def cleanup(self) -> None:
        """Clean up resources"""
        super().cleanup() 