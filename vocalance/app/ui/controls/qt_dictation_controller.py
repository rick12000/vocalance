"""Qt-based dictation controller - EXACT LEGACY MATCH.

Business logic controller for dictation functionality with agentic prompts management.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import (
    AgenticPromptActionRequest,
    AgenticPromptListUpdatedEvent,
    AgenticPromptUpdatedEvent,
    DictationStatusChangedEvent,
    FinalDictationTextEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingFailedEvent,
    LLMProcessingStartedEvent,
    PartialDictationTextEvent,
    SmartDictationStartedEvent,
    SmartDictationStoppedEvent,
    VisualDictationStartedEvent,
    VisualDictationStoppedEvent,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtDictationController(QtBaseController):
    """Business logic controller for dictation functionality."""

    # Signals for dictation operations
    prompts_loaded = Signal(list)  # List of prompt dicts
    current_prompt_updated = Signal(str)  # prompt_id
    dictation_status_changed = Signal(bool, str)  # is_active, mode
    dictation_started = Signal(str)  # mode
    dictation_stopped = Signal(str, str)  # mode, text
    partial_text = Signal(str)  # text
    final_text = Signal(str)  # text
    llm_processing_started = Signal(str, str)  # raw_text, prompt
    llm_processing_completed = Signal(str, str)  # processed_text, prompt
    llm_processing_failed = Signal(str, str)  # error_message, original_text
    operation_error = Signal(str)
    status_updated = Signal(str, bool)  # message, is_error

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        dictation_service,
        config: GlobalAppConfig,
        main_window,
    ):
        """Initialize dictation controller.

        Args:
            event_bus: Event bus for pub/sub.
            event_loop: Asyncio event loop.
            dictation_service: Dictation coordinator instance.
            config: Global app configuration.
            main_window: Main window reference.
        """
        super().__init__(
            event_bus=event_bus,
            event_loop=event_loop,
            logger=logging.getLogger("QtDictationController"),
        )

        self.dictation_service = dictation_service
        self.config = config
        self.main_window = main_window

        # State matching legacy
        self.prompts = []
        self.current_prompt_id = None

        # Subscribe to dictation events
        self._subscribe_to_events()

        self.logger.debug("QtDictationController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to dictation-related events using exact legacy event types."""
        try:
            # Prompts management
            self.event_bus.subscribe(AgenticPromptListUpdatedEvent, self._on_prompts_updated)
            self.event_bus.subscribe(AgenticPromptUpdatedEvent, self._on_current_prompt_updated)

            # Dictation status
            self.event_bus.subscribe(DictationStatusChangedEvent, self._on_dictation_status_changed)
            self.event_bus.subscribe(SmartDictationStartedEvent, self._on_smart_started)
            self.event_bus.subscribe(SmartDictationStoppedEvent, self._on_smart_stopped)
            self.event_bus.subscribe(VisualDictationStartedEvent, self._on_visual_started)
            self.event_bus.subscribe(VisualDictationStoppedEvent, self._on_visual_stopped)

            # Text events
            self.event_bus.subscribe(PartialDictationTextEvent, self._on_partial_text)
            self.event_bus.subscribe(FinalDictationTextEvent, self._on_final_text)

            # LLM processing
            self.event_bus.subscribe(LLMProcessingStartedEvent, self._on_llm_started)
            self.event_bus.subscribe(LLMProcessingCompletedEvent, self._on_llm_completed)
            self.event_bus.subscribe(LLMProcessingFailedEvent, self._on_llm_failed)

            self.logger.debug("Subscribed to dictation events (legacy types)")
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    # --- Event Handlers ---

    async def _on_prompts_updated(self, event):
        """Handle prompts list updated event."""
        self.prompts = getattr(event, "prompts", [])
        self.prompts_loaded.emit(self.prompts)
        self.notify_status(f"Loaded {len(self.prompts)} prompts.")

    async def _on_current_prompt_updated(self, event):
        """Handle current prompt updated event."""
        if hasattr(event, "prompt_id"):
            self.current_prompt_id = event.prompt_id
            self.current_prompt_updated.emit(self.current_prompt_id)
            self.notify_status("Current prompt updated.")

    async def _on_dictation_status_changed(self, event):
        """Handle dictation status changed event."""
        is_active = getattr(event, "is_active", False)
        mode = getattr(event, "mode", "inactive")
        self.dictation_status_changed.emit(is_active, mode)

    async def _on_smart_started(self, event):
        """Handle smart dictation started event."""
        self.dictation_started.emit("smart")

    async def _on_smart_stopped(self, event):
        """Handle smart dictation stopped event."""
        raw_text = getattr(event, "raw_text", "")
        self.dictation_stopped.emit("smart", raw_text)

    async def _on_visual_started(self, event):
        """Handle visual dictation started event."""
        self.dictation_started.emit("visual")

    async def _on_visual_stopped(self, event):
        """Handle visual dictation stopped event."""
        text = getattr(event, "accumulated_text", "")
        self.dictation_stopped.emit("visual", text)

    async def _on_partial_text(self, event):
        """Handle partial dictation text event."""
        text = getattr(event, "text", "")
        self.partial_text.emit(text)

    async def _on_final_text(self, event):
        """Handle final dictation text event."""
        text = getattr(event, "text", "")
        self.final_text.emit(text)

    async def _on_llm_started(self, event):
        """Handle LLM processing started event."""
        raw_text = getattr(event, "raw_text", "")
        prompt = getattr(event, "agentic_prompt", "")
        self.llm_processing_started.emit(raw_text, prompt)

    async def _on_llm_completed(self, event):
        """Handle LLM processing completed event."""
        processed_text = getattr(event, "processed_text", "")
        prompt = getattr(event, "agentic_prompt", "")
        self.llm_processing_completed.emit(processed_text, prompt)

    async def _on_llm_failed(self, event):
        """Handle LLM processing failed event."""
        error_message = getattr(event, "error_message", "Unknown error")
        original_text = getattr(event, "original_text", "")
        self.llm_processing_failed.emit(error_message, original_text)

    # --- Public Methods (Publish Events) ---

    def add_prompt(self, name: str, prompt_text: str) -> bool:
        """Add a new agentic prompt."""
        if not name.strip():
            self.notify_status("Please enter a prompt name.", True)
            return False

        if not prompt_text.strip():
            self.notify_status("Please enter prompt instructions.", True)
            return False

        event = AgenticPromptActionRequest(action="add_prompt", name=name, text=prompt_text)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)
        self.notify_status(f"Added custom prompt: {name}")
        return True

    def select_prompt(self, prompt_id: str) -> None:
        """Select a prompt as the current one."""
        event = AgenticPromptActionRequest(action="set_current_prompt", prompt_id=prompt_id)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)
        self.notify_status("Prompt selection updated.")

    def is_default_prompt(self, prompt_id: str) -> bool:
        """Check if a prompt is the default prompt."""
        for prompt_data in self.prompts:
            if prompt_data.get("id") == prompt_id:
                return prompt_data.get("is_default", False)
        return False

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt."""
        prompt_name = "Unknown"
        is_default = False
        for prompt_data in self.prompts:
            if prompt_data.get("id") == prompt_id:
                prompt_name = prompt_data.get("name", "Unknown")
                is_default = prompt_data.get("is_default", False)
                break

        if is_default:
            self.notify_status("The default prompt cannot be deleted.", True)
            return False

        event = AgenticPromptActionRequest(action="delete_prompt", prompt_id=prompt_id)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)
        self.notify_status(f"Deleted prompt: {prompt_name}")
        return True

    def edit_prompt(self, prompt_id: str, name: str, text: str) -> bool:
        """Edit an existing prompt."""
        if not name.strip():
            self.notify_status("Please enter a prompt name.", True)
            return False

        if not text.strip():
            self.notify_status("Please enter prompt instructions.", True)
            return False

        if self.is_default_prompt(prompt_id):
            self.notify_status("The default prompt cannot be edited.", True)
            return False

        event = AgenticPromptActionRequest(action="edit_prompt", prompt_id=prompt_id, name=name, text=text)
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)
        self.notify_status(f"Updated prompt: {name}")
        return True

    def refresh_prompts(self) -> None:
        """Refresh the prompts list."""
        event = AgenticPromptActionRequest(action="get_prompts")
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), self.event_loop)
        self.notify_status("Requesting prompts...")

    # --- Getters for View ---

    def get_prompts(self) -> List[Dict[str, Any]]:
        """Get current prompts list."""
        return self.prompts

    def get_current_prompt_id(self) -> Optional[str]:
        """Get current prompt ID."""
        return self.current_prompt_id

    def notify_status(self, message: str, is_error: bool = False) -> None:
        """Notify status message."""
        self.status_updated.emit(message, is_error)
        if is_error:
            self.operation_error.emit(message)

    def cleanup(self) -> None:
        """Clean up controller resources."""
        try:
            self.event_bus.unsubscribe(AgenticPromptListUpdatedEvent, self._on_prompts_updated)
            self.event_bus.unsubscribe(AgenticPromptUpdatedEvent, self._on_current_prompt_updated)
            self.event_bus.unsubscribe(DictationStatusChangedEvent, self._on_dictation_status_changed)
            self.event_bus.unsubscribe(SmartDictationStartedEvent, self._on_smart_started)
            self.event_bus.unsubscribe(SmartDictationStoppedEvent, self._on_smart_stopped)
            self.event_bus.unsubscribe(VisualDictationStartedEvent, self._on_visual_started)
            self.event_bus.unsubscribe(VisualDictationStoppedEvent, self._on_visual_stopped)
            self.event_bus.unsubscribe(PartialDictationTextEvent, self._on_partial_text)
            self.event_bus.unsubscribe(FinalDictationTextEvent, self._on_final_text)
            self.event_bus.unsubscribe(LLMProcessingStartedEvent, self._on_llm_started)
            self.event_bus.unsubscribe(LLMProcessingCompletedEvent, self._on_llm_completed)
            self.event_bus.unsubscribe(LLMProcessingFailedEvent, self._on_llm_failed)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
