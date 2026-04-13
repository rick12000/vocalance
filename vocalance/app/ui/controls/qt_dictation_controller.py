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
    DictationSessionEvent,
    DictationStatusChangedEvent,
    FinalDictationTextEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingFailedEvent,
    LLMProcessingStartedEvent,
    PartialDictationTextEvent,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtDictationController(QtBaseController):
    """Business logic controller for dictation functionality."""

    prompts_loaded = Signal(list)
    current_prompt_updated = Signal(str)
    dictation_status_changed = Signal(bool, str)
    dictation_started = Signal(str)
    dictation_stopped = Signal(str, str)
    partial_text = Signal(str)
    final_text = Signal(str)
    llm_processing_started = Signal(str, str)
    llm_processing_completed = Signal(str, str)
    llm_processing_failed = Signal(str, str)
    operation_error = Signal(str)
    status_updated = Signal(str, bool)

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
    ) -> None:
        """Initialize dictation controller.

        Args:
            event_bus: Event bus for pub/sub.
            config: Global app configuration.
        """
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtDictationController"),
        )

        self.config = config
        self.prompts = []
        self.current_prompt_id = None

        self._subscribe_to_events()
        self.logger.debug("QtDictationController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to dictation-related events."""
        try:
            self.event_bus.subscribe(AgenticPromptListUpdatedEvent, self._on_prompts_updated)
            self.event_bus.subscribe(AgenticPromptUpdatedEvent, self._on_current_prompt_updated)
            self.event_bus.subscribe(DictationStatusChangedEvent, self._on_dictation_status_changed)
            self.event_bus.subscribe(DictationSessionEvent, self._on_dictation_session)
            self.event_bus.subscribe(PartialDictationTextEvent, self._on_partial_text)
            self.event_bus.subscribe(FinalDictationTextEvent, self._on_final_text)
            self.event_bus.subscribe(LLMProcessingStartedEvent, self._on_llm_started)
            self.event_bus.subscribe(LLMProcessingCompletedEvent, self._on_llm_completed)
            self.event_bus.subscribe(LLMProcessingFailedEvent, self._on_llm_failed)
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def _on_prompts_updated(self, list_update: AgenticPromptListUpdatedEvent) -> None:
        """Handle prompts list updated event."""
        self.prompts = list_update.prompts
        self.prompts_loaded.emit(self.prompts)
        self.notify_status(f"Loaded {len(self.prompts)} prompts.")

    def _on_current_prompt_updated(self, selection: AgenticPromptUpdatedEvent) -> None:
        """Handle current prompt updated event."""
        self.current_prompt_id = selection.prompt_id
        self.current_prompt_updated.emit(self.current_prompt_id)
        self.notify_status("Current prompt updated.")

    def _on_dictation_status_changed(self, status: DictationStatusChangedEvent) -> None:
        """Handle dictation status changed event."""
        self.dictation_status_changed.emit(status.is_active, status.mode)

    def _on_dictation_session(self, session: DictationSessionEvent) -> None:
        """Handle dictation session events."""
        if session.state == "started":
            self.dictation_started.emit(session.mode)
        elif session.state == "stopped":
            if session.mode in ("smart", "amend"):
                self.dictation_stopped.emit(session.mode, session.raw_text or "")
            elif session.mode in ("visual", "hidden"):
                self.dictation_stopped.emit(session.mode, session.accumulated_text or "")

    def _on_partial_text(self, partial: PartialDictationTextEvent) -> None:
        """Handle partial dictation text event."""
        self.partial_text.emit(partial.text)

    def _on_final_text(self, final: FinalDictationTextEvent) -> None:
        """Handle final dictation text event."""
        self.final_text.emit(final.text)

    def _on_llm_started(self, started: LLMProcessingStartedEvent) -> None:
        """Handle LLM processing started event."""
        self.llm_processing_started.emit(started.raw_text, started.agentic_prompt)

    def _on_llm_completed(self, completed: LLMProcessingCompletedEvent) -> None:
        """Handle LLM processing completed event."""
        self.llm_processing_completed.emit(completed.processed_text, completed.agentic_prompt)

    def _on_llm_failed(self, failed: LLMProcessingFailedEvent) -> None:
        """Handle LLM processing failed event."""
        self.llm_processing_failed.emit(failed.error_message, failed.original_text)

    def add_prompt(self, name: str, prompt_text: str) -> bool:
        """Publish an add-prompt action request.

        Args:
            name: Display name for the prompt.
            prompt_text: Prompt instruction text.

        Returns:
            False if validation fails, True if the request was published.
        """
        if not name.strip():
            self.notify_status("Please enter a prompt name.", True)
            return False
        if not prompt_text.strip():
            self.notify_status("Please enter prompt instructions.", True)
            return False

        asyncio.ensure_future(self.event_bus.publish(AgenticPromptActionRequest(action="add_prompt", name=name, text=prompt_text)))
        self.notify_status(f"Added custom prompt: {name}")
        return True

    def select_prompt(self, prompt_id: str) -> None:
        """Publish a set-current-prompt action request.

        Args:
            prompt_id: ID of the prompt to select.
        """
        asyncio.ensure_future(self.event_bus.publish(AgenticPromptActionRequest(action="set_current_prompt", prompt_id=prompt_id)))
        self.notify_status("Prompt selection updated.")

    def is_default_prompt(self, prompt_id: str) -> bool:
        """Return True if the given prompt ID is the default prompt."""
        for prompt_data in self.prompts:
            if prompt_data.get("id") == prompt_id:
                return prompt_data.get("is_default", False)
        return False

    def delete_prompt(self, prompt_id: str) -> bool:
        """Publish a delete-prompt action request.

        Args:
            prompt_id: ID of the prompt to delete.

        Returns:
            False if the prompt is the default or not found, True otherwise.
        """
        prompt_name = "Unknown"
        for prompt_data in self.prompts:
            if prompt_data.get("id") == prompt_id:
                prompt_name = prompt_data.get("name", "Unknown")
                if prompt_data.get("is_default", False):
                    self.notify_status("The default prompt cannot be deleted.", True)
                    return False
                break

        asyncio.ensure_future(self.event_bus.publish(AgenticPromptActionRequest(action="delete_prompt", prompt_id=prompt_id)))
        self.notify_status(f"Deleted prompt: {prompt_name}")
        return True

    def edit_prompt(self, prompt_id: str, name: str, text: str) -> bool:
        """Publish an edit-prompt action request.

        Args:
            prompt_id: ID of the prompt to edit.
            name: New display name.
            text: New prompt instruction text.

        Returns:
            False if validation fails or prompt is default, True otherwise.
        """
        if not name.strip():
            self.notify_status("Please enter a prompt name.", True)
            return False
        if not text.strip():
            self.notify_status("Please enter prompt instructions.", True)
            return False
        if self.is_default_prompt(prompt_id):
            self.notify_status("The default prompt cannot be edited.", True)
            return False

        asyncio.ensure_future(
            self.event_bus.publish(AgenticPromptActionRequest(action="edit_prompt", prompt_id=prompt_id, name=name, text=text))
        )
        self.notify_status(f"Updated prompt: {name}")
        return True

    def refresh_prompts(self) -> None:
        """Publish a get-prompts action request."""
        asyncio.ensure_future(self.event_bus.publish(AgenticPromptActionRequest(action="get_prompts")))
        self.notify_status("Requesting prompts...")

    def get_prompts(self) -> List[Dict[str, Any]]:
        """Return the cached prompts list."""
        return self.prompts

    def get_current_prompt_id(self) -> Optional[str]:
        """Return the currently selected prompt ID."""
        return self.current_prompt_id

    def notify_status(self, message: str, is_error: bool = False) -> None:
        """Emit a status update signal.

        Args:
            message: Status message text.
            is_error: True if this represents an error condition.
        """
        self.status_updated.emit(message, is_error)
        if is_error:
            self.operation_error.emit(message)

    def cleanup(self) -> None:
        """Unsubscribe from all events and release resources."""
        try:
            self.event_bus.unsubscribe(AgenticPromptListUpdatedEvent, self._on_prompts_updated)
            self.event_bus.unsubscribe(AgenticPromptUpdatedEvent, self._on_current_prompt_updated)
            self.event_bus.unsubscribe(DictationStatusChangedEvent, self._on_dictation_status_changed)
            self.event_bus.unsubscribe(DictationSessionEvent, self._on_dictation_session)
            self.event_bus.unsubscribe(PartialDictationTextEvent, self._on_partial_text)
            self.event_bus.unsubscribe(FinalDictationTextEvent, self._on_final_text)
            self.event_bus.unsubscribe(LLMProcessingStartedEvent, self._on_llm_started)
            self.event_bus.unsubscribe(LLMProcessingCompletedEvent, self._on_llm_completed)
            self.event_bus.unsubscribe(LLMProcessingFailedEvent, self._on_llm_failed)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
