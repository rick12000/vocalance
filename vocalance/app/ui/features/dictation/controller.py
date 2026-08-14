import asyncio
import logging
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import (
    AgenticPromptListUpdatedEvent,
    AgenticPromptUiOperationEvent,
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

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger("QtDictationController"))
        self.config = config
        self.prompts: List[Dict[str, Any]] = []
        self.current_prompt_id: Optional[str] = None
        self.subscribe(AgenticPromptListUpdatedEvent, self.on_prompts_updated)
        self.subscribe(AgenticPromptUpdatedEvent, self.on_current_prompt_updated)
        self.subscribe(DictationStatusChangedEvent, self.on_dictation_status_changed)
        self.subscribe(DictationSessionEvent, self.on_dictation_session)
        self.subscribe(PartialDictationTextEvent, self.on_partial_text)
        self.subscribe(FinalDictationTextEvent, self.on_final_text)
        self.subscribe(LLMProcessingStartedEvent, self.on_llm_started)
        self.subscribe(LLMProcessingCompletedEvent, self.on_llm_completed)
        self.subscribe(LLMProcessingFailedEvent, self.on_llm_failed)

    def prompt_ui(self, op: str, **kwargs: Any) -> None:
        asyncio.create_task(self.event_bus.publish(AgenticPromptUiOperationEvent(op=op, **kwargs)))

    def on_prompts_updated(self, list_update: AgenticPromptListUpdatedEvent) -> None:
        self.prompts = list_update.prompts
        self.prompts_loaded.emit(self.prompts)
        self.notify_status(f"Loaded {len(self.prompts)} prompts.")

    def on_current_prompt_updated(self, selection: AgenticPromptUpdatedEvent) -> None:
        self.current_prompt_id = selection.prompt_id
        self.current_prompt_updated.emit(self.current_prompt_id)
        self.notify_status("Current prompt updated.")

    def on_dictation_status_changed(self, status: DictationStatusChangedEvent) -> None:
        self.dictation_status_changed.emit(status.is_active, status.mode)

    def on_dictation_session(self, session: DictationSessionEvent) -> None:
        if session.state == "started":
            self.dictation_started.emit(session.mode)
        elif session.state == "stopped":
            if session.mode in ("smart", "amend"):
                self.dictation_stopped.emit(session.mode, session.raw_text or "")
            elif session.mode in ("visual", "hidden"):
                self.dictation_stopped.emit(session.mode, session.accumulated_text or "")

    def on_partial_text(self, partial: PartialDictationTextEvent) -> None:
        self.partial_text.emit(partial.text)

    def on_final_text(self, final: FinalDictationTextEvent) -> None:
        self.final_text.emit(final.text)

    def on_llm_started(self, started: LLMProcessingStartedEvent) -> None:
        self.llm_processing_started.emit(started.raw_text, started.agentic_prompt)

    def on_llm_completed(self, completed: LLMProcessingCompletedEvent) -> None:
        self.llm_processing_completed.emit(completed.processed_text, completed.agentic_prompt)

    def on_llm_failed(self, failed: LLMProcessingFailedEvent) -> None:
        self.llm_processing_failed.emit(failed.error_message, failed.original_text)

    def add_prompt(self, name: str, prompt_text: str) -> bool:
        if not name.strip():
            self.notify_status("Please enter a prompt name.", True)
            return False
        if not prompt_text.strip():
            self.notify_status("Please enter prompt instructions.", True)
            return False
        self.prompt_ui("add", name=name, prompt_text=prompt_text)
        self.notify_status(f"Added custom prompt: {name}")
        return True

    def select_prompt(self, prompt_id: str) -> None:
        self.prompt_ui("select", prompt_id=prompt_id)
        self.notify_status("Prompt selection updated.")

    def _find_prompt_data(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        return next((p for p in self.prompts if p.get("id") == prompt_id), None)

    def is_protected_prompt(self, prompt_id: str) -> bool:
        prompt = self._find_prompt_data(prompt_id)
        if prompt is None:
            return False
        return bool(prompt.get("is_default", False)) or bool(prompt.get("system_key"))

    def delete_prompt(self, prompt_id: str) -> bool:
        prompt = self._find_prompt_data(prompt_id)
        if prompt is not None and self.is_protected_prompt(prompt_id):
            self.notify_status("This prompt cannot be deleted.", True)
            return False
        prompt_name = prompt.get("name", "Unknown") if prompt is not None else "Unknown"
        self.prompt_ui("delete", prompt_id=prompt_id)
        self.notify_status(f"Deleted prompt: {prompt_name}")
        return True

    def edit_prompt(self, prompt_id: str, name: str, text: str) -> bool:
        if not name.strip():
            self.notify_status("Please enter a prompt name.", True)
            return False
        if not text.strip():
            self.notify_status("Please enter prompt instructions.", True)
            return False
        prompt = self._find_prompt_data(prompt_id)
        if prompt is not None and prompt.get("is_default", False):
            self.notify_status("The default prompt cannot be edited.", True)
            return False
        self.prompt_ui("edit", prompt_id=prompt_id, name=name, text=text)
        self.notify_status(f"Updated prompt: {name}")
        return True

    def refresh_prompts(self) -> None:
        self.prompt_ui("publish_state")
        self.notify_status("Requesting prompts...")

    def get_prompts(self) -> List[Dict[str, Any]]:
        return self.prompts

    def get_current_prompt_id(self) -> Optional[str]:
        return self.current_prompt_id

    def notify_status(self, message: str, is_error: bool = False) -> None:
        self.emit_status(message, is_error)
        if is_error:
            self.operation_error.emit(message)
