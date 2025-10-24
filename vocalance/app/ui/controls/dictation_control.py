from typing import Any, Dict, List, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.events.dictation_events import (
    AgenticPromptActionRequest,
    AgenticPromptListUpdatedEvent,
    AgenticPromptUpdatedEvent,
)
from vocalance.app.ui.controls.base_control import BaseController


class DictationController(BaseController):
    """Business logic controller for dictation functionality."""

    def __init__(self, event_bus, event_loop, logger, config: GlobalAppConfig):
        super().__init__(event_bus, event_loop, logger, "DictationController")
        self.config = config

        # State
        self.prompts = []
        self.current_prompt_id = None

        self.subscribe_to_events(
            [
                (AgenticPromptListUpdatedEvent, self._on_prompts_updated),
                (AgenticPromptUpdatedEvent, self._on_current_prompt_updated),
            ]
        )

    async def _on_prompts_updated(self, event):
        """Handle prompts list updated event."""
        self.prompts = getattr(event, "prompts", [])
        if self.view_callback:
            self.schedule_ui_update(self.view_callback.on_prompts_updated, self.prompts)
        self.notify_status(f"Loaded {len(self.prompts)} prompts.")

    async def _on_current_prompt_updated(self, event):
        """Handle current prompt updated event."""
        if hasattr(event, "prompt_id"):
            self.current_prompt_id = event.prompt_id
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_current_prompt_updated, self.current_prompt_id)
            self.notify_status("Current prompt updated.")

    def add_prompt(self, name: str, prompt_text: str) -> bool:
        """Add a new agentic prompt."""
        if not name.strip():
            self.notify_status("Please enter a prompt name.", True)
            return False

        if not prompt_text.strip():
            self.notify_status("Please enter prompt instructions.", True)
            return False

        event = AgenticPromptActionRequest(action="add_prompt", name=name, text=prompt_text)
        self.publish_event(event)
        self.notify_status(f"Added custom prompt: {name}")
        return True

    def select_prompt(self, prompt_id: str) -> None:
        """Select a prompt as the current one."""
        event = AgenticPromptActionRequest(action="set_current_prompt", prompt_id=prompt_id)
        self.publish_event(event)
        self.notify_status("Prompt selection updated.")

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt."""
        prompt_name = "Unknown"
        for prompt_data in self.prompts:
            if prompt_data.get("id") == prompt_id:
                prompt_name = prompt_data.get("name", "Unknown")
                break

        event = AgenticPromptActionRequest(action="delete_prompt", prompt_id=prompt_id)
        self.publish_event(event)
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

        event = AgenticPromptActionRequest(action="edit_prompt", prompt_id=prompt_id, name=name, text=text)
        self.publish_event(event)
        self.notify_status(f"Updated prompt: {name}")
        return True

    def refresh_prompts(self) -> None:
        """Refresh the prompts list."""
        event = AgenticPromptActionRequest(action="get_prompts")
        self.publish_event(event)
        self.notify_status("Requesting prompts...")

    def get_prompts(self) -> List[Dict[str, Any]]:
        """Get current prompts list."""
        return self.prompts

    def get_current_prompt_id(self) -> Optional[str]:
        """Get current prompt ID."""
        return self.current_prompt_id
