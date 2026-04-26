import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import (
    AgenticPromptListUpdatedEvent,
    AgenticPromptUiOperationEvent,
    AgenticPromptUpdatedEvent,
)
from vocalance.app.services.base_service import Service
from vocalance.app.services.storage.storage_models import AgenticPrompt, AgenticPromptsData
from vocalance.app.services.storage.storage_service import StorageService

_DEFAULT_PROMPT = (
    "Fix grammar errors, punctuation errors, and capitalization errors. Correct any misspelled words. "
    "Make the overall text more succinct and readable. Ensure to preserve all original meaning and content."
)


class AgenticPromptService(Service):
    """CRUD and current selection for agentic prompts persisted via ``StorageService``."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService) -> None:
        super().__init__(event_bus)
        self.config = config
        self.storage = storage
        self.prompt_lock = threading.RLock()
        self.prompts: Dict[str, AgenticPrompt] = {}
        self.current_prompt_id: Optional[str] = None
        self.default_prompt_text = _DEFAULT_PROMPT
        self.subscribe(AgenticPromptUiOperationEvent, self.handle_agentic_prompt_ui_operation)

    async def handle_agentic_prompt_ui_operation(self, event: AgenticPromptUiOperationEvent) -> None:
        op = event.op
        if op == "add":
            await self.add_prompt(event.prompt_text, event.name)
        elif op == "select":
            self.set_current_prompt(event.prompt_id)
            await self.publish_state()
        elif op == "delete":
            await self.delete_prompt(event.prompt_id)
        elif op == "edit":
            await self.edit_prompt(event.prompt_id, event.name, event.text)
        elif op == "publish_state":
            await self.publish_state()

    async def initialize(self) -> bool:
        await self.load_prompts()
        await self.ensure_default_prompt()
        if not self.current_prompt_id:
            default_prompt = self.get_default_prompt()
            if default_prompt:
                self.current_prompt_id = default_prompt.id
        await self.publish_state()
        return True

    async def ensure_default_prompt(self) -> None:
        if any(prompt.is_default for prompt in self.prompts.values()):
            return
        default_id = str(uuid.uuid4())
        self.prompts[default_id] = AgenticPrompt(
            id=default_id,
            text=self.default_prompt_text,
            name="Default",
            created_at=datetime.now().isoformat(),
            is_default=True,
        )
        await self.save_prompts()

    def get_default_prompt(self) -> Optional[AgenticPrompt]:
        with self.prompt_lock:
            for prompt in self.prompts.values():
                if prompt.is_default:
                    return prompt
        return None

    async def add_prompt(self, text: str, name: str) -> Optional[str]:
        with self.prompt_lock:
            prompt_id = str(uuid.uuid4())
            self.prompts[prompt_id] = AgenticPrompt(
                id=prompt_id,
                text=text.strip(),
                name=name.strip(),
                created_at=datetime.now().isoformat(),
                is_default=False,
            )
        await self.save_prompts()
        await self.publish_state()
        return prompt_id

    async def delete_prompt(self, prompt_id: str) -> bool:
        with self.prompt_lock:
            if prompt_id not in self.prompts:
                return False
            prompt = self.prompts[prompt_id]
            if prompt.is_default:
                return False
            if self.current_prompt_id == prompt_id:
                remaining = [pid for pid in self.prompts if pid != prompt_id]
                self.current_prompt_id = remaining[0] if remaining else None
            del self.prompts[prompt_id]
        await self.save_prompts()
        await self.publish_state()
        return True

    async def edit_prompt(self, prompt_id: str, name: str, text: str) -> bool:
        with self.prompt_lock:
            if prompt_id not in self.prompts:
                return False
            prompt = self.prompts[prompt_id]
            if prompt.is_default:
                return False
            self.prompts[prompt_id] = prompt.model_copy(update={"name": name.strip(), "text": text.strip()})
        await self.save_prompts()
        await self.publish_state()
        return True

    def set_current_prompt(self, prompt_id: str) -> bool:
        with self.prompt_lock:
            if prompt_id not in self.prompts:
                return False
            self.current_prompt_id = prompt_id
        return True

    def get_current_prompt(self) -> Optional[str]:
        with self.prompt_lock:
            if self.current_prompt_id and self.current_prompt_id in self.prompts:
                return self.prompts[self.current_prompt_id].text
            return None

    def get_current_prompt_data(self) -> Optional[AgenticPrompt]:
        with self.prompt_lock:
            if self.current_prompt_id and self.current_prompt_id in self.prompts:
                return self.prompts[self.current_prompt_id]
            return None

    def get_all_prompts(self) -> List[AgenticPrompt]:
        with self.prompt_lock:
            return list(self.prompts.values())

    async def load_prompts(self) -> None:
        prompts_data = await self.storage.read(model_type=AgenticPromptsData)
        with self.prompt_lock:
            self.prompts.clear()
            for prompt in prompts_data.prompts:
                self.prompts[prompt.id] = prompt
            self.current_prompt_id = prompts_data.current_prompt_id

    async def save_prompts(self) -> None:
        with self.prompt_lock:
            data = AgenticPromptsData(prompts=list(self.prompts.values()), current_prompt_id=self.current_prompt_id)
        await self.storage.write(data=data)

    async def publish_state(self) -> None:
        current = self.get_current_prompt_data()
        if current:
            await self.event_bus.publish(AgenticPromptUpdatedEvent(prompt=current.text, prompt_id=current.id))
        await self.event_bus.publish(AgenticPromptListUpdatedEvent(prompts=[p.model_dump() for p in self.prompts.values()]))

    async def shutdown(self) -> None:
        await self.save_prompts()
        await super().shutdown()
