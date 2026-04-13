import logging
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import (
    AgenticPromptActionRequest,
    AgenticPromptListUpdatedEvent,
    AgenticPromptUpdatedEvent,
)
from vocalance.app.services.storage.storage_models import AgenticPrompt, AgenticPromptsData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Fix grammar errors, punctuation errors, and capitalization errors. Correct any misspelled words. "
    "Make the overall text more succinct and readable. Ensure to preserve all original meaning and content."
)


class AgenticPromptService:
    """CRUD + selection for agentic LLM prompts; persisted via ``StorageService`` (``RLock``)."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService) -> None:
        self.event_bus = event_bus
        self.config = config
        self._storage = storage
        self._lock = threading.RLock()
        self.prompts: Dict[str, AgenticPrompt] = {}
        self.current_prompt_id: Optional[str] = None
        self.default_prompt_text = _DEFAULT_PROMPT

    async def initialize(self) -> bool:
        try:
            await self._load_prompts()
            await self._ensure_default_prompt()
            if not self.current_prompt_id:
                default_prompt = self._get_default_prompt()
                if default_prompt:
                    self.current_prompt_id = default_prompt.id
            await self._publish_state()
            return True
        except Exception as e:
            logger.error("AgenticPromptService init: %s", e, exc_info=True)
            return False

    async def _ensure_default_prompt(self) -> None:
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
        await self._save_prompts()

    def _get_default_prompt(self) -> Optional[AgenticPrompt]:
        for prompt in self.prompts.values():
            if prompt.is_default:
                return prompt
        return None

    async def add_prompt(self, text: str, name: str) -> Optional[str]:
        with self._lock:
            prompt_id = str(uuid.uuid4())
            self.prompts[prompt_id] = AgenticPrompt(
                id=prompt_id,
                text=text.strip(),
                name=name.strip(),
                created_at=datetime.now().isoformat(),
                is_default=False,
            )
        await self._save_prompts()
        logger.info("Added prompt: %s", name)
        return prompt_id

    async def delete_prompt(self, prompt_id: str) -> bool:
        with self._lock:
            if prompt_id not in self.prompts:
                return False
            prompt = self.prompts[prompt_id]
            if prompt.is_default:
                logger.warning("Cannot delete default prompt")
                return False
            if self.current_prompt_id == prompt_id:
                remaining = [pid for pid in self.prompts if pid != prompt_id]
                self.current_prompt_id = remaining[0] if remaining else None
            del self.prompts[prompt_id]
            prompt_name = prompt.name
        await self._save_prompts()
        logger.info("Deleted prompt: %s", prompt_name)
        return True

    async def edit_prompt(self, prompt_id: str, name: str, text: str) -> bool:
        with self._lock:
            if prompt_id not in self.prompts:
                return False
            prompt = self.prompts[prompt_id]
            if prompt.is_default:
                logger.warning("Cannot edit default prompt")
                return False
            prompt.name = name.strip()
            prompt.text = text.strip()
        await self._save_prompts()
        logger.info("Edited prompt: %s", name)
        return True

    def set_current_prompt(self, prompt_id: str) -> bool:
        with self._lock:
            if prompt_id not in self.prompts:
                return False
            self.current_prompt_id = prompt_id
            prompt_name = self.prompts[prompt_id].name
        logger.info("Set current prompt: %s", prompt_name)
        return True

    def get_current_prompt(self) -> Optional[str]:
        with self._lock:
            if self.current_prompt_id and self.current_prompt_id in self.prompts:
                return self.prompts[self.current_prompt_id].text
            return None

    def get_current_prompt_data(self) -> Optional[AgenticPrompt]:
        with self._lock:
            if self.current_prompt_id and self.current_prompt_id in self.prompts:
                return self.prompts[self.current_prompt_id]
            return None

    def get_all_prompts(self) -> List[AgenticPrompt]:
        with self._lock:
            return list(self.prompts.values())

    async def _load_prompts(self) -> None:
        prompts_data = await self._storage.read(model_type=AgenticPromptsData)
        with self._lock:
            for prompt in prompts_data.prompts:
                self.prompts[prompt.id] = prompt
            self.current_prompt_id = prompts_data.current_prompt_id
        logger.info("Loaded %s prompts", len(self.prompts))

    async def _save_prompts(self) -> None:
        with self._lock:
            data = AgenticPromptsData(prompts=list(self.prompts.values()), current_prompt_id=self.current_prompt_id)
        await self._storage.write(data=data)

    async def _publish_state(self) -> None:
        current = self.get_current_prompt_data()
        if current:
            await self.event_bus.publish(AgenticPromptUpdatedEvent(prompt=current.text, prompt_id=current.id))
        await self.event_bus.publish(AgenticPromptListUpdatedEvent(prompts=[p.model_dump() for p in self.prompts.values()]))

    async def shutdown(self) -> None:
        await self._save_prompts()

    def setup_subscriptions(self) -> None:
        self.event_bus.subscribe(event_type=AgenticPromptActionRequest, handler=self._handle_agentic_prompt_action)

    async def _handle_agentic_prompt_action(self, action_request: AgenticPromptActionRequest) -> None:
        action = action_request.action
        if action == "add_prompt" and action_request.name and action_request.text:
            await self.add_prompt(action_request.text, action_request.name)
        elif action == "delete_prompt" and action_request.prompt_id:
            await self.delete_prompt(action_request.prompt_id)
        elif action == "edit_prompt" and action_request.prompt_id and action_request.name and action_request.text:
            await self.edit_prompt(action_request.prompt_id, action_request.name, action_request.text)
        elif action == "set_current_prompt" and action_request.prompt_id:
            self.set_current_prompt(action_request.prompt_id)
        elif action == "get_prompts":
            pass
        else:
            logger.warning("Unhandled agentic prompt action: %s", action)
        await self._publish_state()
