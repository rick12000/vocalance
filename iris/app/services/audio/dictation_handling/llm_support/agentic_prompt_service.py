"""
Streamlined Agentic Prompt Service

Simplified prompt management using unified storage service.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from iris.app.config.app_config import GlobalAppConfig
from iris.app.event_bus import EventBus
from iris.app.events.dictation_events import AgenticPromptActionRequest, AgenticPromptListUpdatedEvent, AgenticPromptUpdatedEvent
from iris.app.services.storage.storage_models import AgenticPrompt, AgenticPromptsData
from iris.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class AgenticPromptService:
    """Streamlined prompt management service using unified storage"""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService):
        self.event_bus = event_bus
        self.config = config
        self._storage = storage

        # In-memory storage
        self.prompts: Dict[str, AgenticPrompt] = {}
        self.current_prompt_id: Optional[str] = None

        # Default prompt
        self.default_prompt_text = "Fix grammar, improve clarity, and make the text more succinct and readable while preserving all original meaning and content. Output ONLY the processed text."

        logger.info("AgenticPromptService initialized")

    async def initialize(self) -> bool:
        """Initialize service"""
        try:
            await self._load_prompts()
            await self._ensure_default_prompt()

            if not self.current_prompt_id:
                default_prompt = self._get_default_prompt()
                if default_prompt:
                    self.current_prompt_id = default_prompt.id

            await self._publish_state()

            logger.info("AgenticPromptService ready")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False

    async def _ensure_default_prompt(self) -> None:
        """Ensure default prompt exists"""
        if any(prompt.is_default for prompt in self.prompts.values()):
            return

        default_id = str(uuid.uuid4())
        default_prompt = AgenticPrompt(
            id=default_id, text=self.default_prompt_text, name="Default", created_at=datetime.now().isoformat(), is_default=True
        )

        self.prompts[default_id] = default_prompt
        await self._save_prompts()
        logger.info("Created default prompt")

    def _get_default_prompt(self) -> Optional[AgenticPrompt]:
        """Get default prompt"""
        for prompt in self.prompts.values():
            if prompt.is_default:
                return prompt
        return None

    async def add_prompt(self, text: str, name: str) -> Optional[str]:
        """Add new prompt"""
        prompt_id = str(uuid.uuid4())
        prompt = AgenticPrompt(
            id=prompt_id, text=text.strip(), name=name.strip(), created_at=datetime.now().isoformat(), is_default=False
        )

        self.prompts[prompt_id] = prompt
        await self._save_prompts()

        logger.info(f"Added prompt: {name}")
        return prompt_id

    async def delete_prompt(self, prompt_id: str) -> bool:
        """Delete prompt"""
        if prompt_id not in self.prompts:
            return False

        prompt = self.prompts[prompt_id]

        if prompt.is_default and len(self.prompts) == 1:
            logger.warning("Cannot delete only remaining prompt")
            return False

        if self.current_prompt_id == prompt_id:
            remaining = [pid for pid in self.prompts.keys() if pid != prompt_id]
            self.current_prompt_id = remaining[0] if remaining else None

        del self.prompts[prompt_id]
        await self._save_prompts()

        logger.info(f"Deleted prompt: {prompt.name}")
        return True

    async def edit_prompt(self, prompt_id: str, name: str, text: str) -> bool:
        """Edit existing prompt"""
        if prompt_id not in self.prompts:
            logger.warning(f"Prompt ID {prompt_id} not found")
            return False

        prompt = self.prompts[prompt_id]
        prompt.name = name.strip()
        prompt.text = text.strip()
        await self._save_prompts()
        logger.info(f"Edited prompt: {name}")
        return True

    async def set_current_prompt(self, prompt_id: str) -> bool:
        """Set current active prompt"""
        if prompt_id not in self.prompts:
            return False

        self.current_prompt_id = prompt_id
        logger.info(f"Set current prompt: {self.prompts[prompt_id].name}")
        return True

    def get_current_prompt(self) -> Optional[str]:
        """Get current prompt text"""
        if self.current_prompt_id and self.current_prompt_id in self.prompts:
            return self.prompts[self.current_prompt_id].text
        return None

    def get_current_prompt_data(self) -> Optional[AgenticPrompt]:
        """Get current prompt data"""
        if self.current_prompt_id and self.current_prompt_id in self.prompts:
            return self.prompts[self.current_prompt_id]
        return None

    def get_all_prompts(self) -> List[AgenticPrompt]:
        """Get all prompts"""
        return list(self.prompts.values())

    async def _load_prompts(self) -> None:
        """Load prompts from storage"""
        prompts_data = await self._storage.read(model_type=AgenticPromptsData)

        for prompt in prompts_data.prompts:
            self.prompts[prompt.id] = prompt

        self.current_prompt_id = prompts_data.current_prompt_id
        logger.info(f"Loaded {len(self.prompts)} prompts")

    async def _save_prompts(self) -> None:
        """Save prompts to storage"""
        prompts_data = AgenticPromptsData(prompts=list(self.prompts.values()), current_prompt_id=self.current_prompt_id)
        await self._storage.write(data=prompts_data)

    async def _publish_state(self) -> None:
        """Publish both current prompt and list events"""
        current = self.get_current_prompt_data()
        if current:
            await self.event_bus.publish(AgenticPromptUpdatedEvent(prompt=current.text, prompt_id=current.id))

        prompts_data = [prompt.model_dump() for prompt in self.prompts.values()]
        await self.event_bus.publish(AgenticPromptListUpdatedEvent(prompts=prompts_data))

    async def shutdown(self) -> None:
        """Shutdown service"""
        await self._save_prompts()
        logger.info("AgenticPromptService shutdown complete")

    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        self.event_bus.subscribe(event_type=AgenticPromptActionRequest, handler=self._handle_agentic_prompt_action)
        logger.info("AgenticPromptService subscriptions configured")

    async def _handle_agentic_prompt_action(self, event_data) -> None:
        """Handle prompt action requests"""
        action = event_data.action

        if action == "add_prompt" and event_data.name and event_data.text:
            await self.add_prompt(event_data.text, event_data.name)
        elif action == "delete_prompt" and event_data.prompt_id:
            await self.delete_prompt(event_data.prompt_id)
        elif action == "edit_prompt" and event_data.prompt_id and event_data.name and event_data.text:
            await self.edit_prompt(event_data.prompt_id, event_data.name, event_data.text)
        elif action == "set_current_prompt" and event_data.prompt_id:
            await self.set_current_prompt(event_data.prompt_id)
        elif action == "get_prompts":
            pass
        else:
            logger.warning(f"Unhandled action or missing parameters: {action}")

        await self._publish_state()
