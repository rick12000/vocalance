"""
Streamlined Agentic Prompt Service

Simplified prompt management using unified storage service.
"""
import asyncio
import logging
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from iris.event_bus import EventBus
from iris.config.app_config import GlobalAppConfig
from iris.events.dictation_events import AgenticPromptActionRequest, AgenticPromptUpdatedEvent, AgenticPromptListUpdatedEvent
from iris.services.storage.storage_adapters import StorageAdapterFactory

logger = logging.getLogger(__name__)

@dataclass
class AgenticPrompt:
    """Data class for agentic prompts"""
    id: str
    text: str
    name: str
    created_at: str
    is_default: bool = False

class AgenticPromptService:
    """Streamlined prompt management service using unified storage"""
    
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage_factory: StorageAdapterFactory):
        self.event_bus = event_bus
        self.config = config
        self.storage_adapter = storage_factory.get_agentic_prompt_adapter()
        
        # In-memory storage
        self.prompts: Dict[str, AgenticPrompt] = {}
        self.current_prompt_id: Optional[str] = None
        
        # Default prompt
        self.default_prompt_text = "Fix grammar, improve clarity, and make the text more succinct and readable while preserving all original meaning and content."
        
        logger.info("AgenticPromptService initialized")
    
    async def initialize(self) -> bool:
        """Initialize service"""
        try:
            await self._load_prompts()
            await self._ensure_default_prompt()
            
            # Set default as current if none selected
            if not self.current_prompt_id:
                default_prompt = self._get_default_prompt()
                if default_prompt:
                    self.current_prompt_id = default_prompt.id
            
            # Publish initial state
            await self._publish_prompts_updated()
            await self._publish_current_updated()
            
            logger.info("AgenticPromptService ready")
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False
    
    async def _ensure_default_prompt(self) -> None:
        """Ensure default prompt exists"""
        try:
            # Check if default exists
            if any(prompt.is_default for prompt in self.prompts.values()):
                return
            
            # Create default prompt
            default_id = str(uuid.uuid4())
            default_prompt = AgenticPrompt(
                id=default_id,
                text=self.default_prompt_text,
                name="Default",
                created_at=datetime.now().isoformat(),
                is_default=True
            )
            
            self.prompts[default_id] = default_prompt
            await self._save_prompts()
            logger.info("Created default prompt")
            
        except Exception as e:
            logger.error(f"Default prompt creation error: {e}", exc_info=True)
    
    def _get_default_prompt(self) -> Optional[AgenticPrompt]:
        """Get default prompt"""
        for prompt in self.prompts.values():
            if prompt.is_default:
                return prompt
        return None
    
    async def add_prompt(self, text: str, name: str) -> Optional[str]:
        """Add new prompt"""
        try:
            prompt_id = str(uuid.uuid4())
            prompt = AgenticPrompt(
                id=prompt_id,
                text=text.strip(),
                name=name.strip(),
                created_at=datetime.now().isoformat(),
                is_default=False
            )
            
            self.prompts[prompt_id] = prompt
            await self._save_prompts()
            
            logger.info(f"Added prompt: {name}")
            return prompt_id
            
        except Exception as e:
            logger.error(f"Add prompt error: {e}", exc_info=True)
            return None
    
    async def delete_prompt(self, prompt_id: str) -> bool:
        """Delete prompt"""
        try:
            if prompt_id not in self.prompts:
                return False
            
            prompt = self.prompts[prompt_id]
            
            # Don't delete default if it's the only one
            if prompt.is_default and len(self.prompts) == 1:
                logger.warning("Cannot delete only remaining prompt")
                return False
            
            # Switch current if deleting it
            if self.current_prompt_id == prompt_id:
                remaining = [pid for pid in self.prompts.keys() if pid != prompt_id]
                self.current_prompt_id = remaining[0] if remaining else None
            
            del self.prompts[prompt_id]
            await self._save_prompts()
            
            logger.info(f"Deleted prompt: {prompt.name}")
            return True
            
        except Exception as e:
            logger.error(f"Delete prompt error: {e}", exc_info=True)
            return False
    
    async def edit_prompt(self, prompt_id: str, name: str, text: str) -> bool:
        """Edit existing prompt"""
        try:
            if prompt_id not in self.prompts:
                logger.warning(f"Prompt ID {prompt_id} not found in prompts")
                return False
            prompt = self.prompts[prompt_id]
            prompt.name = name.strip()
            prompt.text = text.strip()
            await self._save_prompts()
            logger.info(f"Edited prompt: {name}")
            return True
        except Exception as e:
            logger.error(f"Edit prompt error: {e}", exc_info=True)
            return False
    
    async def set_current_prompt(self, prompt_id: str) -> bool:
        """Set current active prompt"""
        try:
            if prompt_id not in self.prompts:
                return False
            
            self.current_prompt_id = prompt_id
            await self._publish_current_updated()
            
            logger.info(f"Set current prompt: {self.prompts[prompt_id].name}")
            return True
            
        except Exception as e:
            logger.error(f"Set current prompt error: {e}", exc_info=True)
            return False
    
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
        """Load prompts from unified storage"""
        try:
            data = await self.storage_adapter.load_prompts()
            
            # Load prompts
            for prompt_data in data.get('prompts', []):
                prompt = AgenticPrompt(**prompt_data)
                self.prompts[prompt.id] = prompt
            
            # Load current
            self.current_prompt_id = data.get('current_prompt_id')
            
            logger.info(f"Loaded {len(self.prompts)} prompts")
            
        except Exception as e:
            logger.error(f"Load prompts error: {e}", exc_info=True)
    
    async def _save_prompts(self) -> None:
        """Save prompts to unified storage"""
        try:
            data = {
                'prompts': [asdict(prompt) for prompt in self.prompts.values()],
                'current_prompt_id': self.current_prompt_id
            }
            
            await self.storage_adapter.save_prompts(data)
            
        except Exception as e:
            logger.error(f"Save prompts error: {e}", exc_info=True)
    
    async def _publish_current_updated(self) -> None:
        """Publish current prompt updated event"""
        try:
            current = self.get_current_prompt_data()
            if current:
                event = AgenticPromptUpdatedEvent(
                    prompt=current.text,
                    prompt_id=current.id
                )
                await self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Event publishing error: {e}", exc_info=True)
    
    async def _publish_prompts_updated(self) -> None:
        """Publish prompts list updated event"""
        try:
            prompts_data = [asdict(prompt) for prompt in self.prompts.values()]
            event = AgenticPromptListUpdatedEvent(prompts=prompts_data)
            await self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Event publishing error: {e}", exc_info=True)
    
    async def shutdown(self) -> None:
        """Shutdown service"""
        try:
            await self._save_prompts()
            logger.info("AgenticPromptService shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True)
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        try:
            # Subscribe to prompt action requests from UI
            self.event_bus.subscribe(AgenticPromptActionRequest, self._handle_agentic_prompt_action)
            logger.info("AgenticPromptService subscriptions configured")
        except Exception as e:
            logger.error(f"Subscription setup error: {e}", exc_info=True)
    
    async def _handle_agentic_prompt_action(self, event_data) -> None:
        """Handle prompt action requests"""
        try:
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
            # Always publish updates after any action
            await self._publish_prompts_updated()
            await self._publish_current_updated()
        except Exception as e:
            logger.error(f"Prompt action error: {e}", exc_info=True) 