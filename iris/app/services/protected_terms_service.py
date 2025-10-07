"""
Protected Terms Service

Centralized service for managing protected terms and command naming validation.
Fetches live data from mark and sound services to ensure no naming conflicts.
"""

import logging
from typing import Set, Optional
from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.automation_command_registry import AutomationCommandRegistry
from iris.app.services.storage.unified_storage_service import UnifiedStorageService, UnifiedStorageServiceExtensions, read_sound_mappings

logger = logging.getLogger(__name__)


class ProtectedTermsService:
    """
    Centralized service for managing protected terms and validation.
    Fetches live data from storage to ensure no naming conflicts.
    """
    
    def __init__(self, app_config: GlobalAppConfig, storage: UnifiedStorageService):
        self._app_config = app_config
        self._storage = storage
        
    async def get_all_protected_terms(self) -> Set[str]:
        """
        Get all protected terms that cannot be used as custom command names.
        Includes automation commands, system triggers, live mark names, and live sound names.
        """
        protected = set()
        
        # Add automation command phrases
        protected.update(phrase.lower().strip() for phrase in 
                        AutomationCommandRegistry.get_protected_phrases())
        
        # Add grid terms
        protected.add(self._app_config.grid.show_grid_phrase.lower().strip())
        
        # Add mark configuration terms
        mark_triggers = self._app_config.mark.triggers
        protected.add(mark_triggers.create_mark.lower().strip())
        protected.add(mark_triggers.delete_mark.lower().strip())
        protected.update(phrase.lower().strip() for phrase in mark_triggers.visualize_marks)
        protected.update(phrase.lower().strip() for phrase in mark_triggers.reset_marks)
        
        # Add dictation terms
        dictation = self._app_config.dictation
        protected.add(dictation.start_trigger.lower().strip())
        protected.add(dictation.stop_trigger.lower().strip())
        protected.add(dictation.type_trigger.lower().strip())
        protected.add(dictation.smart_start_trigger.lower().strip())
        
        # Add core system commands
        system_commands = {
            "start dictation", "stop dictation", "stop", "exit", "quit",
            "grid", "cancel", "yes", "no"
        }
        protected.update(cmd.lower().strip() for cmd in system_commands)
        
        # Add live mark names from storage
        try:
            mark_names = await UnifiedStorageServiceExtensions.get_all_mark_names(self._storage)
            protected.update(name.lower().strip() for name in mark_names)
        except Exception as e:
            logger.warning(f"Could not fetch mark names for protection: {e}")
        
        # Add live sound names from storage
        try:
            sound_mappings = await read_sound_mappings(self._storage, {})
            protected.update(sound.lower().strip() for sound in sound_mappings.keys())
        except Exception as e:
            logger.warning(f"Could not fetch sound names for protection: {e}")
        
        return protected
    
    async def is_term_protected(self, term: str) -> bool:
        """Check if a specific term is protected"""
        if not term or not term.strip():
            return True  # Empty terms are protected
            
        normalized_term = term.lower().strip()
        protected_terms = await self.get_all_protected_terms()
        return normalized_term in protected_terms
    
    async def validate_command_name(self, name: str, exclude_name: str = "") -> Optional[str]:
        """
        Validate a command name for conflicts.
        
        Args:
            name: The name to validate
            exclude_name: Optional name to exclude from validation (for updates)
            
        Returns:
            None if valid, error message if invalid
        """
        if not name or not name.strip():
            return "Command name cannot be empty"
        
        normalized_name = name.lower().strip()
        
        # Skip validation if this is the same name being updated
        if exclude_name and normalized_name == exclude_name.lower().strip():
            return None
        
        if await self.is_term_protected(normalized_name):
            return f"'{name}' is a protected term and cannot be used"
        
        return None
    
    async def get_protected_terms_by_category(self) -> dict:
        """Get protected terms organized by category for debugging/inspection"""
        categories = {
            "automation_commands": set(AutomationCommandRegistry.get_protected_phrases()),
            "grid_commands": {self._app_config.grid.show_grid_phrase},
            "mark_triggers": set(),
            "dictation_triggers": set(),
            "system_commands": {"start dictation", "stop dictation", "stop", "exit", "quit", "grid", "cancel", "yes", "no"},
            "live_marks": set(),
            "live_sounds": set()
        }
        
        # Mark triggers
        mark_triggers = self._app_config.mark.triggers
        categories["mark_triggers"].add(mark_triggers.create_mark)
        categories["mark_triggers"].add(mark_triggers.delete_mark)
        categories["mark_triggers"].update(mark_triggers.visualize_marks)
        categories["mark_triggers"].update(mark_triggers.reset_marks)
        
        # Dictation triggers
        dictation = self._app_config.dictation
        categories["dictation_triggers"].add(dictation.start_trigger)
        categories["dictation_triggers"].add(dictation.stop_trigger)
        categories["dictation_triggers"].add(dictation.type_trigger)
        categories["dictation_triggers"].add(dictation.smart_start_trigger)
        
        # Live data
        try:
            mark_names = await UnifiedStorageServiceExtensions.get_all_mark_names(self._storage)
            categories["live_marks"] = mark_names
        except Exception as e:
            logger.warning(f"Could not fetch mark names: {e}")
        
        try:
            sound_mappings = await read_sound_mappings(self._storage, {})
            categories["live_sounds"] = set(sound_mappings.keys())
        except Exception as e:
            logger.warning(f"Could not fetch sound names: {e}")
        
        return categories 