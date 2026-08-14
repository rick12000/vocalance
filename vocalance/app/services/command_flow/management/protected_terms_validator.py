import time
from typing import Optional, Set, Tuple

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.automation_command_registry import AutomationCommandRegistry
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_management_events import CommandMappingsUpdatedEvent
from vocalance.app.events.sound_events import SoundToCommandMappingUpdatedEvent
from vocalance.app.services.base_service import Service
from vocalance.app.services.storage.storage_models import MarksData, SoundMappingsData
from vocalance.app.services.storage.storage_service import StorageService


class ProtectedTermsValidator(Service):
    """Validates reserved terms for commands, marks, and sound labels with a short-lived cache.

    The event bus is optional at construction so the validator can be built before
    the bus exists (used by tests). Call :meth:`setup_invalidation_subscriptions`
    once an event bus is available to wire cache invalidation.
    """

    def __init__(self, config: GlobalAppConfig, storage: StorageService) -> None:
        self.config = config
        self.storage = storage
        self.cached_terms: Optional[Set[str]] = None
        self.cache_expiry: float = 0.0
        self.cache_ttl: float = config.protected_terms_validator.cache_ttl_seconds
        self._wired = False

    def setup_invalidation_subscriptions(self, event_bus: EventBus) -> None:
        """Bind to ``event_bus`` and subscribe to mapping updates that invalidate the cache."""
        Service.__init__(self, event_bus)
        self._wired = True
        self.subscribe(CommandMappingsUpdatedEvent, self.handle_command_mappings_updated)
        self.subscribe(SoundToCommandMappingUpdatedEvent, self.handle_sound_mapping_updated)

    async def handle_command_mappings_updated(self, event: CommandMappingsUpdatedEvent) -> None:
        if event.success:
            self.invalidate_cache()

    async def handle_sound_mapping_updated(self, event: SoundToCommandMappingUpdatedEvent) -> None:
        if event.success:
            self.invalidate_cache()

    async def get_all_protected_terms(self) -> Set[str]:
        """Return normalized (lowercase, stripped) protected terms, using cache when valid."""
        current_time: float = time.time()

        if self.cached_terms and current_time < self.cache_expiry:
            return self.cached_terms

        protected: Set[str] = set()

        protected.update(phrase.lower().strip() for phrase in AutomationCommandRegistry.get_protected_phrases())

        protected.add(self.config.grid.show_grid_phrase.lower().strip())
        protected.add(self.config.grid.hover_grid_phrase.lower().strip())
        protected.add(self.config.grid.drag_grid_phrase.lower().strip())

        protected.update(str(i) for i in range(1, 11))

        mark_triggers = self.config.mark.triggers
        protected.add(mark_triggers.create_mark.lower().strip())
        protected.add(mark_triggers.delete_mark.lower().strip())
        protected.update(p.lower().strip() for p in mark_triggers.visualize_marks)
        protected.update(p.lower().strip() for p in mark_triggers.reset_marks)
        protected.update(p.lower().strip() for p in mark_triggers.visualization_cancel)

        dictation = self.config.dictation
        protected.add(dictation.start_trigger.lower().strip())
        protected.add(dictation.stop_trigger.lower().strip())
        protected.add(dictation.type_trigger.lower().strip())
        protected.add(dictation.smart_start_trigger.lower().strip())
        protected.add(dictation.visual_start_trigger.lower().strip())
        protected.add(dictation.hidden_start_trigger.lower().strip())
        protected.add(dictation.amend_start_trigger.lower().strip())

        protected.add("pause")
        protected.add("resume")
        protected.add("repeat")

        marks_data = await self.storage.read(model_type=MarksData)
        protected.update(name.lower().strip() for name in marks_data.marks.keys())

        sound_data = await self.storage.read(model_type=SoundMappingsData)
        protected.update(sound.lower().strip() for sound in sound_data.mappings.keys())

        self.cached_terms = protected
        self.cache_expiry = current_time + self.cache_ttl

        return protected

    async def is_term_protected(self, term: str) -> bool:
        """Return True if ``term`` matches a protected term (case-insensitive)."""
        protected = await self.get_all_protected_terms()
        return term.lower().strip() in protected

    async def validate_term(self, term: str, exclude_term: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Return ``(True, None)`` if ``term`` is usable, else ``(False, error_message)``."""
        if not term or not term.strip():
            return False, "Term cannot be empty"

        normalized: str = term.lower().strip()

        if exclude_term and normalized == exclude_term.lower().strip():
            return True, None

        protected = await self.get_all_protected_terms()
        if normalized in protected:
            return False, f"'{term}' is a protected term and cannot be used"

        return True, None

    def invalidate_cache(self) -> None:
        """Drop cached terms so the next read rebuilds from storage and config."""
        self.cached_terms = None
        self.cache_expiry = 0.0

    async def shutdown(self) -> None:
        if self._wired:
            await super().shutdown()
            self._wired = False
