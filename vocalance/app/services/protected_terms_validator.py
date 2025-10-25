import logging
import time
from typing import Optional, Set, Tuple

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.automation_command_registry import AutomationCommandRegistry
from vocalance.app.services.storage.storage_models import MarksData, SoundMappingsData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class ProtectedTermsValidator:
    """Centralized validation for protected/reserved terms across all command types.

    Provides single source of truth for terms that cannot be used as command names,
    mark labels, or sound labels. Caches protected terms for performance.
    """

    def __init__(self, config: GlobalAppConfig, storage: StorageService) -> None:
        self._config: GlobalAppConfig = config
        self._storage: StorageService = storage
        self._cached_terms: Optional[Set[str]] = None
        self._cache_expiry: float = 0.0
        self._cache_ttl: float = config.protected_terms_validator.cache_ttl_seconds

        logger.debug("ProtectedTermsValidator initialized")

    async def get_all_protected_terms(self) -> Set[str]:
        """Get all protected terms from all sources with caching.

        Returns:
            Set of normalized (lowercase, stripped) protected terms.
        """
        current_time = time.time()

        if self._cached_terms and current_time < self._cache_expiry:
            return self._cached_terms

        protected: Set[str] = set()

        protected.update(phrase.lower().strip() for phrase in AutomationCommandRegistry.get_protected_phrases())

        protected.add(self._config.grid.show_grid_phrase.lower().strip())
        protected.add(self._config.grid.cancel_grid_phrase.lower().strip())

        mark_triggers = self._config.mark.triggers
        protected.add(mark_triggers.create_mark.lower().strip())
        protected.add(mark_triggers.delete_mark.lower().strip())
        protected.update(p.lower().strip() for p in mark_triggers.visualize_marks)
        protected.update(p.lower().strip() for p in mark_triggers.reset_marks)
        protected.update(p.lower().strip() for p in mark_triggers.visualization_cancel)

        dictation = self._config.dictation
        protected.add(dictation.start_trigger.lower().strip())
        protected.add(dictation.stop_trigger.lower().strip())
        protected.add(dictation.type_trigger.lower().strip())
        protected.add(dictation.smart_start_trigger.lower().strip())

        try:
            marks_data = await self._storage.read(model_type=MarksData)
            protected.update(name.lower().strip() for name in marks_data.marks.keys())
        except Exception as e:
            logger.debug(f"Could not fetch mark names for protection: {e}")

        try:
            sound_data = await self._storage.read(model_type=SoundMappingsData)
            protected.update(sound.lower().strip() for sound in sound_data.mappings.keys())
        except Exception as e:
            logger.debug(f"Could not fetch sound names for protection: {e}")

        self._cached_terms = protected
        self._cache_expiry = current_time + self._cache_ttl

        return protected

    async def is_term_protected(self, term: str) -> bool:
        """Check if a term is protected.

        Args:
            term: Term to check.

        Returns:
            True if term is protected, False otherwise.
        """
        protected = await self.get_all_protected_terms()
        return term.lower().strip() in protected

    async def validate_term(self, term: str, exclude_term: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Validate a term for use as command/mark/sound name.

        Args:
            term: Term to validate.
            exclude_term: Optional term to exclude from conflict check (for updates).

        Returns:
            Tuple of (is_valid, error_message) where error_message is None if valid.
        """
        if not term or not term.strip():
            return False, "Term cannot be empty"

        normalized = term.lower().strip()

        if exclude_term and normalized == exclude_term.lower().strip():
            return True, None

        protected = await self.get_all_protected_terms()
        if normalized in protected:
            return False, f"'{term}' is a protected term and cannot be used"

        return True, None

    def invalidate_cache(self) -> None:
        """Invalidate cached protected terms to force reload on next access."""
        self._cached_terms = None
        self._cache_expiry = 0.0
        logger.debug("Protected terms cache invalidated")
