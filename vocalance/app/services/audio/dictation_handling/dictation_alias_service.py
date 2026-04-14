import asyncio
import logging
import re
import threading
from typing import Dict, Optional

from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import DictationAliasListUpdatedEvent
from vocalance.app.services.storage.storage_models import DictationAliasData
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.utils.concurrency import schedule_on_loop

logger = logging.getLogger(__name__)

# Flag word prefix for activation phrases
ALIAS_FLAG_WORD = "insert"


class DictationAliasService:
    """Service for managing dictation alias substitutions.

    Handles CRUD operations for alias mappings (activation phrase -> substitution)
    and applies substitutions to recognized text. Thread-safe with RLock protection.

    The substitution pattern is: "insert {activation_phrase}" -> substitution value
    Matching is case-insensitive, but original capitalization is preserved.
    """

    def __init__(
        self,
        event_bus: EventBus,
        storage: StorageService,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._event_bus = event_bus
        self._storage = storage
        self._event_loop = event_loop

        self._lock = threading.RLock()
        self._aliases: Dict[str, str] = {}
        self._loaded = False

        logger.debug("DictationAliasService initialized")

    async def initialize(self) -> bool:
        """Initialize service by loading aliases from storage.

        Returns:
            True if initialization succeeded.
        """
        try:
            await self._load_aliases()
            logger.info(f"DictationAliasService initialized with {len(self._aliases)} aliases")
            return True
        except Exception as e:
            logger.error(f"Error initializing DictationAliasService: {e}", exc_info=True)
            return False

    async def _load_aliases(self) -> None:
        """Load aliases from storage into memory cache."""
        try:
            data = await self._storage.read(DictationAliasData)
            with self._lock:
                self._aliases = dict(data.aliases)
                self._loaded = True
            logger.debug(f"Loaded {len(self._aliases)} aliases from storage")
        except Exception as e:
            logger.error(f"Error loading aliases: {e}", exc_info=True)
            with self._lock:
                self._aliases = {}
                self._loaded = True

    async def _save_aliases(self) -> bool:
        """Save current aliases to storage.

        Returns:
            True if save succeeded.
        """
        try:
            with self._lock:
                data = DictationAliasData(aliases=dict(self._aliases))
            success = await self._storage.write(data)
            if success:
                logger.debug(f"Saved {len(self._aliases)} aliases to storage")
            return success
        except Exception as e:
            logger.error(f"Error saving aliases: {e}", exc_info=True)
            return False

    def _publish_update(self) -> None:
        """Schedule alias list updated event onto the asyncio loop."""
        with self._lock:
            aliases_copy = dict(self._aliases)
        loop = self._event_loop
        if loop and not loop.is_closed():
            schedule_on_loop(loop, self._event_bus.publish(DictationAliasListUpdatedEvent(aliases=aliases_copy)))
        else:
            logger.warning("DictationAliasService: no event loop available for _publish_update")

    def get_aliases(self) -> Dict[str, str]:
        """Get all aliases (thread-safe copy).

        Returns:
            Dict mapping activation phrases to substitution values.
        """
        with self._lock:
            return dict(self._aliases)

    async def add_alias(self, key: str, value: str) -> bool:
        """Add a new alias mapping.

        Args:
            key: Activation phrase (case-insensitive for matching).
            value: Substitution text.

        Returns:
            True if alias was added successfully.
        """
        key = key.strip().lower()
        value = value.strip()

        if not key or not value:
            logger.warning("Cannot add alias with empty key or value")
            return False

        with self._lock:
            if key in self._aliases:
                logger.warning(f"Alias '{key}' already exists, use update instead")
                return False
            self._aliases[key] = value

        success = await self._save_aliases()
        if success:
            self._publish_update()
            logger.info(f"Added alias: '{key}' -> '{value}'")
        else:
            with self._lock:
                del self._aliases[key]
        return success

    async def update_alias(self, key: str, value: str) -> bool:
        """Update an existing alias mapping.

        Args:
            key: Activation phrase to update.
            value: New substitution text.

        Returns:
            True if alias was updated successfully.
        """
        key = key.strip().lower()
        value = value.strip()

        if not key or not value:
            logger.warning("Cannot update alias with empty key or value")
            return False

        with self._lock:
            if key not in self._aliases:
                logger.warning(f"Alias '{key}' does not exist, use add instead")
                return False
            old_value = self._aliases[key]
            self._aliases[key] = value

        success = await self._save_aliases()
        if success:
            self._publish_update()
            logger.info(f"Updated alias: '{key}' -> '{value}'")
        else:
            with self._lock:
                self._aliases[key] = old_value
        return success

    async def delete_alias(self, key: str) -> bool:
        """Delete an alias mapping.

        Args:
            key: Activation phrase to delete.

        Returns:
            True if alias was deleted successfully.
        """
        key = key.strip().lower()

        with self._lock:
            if key not in self._aliases:
                logger.warning(f"Alias '{key}' does not exist")
                return False
            old_value = self._aliases.pop(key)

        success = await self._save_aliases()
        if success:
            self._publish_update()
            logger.info(f"Deleted alias: '{key}'")
        else:
            with self._lock:
                self._aliases[key] = old_value
        return success

    def extract_aliases(self, text: str) -> tuple[str, dict[str, str]]:
        """Extract aliases from text, returning text with placeholders and a map of placeholders to original text.

        The placeholders are formatted as 'vocalancealiasN' to survive post-processing modifiers
        (like camel, snake, strip) without being split or removed.
        """
        if not text:
            return text, {}

        with self._lock:
            if not self._aliases:
                return text, {}
            aliases_copy = dict(self._aliases)

        sorted_keys = sorted(aliases_copy.keys(), key=len, reverse=True)
        escaped_keys = [re.escape(k) for k in sorted_keys]
        if not escaped_keys:
            return text, {}

        pattern = rf"\b{ALIAS_FLAG_WORD}\s+({'|'.join(escaped_keys)})\b"

        alias_map = {}
        counter = [0]

        def replace_match(match: re.Match) -> str:
            matched_key = match.group(1).lower()
            substitution = aliases_copy.get(matched_key, match.group(0))
            placeholder = f"vocalancealias{counter[0]}"
            alias_map[placeholder] = substitution
            counter[0] += 1
            return placeholder

        result = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

        if result != text:
            logger.debug(f"Extracted aliases: '{text}' -> '{result}' with map {alias_map}")

        return result, alias_map

    def apply_substitutions(self, text: str) -> str:
        if not text:
            return text

        with self._lock:
            if not self._aliases:
                return text
            aliases_copy = dict(self._aliases)

        sorted_keys = sorted(aliases_copy.keys(), key=len, reverse=True)
        escaped_keys = [re.escape(k) for k in sorted_keys]
        if not escaped_keys:
            return text

        pattern = rf"\b{ALIAS_FLAG_WORD}\s+({'|'.join(escaped_keys)})\b"

        def replace_match(match: re.Match) -> str:
            matched_key = match.group(1).lower()
            return aliases_copy.get(matched_key, match.group(0))

        result = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

        if result != text:
            logger.debug(f"Applied alias substitutions: '{text}' -> '{result}'")

        return result

    async def shutdown(self) -> None:
        """Shutdown service and save any pending changes."""
        try:
            await self._save_aliases()
            logger.info("DictationAliasService shutdown complete")
        except Exception as e:
            logger.error(f"Error during DictationAliasService shutdown: {e}")
