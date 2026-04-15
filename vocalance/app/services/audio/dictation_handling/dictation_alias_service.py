import asyncio
import logging
import re
import threading
from typing import Dict

from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import DictationAliasListUpdatedEvent
from vocalance.app.services.storage.storage_models import DictationAliasData
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.utils.concurrency import schedule_on_loop

logger = logging.getLogger(__name__)

ALIAS_FLAG_WORD = "insert"


class DictationAliasService:
    """CRUD and substitution for dictation aliases (activation phrase → text). Thread-safe."""

    def __init__(
        self,
        event_bus: EventBus,
        storage: StorageService,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.event_bus = event_bus
        self.storage = storage
        self.event_loop = event_loop
        self.alias_lock = threading.RLock()
        self.aliases: Dict[str, str] = {}

    async def initialize(self) -> bool:
        await self.load_aliases()
        return True

    async def load_aliases(self) -> None:
        try:
            data = await self.storage.read(DictationAliasData)
            with self.alias_lock:
                self.aliases = dict(data.aliases)
        except Exception as e:
            logger.error("Error loading aliases: %s", e, exc_info=True)
            with self.alias_lock:
                self.aliases = {}

    async def save_aliases(self) -> bool:
        try:
            with self.alias_lock:
                data = DictationAliasData(aliases=dict(self.aliases))
            return await self.storage.write(data)
        except Exception as e:
            logger.error("Error saving aliases: %s", e, exc_info=True)
            return False

    def publish_alias_list_updated(self) -> None:
        with self.alias_lock:
            aliases_copy = dict(self.aliases)
        if self.event_loop.is_closed():
            return
        schedule_on_loop(self.event_loop, self.event_bus.publish(DictationAliasListUpdatedEvent(aliases=aliases_copy)))

    def get_aliases(self) -> Dict[str, str]:
        with self.alias_lock:
            return dict(self.aliases)

    async def add_alias(self, key: str, value: str) -> bool:
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            logger.warning("Cannot add alias with empty key or value")
            return False
        with self.alias_lock:
            if key in self.aliases:
                logger.warning("Alias '%s' already exists, use update instead", key)
                return False
            self.aliases[key] = value
        success = await self.save_aliases()
        if success:
            self.publish_alias_list_updated()
        else:
            with self.alias_lock:
                del self.aliases[key]
        return success

    async def update_alias(self, key: str, value: str) -> bool:
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            logger.warning("Cannot update alias with empty key or value")
            return False
        with self.alias_lock:
            if key not in self.aliases:
                logger.warning("Alias '%s' does not exist, use add instead", key)
                return False
            old_value = self.aliases[key]
            self.aliases[key] = value
        success = await self.save_aliases()
        if success:
            self.publish_alias_list_updated()
        else:
            with self.alias_lock:
                self.aliases[key] = old_value
        return success

    async def delete_alias(self, key: str) -> bool:
        key = key.strip().lower()
        with self.alias_lock:
            if key not in self.aliases:
                logger.warning("Alias '%s' does not exist", key)
                return False
            old_value = self.aliases.pop(key)
        success = await self.save_aliases()
        if success:
            self.publish_alias_list_updated()
        else:
            with self.alias_lock:
                self.aliases[key] = old_value
        return success

    def extract_aliases(self, text: str) -> tuple[str, dict[str, str]]:
        if not text:
            return text, {}
        with self.alias_lock:
            if not self.aliases:
                return text, {}
            aliases_copy = dict(self.aliases)
        sorted_keys = sorted(aliases_copy.keys(), key=len, reverse=True)
        escaped_keys = [re.escape(k) for k in sorted_keys]
        if not escaped_keys:
            return text, {}
        pattern = rf"\b{ALIAS_FLAG_WORD}\s+({'|'.join(escaped_keys)})\b"
        alias_map: dict[str, str] = {}
        counter = [0]

        def replace_match(match: re.Match) -> str:
            matched_key = match.group(1).lower()
            substitution = aliases_copy.get(matched_key, match.group(0))
            placeholder = f"vocalancealias{counter[0]}"
            alias_map[placeholder] = substitution
            counter[0] += 1
            return placeholder

        return re.sub(pattern, replace_match, text, flags=re.IGNORECASE), alias_map

    def apply_substitutions(self, text: str) -> str:
        if not text:
            return text
        with self.alias_lock:
            if not self.aliases:
                return text
            aliases_copy = dict(self.aliases)
        sorted_keys = sorted(aliases_copy.keys(), key=len, reverse=True)
        escaped_keys = [re.escape(k) for k in sorted_keys]
        if not escaped_keys:
            return text
        pattern = rf"\b{ALIAS_FLAG_WORD}\s+({'|'.join(escaped_keys)})\b"

        def replace_match(match: re.Match) -> str:
            matched_key = match.group(1).lower()
            return aliases_copy.get(matched_key, match.group(0))

        return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

    async def shutdown(self) -> None:
        await self.save_aliases()
