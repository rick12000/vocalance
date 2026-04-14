"""In-memory command history for the session; persisted on shutdown."""

from __future__ import annotations

import logging
import time
from typing import List

from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class CommandHistoryManager:
    """Buffers executed commands in memory, then writes them to storage when the app exits."""

    def __init__(self, storage: StorageService, protected_terms_validator: ProtectedTermsValidator) -> None:
        self._storage = storage
        self._protected_terms_validator = protected_terms_validator
        self._session_history: List[CommandHistoryEntry] = []

    async def initialize(self) -> bool:
        try:
            history_data = await self._storage.read(model_type=CommandHistoryData)
        except Exception as e:
            logger.debug("No existing command history (starting fresh): %s", e)
            self._session_history = []
            return True

        self._session_history = list(history_data.history)
        logger.debug("Loaded %s commands from history", len(self._session_history))
        return True

    async def record_command(self, command: str, source: str) -> None:
        if not await self._protected_terms_validator.is_term_protected(command):
            logger.warning("Rejected non-command text from history: %r (source=%s)", command, source)
            return

        entry = CommandHistoryEntry(command=command, timestamp=time.time(), success=None, metadata={"source": source})
        self._session_history.append(entry)
        logger.debug("Recorded to history: %r (source=%s)", command, source)

    async def get_recent_history(self, count: int) -> List[CommandHistoryEntry]:
        return list(self._session_history[-count:])

    async def get_full_history(self) -> List[CommandHistoryEntry]:
        return list(self._session_history)

    async def shutdown(self) -> bool:
        if not self._session_history:
            return True
        payload = CommandHistoryData(history=self._session_history)
        ok = await self._storage.write(data=payload)
        if not ok:
            logger.error("Failed to write command history")
        return ok
