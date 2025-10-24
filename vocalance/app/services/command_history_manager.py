import asyncio
import logging
import time
from typing import List

from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class CommandHistoryManager:
    """Manages command execution history with in-memory accumulation.

    Records commands to in-memory buffer during session for fast zero-I/O tracking,
    then persists full history to storage at shutdown. History used by Markov
    predictor for training. Thread-safe using async locks.
    """

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage
        self._session_history: List[CommandHistoryEntry] = []
        self._lock = asyncio.Lock()

        logger.info("CommandHistoryManager initialized")

    async def initialize(self) -> bool:
        """Load existing history from storage into memory.

        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            history_data = await self._storage.read(model_type=CommandHistoryData)
            async with self._lock:
                self._session_history = list(history_data.history)
            logger.info(f"Loaded {len(self._session_history)} commands from history")
            return True
        except Exception as e:
            logger.warning(f"Could not load history (starting fresh): {e}")
            async with self._lock:
                self._session_history = []
            return False

    async def record_command(self, command: str, source: str) -> None:
        """Record command to in-memory history (fast, no I/O).

        Args:
            command: The command text that was executed
            source: Source of the command ("stt", "sound", "markov")
        """
        entry = CommandHistoryEntry(command=command, timestamp=time.time(), success=None, metadata={"source": source})

        async with self._lock:
            self._session_history.append(entry)

        logger.debug(f"Recorded to history: '{command}' (source={source}, total={len(self._session_history)})")

    async def get_recent_history(self, count: int) -> List[CommandHistoryEntry]:
        """Get N most recent commands from history.

        Args:
            count: Number of recent commands to retrieve

        Returns:
            List of most recent command history entries
        """
        async with self._lock:
            return list(self._session_history[-count:])

    async def get_full_history(self) -> List[CommandHistoryEntry]:
        """Get complete command history.

        Returns:
            Full list of command history entries
        """
        async with self._lock:
            return list(self._session_history)

    async def shutdown(self) -> bool:
        """Write accumulated history to storage.

        Returns:
            True if write succeeded, False otherwise
        """
        async with self._lock:
            if not self._session_history:
                logger.info("No commands to write at shutdown")
                return True

            history_data = CommandHistoryData(history=self._session_history)

        success = await self._storage.write(data=history_data)

        if success:
            logger.info(f"Successfully wrote {len(history_data.history)} commands")
        else:
            logger.error("Failed to write command history")

        return success
