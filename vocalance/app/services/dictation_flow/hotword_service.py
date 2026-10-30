from __future__ import annotations

import asyncio
import logging
import threading

from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import (
    DictationHotwordListUpdatedEvent,
    DictationHotwordSaveFailedEvent,
    DictationHotwordUiOperationEvent,
)
from vocalance.app.lifecycle.worker import schedule_on_loop
from vocalance.app.services.base_service import Service
from vocalance.app.services.dictation_flow.hotwords import HotwordValidationError, parse_hotword_text
from vocalance.app.services.storage.storage_models import DictationHotwordData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class DictationHotwordService(Service):
    """Persists hotwords and exposes the slash-joined string sherpa expects."""

    def __init__(
        self,
        event_bus: EventBus,
        storage: StorageService,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(event_bus)
        self.storage = storage
        self.event_loop = event_loop
        self.hotword_lock = threading.RLock()
        self.hotwords: list[str] = []
        self.subscribe(DictationHotwordUiOperationEvent, self.handle_hotword_ui_operation)

    async def handle_hotword_ui_operation(self, event: DictationHotwordUiOperationEvent) -> None:
        if event.op == "save":
            await self.save_text(event.text)
        elif event.op == "refresh_list":
            await self.load_hotwords()
            self.publish_list_updated()

    async def initialize(self) -> bool:
        await self.load_hotwords()
        return True

    async def load_hotwords(self) -> None:
        try:
            data = await self.storage.read(DictationHotwordData)
        except Exception:
            logger.exception("Error loading hotwords from storage")
            return
        if not isinstance(data, DictationHotwordData):
            logger.warning("Hotword storage read did not return hotword data; list stays empty")
            return
        with self.hotword_lock:
            self.hotwords = list(data.hotwords)
        logger.info("Loaded %d hotwords from storage: %s", len(self.hotwords), self.hotwords)

    async def save_hotwords(self) -> bool:
        try:
            with self.hotword_lock:
                data = DictationHotwordData(hotwords=list(self.hotwords))
            return await self.storage.write(data)
        except Exception:
            logger.exception("Error saving hotwords")
            return False

    def get_hotwords(self) -> list[str]:
        with self.hotword_lock:
            return list(self.hotwords)

    def stream_hotwords(self) -> str:
        """Return the saved hotwords joined for ``create_stream``."""
        return "/".join(self.get_hotwords())

    async def save_text(self, text: str) -> bool:
        """Validate ``text``, then replace the stored list. A failure leaves storage unchanged."""
        try:
            words = parse_hotword_text(text)
        except HotwordValidationError as exc:
            self.publish_save_failed(str(exc))
            return False
        with self.hotword_lock:
            previous = list(self.hotwords)
            self.hotwords = words
        if not await self.save_hotwords():
            with self.hotword_lock:
                self.hotwords = previous
            self.publish_save_failed("Could not save hotwords.")
            return False
        self.publish_list_updated()
        return True

    def publish_list_updated(self) -> None:
        hotwords = self.get_hotwords()
        if self.event_loop.is_closed():
            return
        schedule_on_loop(self.event_loop, self.event_bus.publish(DictationHotwordListUpdatedEvent(hotwords=hotwords)))

    def publish_save_failed(self, message: str) -> None:
        if self.event_loop.is_closed():
            return
        schedule_on_loop(self.event_loop, self.event_bus.publish(DictationHotwordSaveFailedEvent(message=message)))

    async def shutdown(self) -> None:
        await self.save_hotwords()
        await super().shutdown()
