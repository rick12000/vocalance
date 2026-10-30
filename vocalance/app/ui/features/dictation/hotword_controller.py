import asyncio
import logging

from PySide6.QtCore import Signal

from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import (
    DictationHotwordListUpdatedEvent,
    DictationHotwordSaveFailedEvent,
    DictationHotwordUiOperationEvent,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtHotwordController(QtBaseController):
    hotwords_loaded = Signal(list)
    operation_error = Signal(str)

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger("QtHotwordController"))
        self.hotwords: list[str] = []
        self.subscribe(DictationHotwordListUpdatedEvent, self.on_hotwords_updated)
        self.subscribe(DictationHotwordSaveFailedEvent, self.on_save_failed)

    def hotword_ui(self, op: str, text: str = "") -> None:
        asyncio.create_task(self.event_bus.publish(DictationHotwordUiOperationEvent(op=op, text=text)))

    def on_hotwords_updated(self, list_update: DictationHotwordListUpdatedEvent) -> None:
        self.hotwords = list(list_update.hotwords)
        self.hotwords_loaded.emit(self.hotwords)

    def on_save_failed(self, event: DictationHotwordSaveFailedEvent) -> None:
        self.operation_error.emit(event.message)

    def refresh_hotwords(self) -> None:
        self.hotword_ui("refresh_list")

    def save_hotwords(self, text: str) -> None:
        self.hotword_ui("save", text=text)
