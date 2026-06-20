import asyncio
import logging
from typing import Dict

from PySide6.QtCore import Signal

from vocalance.app.config.alias_validation import is_valid_alias_text
from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import DictationAliasListUpdatedEvent, DictationAliasUiOperationEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtDictationAliasController(QtBaseController):
    aliases_loaded = Signal(dict)
    operation_error = Signal(str)

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger("QtDictationAliasController"))
        self.alias_entries: Dict[str, str] = {}
        self.subscribe(DictationAliasListUpdatedEvent, self.on_aliases_updated)

    def alias_ui(self, op: str, **kwargs: str) -> None:
        asyncio.create_task(self.event_bus.publish(DictationAliasUiOperationEvent(op=op, **kwargs)))

    def on_aliases_updated(self, list_update: DictationAliasListUpdatedEvent) -> None:
        self.alias_entries = list_update.aliases
        self.aliases_loaded.emit(self.alias_entries)

    def refresh_aliases(self) -> None:
        self.alias_ui("refresh_list")
        self.notify_status("Requesting aliases…")

    def add_alias(self, key: str, value: str) -> bool:
        key = key.strip()
        value = value.strip()
        if not key:
            self.notify_status("Please enter an activation phrase.", is_error=True)
            return False
        if not value:
            self.notify_status("Please enter a substitution phrase.", is_error=True)
            return False
        if not is_valid_alias_text(key) or not is_valid_alias_text(value):
            self.notify_status("Alias contains characters that are not permitted.", is_error=True)
            return False
        if key.lower() in {k.lower() for k in self.alias_entries}:
            self.notify_status(f"Alias '{key}' already exists.", is_error=True)
            return False
        self.alias_ui("add", key=key, value=value)
        return True

    def update_alias(self, key: str, value: str) -> bool:
        key = key.strip()
        value = value.strip()
        if not key:
            self.notify_status("Please enter an activation phrase.", is_error=True)
            return False
        if not value:
            self.notify_status("Please enter a substitution phrase.", is_error=True)
            return False
        if not is_valid_alias_text(key) or not is_valid_alias_text(value):
            self.notify_status("Alias contains characters that are not permitted.", is_error=True)
            return False
        self.alias_ui("update", key=key, value=value)
        return True

    def delete_alias(self, key: str) -> bool:
        key = key.strip()
        if not key:
            self.notify_status("Invalid alias key.", is_error=True)
            return False
        self.alias_ui("delete", key=key)
        return True

    def notify_status(self, message: str, is_error: bool = False) -> None:
        self.emit_status(message, is_error)
        if is_error:
            self.operation_error.emit(message)
