import asyncio
import logging
from typing import Dict

from PySide6.QtCore import Signal

from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import DictationAliasListUpdatedEvent, DictationAliasUiOperationEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtDictationAliasController(QtBaseController):
    aliases_loaded = Signal(dict)
    operation_error = Signal(str)
    status_updated = Signal(str, bool)

    def __init__(
        self,
        event_bus: EventBus,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtDictationAliasController"),
        )

        self._aliases: Dict[str, str] = {}

        self._subscribe_to_events()
        self.logger.debug("QtDictationAliasController initialized")

    def _subscribe_to_events(self) -> None:
        try:
            self.event_bus.subscribe(DictationAliasListUpdatedEvent, self._on_aliases_updated)
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def _on_aliases_updated(self, list_update: DictationAliasListUpdatedEvent) -> None:
        self._aliases = list_update.aliases
        self.aliases_loaded.emit(self._aliases)

    def refresh_aliases(self) -> None:
        asyncio.create_task(self.event_bus.publish(DictationAliasUiOperationEvent(op="refresh_list")))
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
        if key.lower() in {k.lower() for k in self._aliases}:
            self.notify_status(f"Alias '{key}' already exists.", is_error=True)
            return False

        asyncio.create_task(self.event_bus.publish(DictationAliasUiOperationEvent(op="add", key=key, value=value)))
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

        asyncio.create_task(self.event_bus.publish(DictationAliasUiOperationEvent(op="update", key=key, value=value)))
        return True

    def delete_alias(self, key: str) -> bool:
        key = key.strip()
        if not key:
            self.notify_status("Invalid alias key.", is_error=True)
            return False

        asyncio.create_task(self.event_bus.publish(DictationAliasUiOperationEvent(op="delete", key=key)))
        return True

    def notify_status(self, message: str, is_error: bool = False) -> None:
        self.status_updated.emit(message, is_error)
        if is_error:
            self.operation_error.emit(message)

    def cleanup(self) -> None:
        try:
            self.event_bus.unsubscribe(DictationAliasListUpdatedEvent, self._on_aliases_updated)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
