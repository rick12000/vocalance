import asyncio
import logging
from typing import Dict

from PySide6.QtCore import Signal

from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import DictationAliasListUpdatedEvent
from vocalance.app.services.audio.dictation_handling.dictation_alias_service import DictationAliasService
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtDictationAliasController(QtBaseController):
    """Business logic controller for dictation alias functionality.

    Manages alias CRUD operations and emits Qt signals for UI updates.
    """

    aliases_loaded = Signal(dict)
    alias_added = Signal(str, str)
    alias_updated = Signal(str, str)
    alias_deleted = Signal(str)
    operation_error = Signal(str)
    status_updated = Signal(str, bool)

    def __init__(
        self,
        event_bus: EventBus,
        alias_service: DictationAliasService,
    ) -> None:
        """Initialize dictation alias controller.

        Args:
            event_bus: Event bus for pub/sub.
            alias_service: DictationAliasService instance.
        """
        super().__init__(
            event_bus=event_bus,
            logger=logging.getLogger("QtDictationAliasController"),
        )

        self.alias_service = alias_service
        self._aliases: Dict[str, str] = {}

        self._subscribe_to_events()
        self.logger.debug("QtDictationAliasController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to alias list update events."""
        try:
            self.event_bus.subscribe(DictationAliasListUpdatedEvent, self._on_aliases_updated)
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    def _on_aliases_updated(self, list_update: DictationAliasListUpdatedEvent) -> None:
        """Handle alias list updated event."""
        self._aliases = list_update.aliases
        self.aliases_loaded.emit(self._aliases)

    def refresh_aliases(self) -> None:
        """Load aliases directly from the service and emit them to the view."""
        try:
            self._aliases = self.alias_service.get_aliases()
            self.aliases_loaded.emit(self._aliases)
            self.notify_status(f"Loaded {len(self._aliases)} aliases")
        except Exception as e:
            self.logger.error(f"Error refreshing aliases: {e}", exc_info=True)
            self.notify_status(f"Error loading aliases: {e}", is_error=True)

    def add_alias(self, key: str, value: str) -> bool:
        """Add a new alias mapping.

        Args:
            key: Activation phrase.
            value: Substitution text.

        Returns:
            False if validation fails, True if the request was submitted.
        """
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

        asyncio.create_task(self._do_add_alias(key, value))
        return True

    async def _do_add_alias(self, key: str, value: str) -> None:
        """Perform async add alias operation."""
        try:
            success = await self.alias_service.add_alias(key, value)
            if success:
                self.alias_added.emit(key, value)
                self.notify_status(f"Added alias: '{key}'")
            else:
                self.notify_status(f"Failed to add alias: '{key}'", is_error=True)
        except Exception as e:
            self.logger.error(f"Error adding alias: {e}", exc_info=True)
            self.notify_status(f"Error adding alias: {e}", is_error=True)

    def update_alias(self, key: str, value: str) -> bool:
        """Update an existing alias mapping.

        Args:
            key: Activation phrase to update.
            value: New substitution text.

        Returns:
            False if validation fails, True if the request was submitted.
        """
        key = key.strip()
        value = value.strip()

        if not key:
            self.notify_status("Please enter an activation phrase.", is_error=True)
            return False
        if not value:
            self.notify_status("Please enter a substitution phrase.", is_error=True)
            return False

        asyncio.create_task(self._do_update_alias(key, value))
        return True

    async def _do_update_alias(self, key: str, value: str) -> None:
        """Perform async update alias operation."""
        try:
            success = await self.alias_service.update_alias(key, value)
            if success:
                self.alias_updated.emit(key, value)
                self.notify_status(f"Updated alias: '{key}'")
            else:
                self.notify_status(f"Failed to update alias: '{key}'", is_error=True)
        except Exception as e:
            self.logger.error(f"Error updating alias: {e}", exc_info=True)
            self.notify_status(f"Error updating alias: {e}", is_error=True)

    def delete_alias(self, key: str) -> bool:
        """Delete an alias mapping.

        Args:
            key: Activation phrase to delete.

        Returns:
            False if the key is empty, True if the request was submitted.
        """
        key = key.strip()
        if not key:
            self.notify_status("Invalid alias key.", is_error=True)
            return False

        asyncio.create_task(self._do_delete_alias(key))
        return True

    async def _do_delete_alias(self, key: str) -> None:
        """Perform async delete alias operation."""
        try:
            success = await self.alias_service.delete_alias(key)
            if success:
                self.alias_deleted.emit(key)
                self.notify_status(f"Deleted alias: '{key}'")
            else:
                self.notify_status(f"Failed to delete alias: '{key}'", is_error=True)
        except Exception as e:
            self.logger.error(f"Error deleting alias: {e}", exc_info=True)
            self.notify_status(f"Error deleting alias: {e}", is_error=True)

    def get_aliases(self) -> Dict[str, str]:
        """Return a copy of the current aliases dict."""
        return dict(self._aliases)

    def notify_status(self, message: str, is_error: bool = False) -> None:
        """Emit a status update signal.

        Args:
            message: Status message text.
            is_error: True if this represents an error condition.
        """
        self.status_updated.emit(message, is_error)
        if is_error:
            self.operation_error.emit(message)

    def cleanup(self) -> None:
        """Unsubscribe from all events and release resources."""
        try:
            self.event_bus.unsubscribe(DictationAliasListUpdatedEvent, self._on_aliases_updated)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
