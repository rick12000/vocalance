"""Qt-based dictation alias controller.

Business logic controller for dictation alias management with Qt signals.
"""

import asyncio
import logging
from typing import Dict

from PySide6.QtCore import Signal

from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import DictationAliasListUpdatedEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtDictationAliasController(QtBaseController):
    """Business logic controller for dictation alias functionality.

    Manages alias CRUD operations and emits Qt signals for UI updates.
    Thread-safe event handling with signal-based communication.
    """

    # Signals for alias operations
    aliases_loaded = Signal(dict)  # Dict[str, str] - aliases mapping
    alias_added = Signal(str, str)  # key, value
    alias_updated = Signal(str, str)  # key, value
    alias_deleted = Signal(str)  # key
    operation_error = Signal(str)  # error message
    status_updated = Signal(str, bool)  # message, is_error

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        alias_service,
        main_window,
    ):
        """Initialize dictation alias controller.

        Args:
            event_bus: Event bus for pub/sub.
            event_loop: Asyncio event loop.
            alias_service: DictationAliasService instance.
            main_window: Main window reference.
        """
        super().__init__(
            event_bus=event_bus,
            event_loop=event_loop,
            logger=logging.getLogger("QtDictationAliasController"),
        )

        self.alias_service = alias_service
        self.main_window = main_window

        # State
        self._aliases: Dict[str, str] = {}

        # Subscribe to events
        self._subscribe_to_events()

        self.logger.debug("QtDictationAliasController initialized")

    def _subscribe_to_events(self) -> None:
        """Subscribe to alias-related events."""
        try:
            self.event_bus.subscribe(DictationAliasListUpdatedEvent, self._on_aliases_updated)
            self.logger.debug("Subscribed to alias events")
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}", exc_info=True)

    # --- Event Handlers ---

    async def _on_aliases_updated(self, event: DictationAliasListUpdatedEvent) -> None:
        """Handle alias list updated event."""
        self._aliases = event.aliases
        self.aliases_loaded.emit(self._aliases)
        self.logger.debug(f"Aliases updated: {len(self._aliases)} aliases")

    # --- Public Methods ---

    def refresh_aliases(self) -> None:
        """Refresh the aliases list from service."""
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
            True if request was submitted successfully.
        """
        key = key.strip()
        value = value.strip()

        if not key:
            self.notify_status("Please enter an activation phrase.", is_error=True)
            return False

        if not value:
            self.notify_status("Please enter a substitution phrase.", is_error=True)
            return False

        # Check if key already exists
        if key.lower() in {k.lower() for k in self._aliases.keys()}:
            self.notify_status(f"Alias '{key}' already exists.", is_error=True)
            return False

        # Submit async operation
        asyncio.run_coroutine_threadsafe(self._do_add_alias(key, value), self.event_loop)
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
            True if request was submitted successfully.
        """
        key = key.strip()
        value = value.strip()

        if not key:
            self.notify_status("Please enter an activation phrase.", is_error=True)
            return False

        if not value:
            self.notify_status("Please enter a substitution phrase.", is_error=True)
            return False

        # Submit async operation
        asyncio.run_coroutine_threadsafe(self._do_update_alias(key, value), self.event_loop)
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
            True if request was submitted successfully.
        """
        key = key.strip()

        if not key:
            self.notify_status("Invalid alias key.", is_error=True)
            return False

        # Submit async operation
        asyncio.run_coroutine_threadsafe(self._do_delete_alias(key), self.event_loop)
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

    # --- Getters ---

    def get_aliases(self) -> Dict[str, str]:
        """Get current aliases."""
        return dict(self._aliases)

    def notify_status(self, message: str, is_error: bool = False) -> None:
        """Notify status message."""
        self.status_updated.emit(message, is_error)
        if is_error:
            self.operation_error.emit(message)

    def cleanup(self) -> None:
        """Clean up controller resources."""
        try:
            self.event_bus.unsubscribe(DictationAliasListUpdatedEvent, self._on_aliases_updated)
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

        super().cleanup()
