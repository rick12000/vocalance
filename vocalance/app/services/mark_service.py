import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

import pyautogui

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import (
    BaseCommand,
    MarkCreateCommand,
    MarkDeleteCommand,
    MarkExecuteCommand,
    MarkResetCommand,
    MarkVisualizeCancelCommand,
    MarkVisualizeCommand,
)
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import MarkCommandParsedEvent
from vocalance.app.events.mark_events import (
    MarkCreatedEventData,
    MarkCreateRequestEventData,
    MarkDeleteAllRequestEventData,
    MarkDeleteByNameRequestEventData,
    MarkDeletedEventData,
    MarkExecuteRequestEventData,
    MarkGetAllRequestEventData,
    MarkOperationSuccessEventData,
    MarksChangedEventData,
    MarkVisualizationStateChangedEventData,
    MarkVisualizeAllRequestEventData,
    MarkVisualizeCancelRequestEventData,
)
from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.storage.storage_models import Coordinate, MarksData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class MarkService:
    """Service for managing screen position marks with unified storage.

    Provides fast mark creation, navigation (click), and deletion with cached
    coordinate lookups and validation against reserved labels (commands, sounds).
    All state access is protected with async locks for thread safety. Integrates
    with UI mark visualization for visual feedback.

    Attributes:
        _storage: Storage service for persistent mark data.
        _protected_terms_validator: Validator ensuring mark labels don't conflict.
        _is_viz_active: Flag tracking mark visualization state.
        _viz_lock: Async lock protecting visualization state.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        storage: StorageService,
        protected_terms_validator: ProtectedTermsValidator,
    ) -> None:
        """Initialize mark service with dependencies.

        Args:
            event_bus: EventBus for pub/sub messaging.
            config: Global application configuration.
            storage: Storage service for persistent mark data.
            protected_terms_validator: Validator for mark label conflicts.
        """
        self._event_bus = event_bus
        self._config = config
        self._storage = storage
        self._protected_terms_validator = protected_terms_validator
        self._is_viz_active: bool = False
        self._viz_lock = asyncio.Lock()

        logger.debug("MarkService initialized with protected terms validation")

    def setup_subscriptions(self) -> None:
        """Setup event subscriptions for mark commands and requests."""
        logger.debug("Setting up MarkService event subscriptions")

        # UI-driven requests
        self._event_bus.subscribe(event_type=MarkGetAllRequestEventData, handler=self._handle_get_all_request)
        self._event_bus.subscribe(event_type=MarkCreateRequestEventData, handler=self._handle_create_mark_request)
        self._event_bus.subscribe(event_type=MarkDeleteByNameRequestEventData, handler=self._handle_delete_by_name_request)
        self._event_bus.subscribe(event_type=MarkDeleteAllRequestEventData, handler=self._handle_delete_all_request)
        self._event_bus.subscribe(event_type=MarkExecuteRequestEventData, handler=self._handle_execute_mark_request)
        self._event_bus.subscribe(event_type=MarkVisualizeAllRequestEventData, handler=self._handle_visualize_all_request)
        self._event_bus.subscribe(event_type=MarkVisualizeCancelRequestEventData, handler=self._handle_visualize_cancel_request)

        # Centralized command events
        self._event_bus.subscribe(event_type=MarkCommandParsedEvent, handler=self._handle_mark_command_parsed)

        logger.debug("MarkService subscriptions set up")

    async def _handle_mark_command_parsed(self, parsed_mark_command: MarkCommandParsedEvent) -> None:
        """Handle parsed mark commands.

        Routes mark commands to the appropriate handler based on command type.
        Validates mark existence before execution for mark execute commands.
        """
        command = parsed_mark_command.command
        logger.debug("MarkService received mark command: %s", type(command).__name__)

        if isinstance(command, MarkExecuteCommand):
            if not await self._mark_exists(command.label):
                logger.warning("MarkService: Mark '%s' does not exist, ignoring execute command", command.label)
                return

        await self._execute_mark_command(command)

    async def _mark_exists(self, label: str) -> bool:
        """Check if a mark with the given label exists using cached lookup."""
        marks_data = await self._storage.read(model_type=MarksData)
        return label.lower().strip() in marks_data.marks

    async def _execute_mark_command(self, command: BaseCommand) -> None:
        """Execute mark commands (create, execute, delete, visualize, reset).

        Processes mark command types through appropriate handlers, performs actions
        (create mark, click at mark, delete mark, etc.), and publishes status events.

        Args:
            command: Parsed mark command to execute
        """
        if isinstance(command, MarkCreateCommand):
            ix, iy = int(round(command.x)), int(round(command.y))
            mark_created, create_msg = await self._add_mark(command.label, ix, iy)
            if mark_created:
                message = f"Mark '{command.label}' created at ({ix}, {iy})."
                logger.info(message)
                await self._event_bus.publish(MarkCreatedEventData(name=command.label, x=ix, y=iy))
            else:
                message = f"Failed to create mark '{command.label}': {create_msg}"
                logger.warning(message)

        elif isinstance(command, MarkExecuteCommand):
            coords = await self._get_mark_coordinates(command.label)
            if coords:
                x, y = coords

                logger.debug(f"Moving mouse to ({x}, {y}) and clicking for mark '{command.label}'")
                pyautogui.click(x, y)

                message = f"Navigated to mark '{command.label}' at ({x}, {y}) and clicked."
                logger.info(message)

                await self._event_bus.publish(
                    MarkOperationSuccessEventData(
                        operation="execute", label=command.label, message=message, marks_data={"x": x, "y": y}
                    )
                )
            else:
                message = f"Mark '{command.label}' not found."
                logger.warning(message)

        elif isinstance(command, MarkDeleteCommand):
            deleted = await self._remove_mark(command.label)
            if deleted:
                message = f"Mark '{command.label}' deleted."
                logger.info(message)
                await self._event_bus.publish(MarkDeletedEventData(name=command.label))
            else:
                message = f"Mark '{command.label}' not found."
                logger.warning(message)

        elif isinstance(command, MarkVisualizeCommand):
            # Publish marks data BEFORE visualizing
            await self._publish_marks_changed_event()
            await self.visualize_marks(True)
            message = "Mark visualization activated."
            logger.info(message)

        elif isinstance(command, MarkResetCommand):
            num_cleared = await self._reset_all_marks()
            message = f"All {num_cleared} marks have been reset."
            logger.info(message)
            await self._publish_marks_changed_event()

        elif isinstance(command, MarkVisualizeCancelCommand):
            await self.visualize_marks(False)
            message = "Mark visualization cancelled."
            logger.info(message)

        else:
            message = f"Unknown mark command: {type(command)}"
            logger.error(message)

    async def _is_label_valid(self, label: str) -> Tuple[bool, str]:
        """Validate mark label using protected terms validator.

        Args:
            label: The mark label to validate

        Returns:
            Tuple of (is_valid, error_message) where error_message is empty if valid
        """
        is_valid, error_msg = await self._protected_terms_validator.validate_term(term=label)

        if not is_valid:
            return False, error_msg

        if await self._mark_exists(label.lower().strip()):
            return False, f"Mark label '{label}' is already in use."

        return True, ""

    async def _add_mark(self, label: str, x: int, y: int) -> Tuple[bool, str]:
        """Add a new mark using unified storage."""
        normalized_label = label.lower().strip()
        is_valid, reason = await self._is_label_valid(normalized_label)
        if not is_valid:
            logger.warning(f"Failed to add mark '{label}' (normalized: '{normalized_label}'): {reason}")
            return False, reason

        # Load current marks, add new one, save
        marks_data = await self._storage.read(model_type=MarksData)
        marks_data.marks[normalized_label] = Coordinate(x=x, y=y)
        success = await self._storage.write(data=marks_data)

        if success:
            self._protected_terms_validator.invalidate_cache()
            logger.info(f"Added mark '{normalized_label}' at ({x}, {y})")
            return True, f"Mark '{normalized_label}' created."
        else:
            logger.error(f"Failed to save mark '{normalized_label}' to storage")
            return False, "Failed to save mark to storage."

    async def _get_mark_coordinates(self, label: str) -> Optional[Tuple[int, int]]:
        """Get coordinates for a mark using cached unified storage."""
        marks_data = await self._storage.read(model_type=MarksData)
        mark_coord = marks_data.marks.get(label.lower().strip())
        if mark_coord:
            return (mark_coord.x, mark_coord.y)
        return None

    async def _get_all_marks(self) -> Dict[str, Tuple[int, int]]:
        """Get all marks using unified storage."""
        marks_data = await self._storage.read(model_type=MarksData)
        return {name: (coord.x, coord.y) for name, coord in marks_data.marks.items()}

    async def _remove_mark(self, label: str) -> bool:
        """Remove a mark using unified storage."""
        normalized_label = label.lower().strip()
        marks_data = await self._storage.read(model_type=MarksData)
        if normalized_label in marks_data.marks:
            del marks_data.marks[normalized_label]
            success = await self._storage.write(data=marks_data)
            if success:
                self._protected_terms_validator.invalidate_cache()
                logger.info(f"Removed mark '{normalized_label}'")
            return success
        else:
            logger.warning(f"Attempted to remove non-existent mark '{normalized_label}'")
            return True

    async def _reset_all_marks(self) -> int:
        """Reset all marks and return count of cleared marks."""
        all_marks = await self._get_all_marks()
        num_cleared = len(all_marks)

        marks_data = MarksData(marks={})
        success = await self._storage.write(data=marks_data)
        if success:
            self._protected_terms_validator.invalidate_cache()
            logger.info(f"All {num_cleared} marks have been reset.")
        else:
            logger.error("Failed to reset marks in storage")

        return num_cleared

    async def _publish_marks_changed_event(self) -> None:
        """Publish marks changed event for UI updates."""
        all_marks = await self.get_all_marks()

        await self._event_bus.publish(MarksChangedEventData(marks=all_marks))

        logger.debug(f"Published marks changed event - {len(all_marks)} marks")

    async def visualize_marks(self, show: bool) -> None:
        """Toggle mark visualization."""
        async with self._viz_lock:
            self._is_viz_active = show

        state_event = MarkVisualizationStateChangedEventData(is_visible=show)
        await self._event_bus.publish(state_event)

        logger.debug(f"Mark visualization {'activated' if show else 'deactivated'}")

    async def get_mark_coordinates(self, name: str) -> Optional[Tuple[int, int]]:
        """Public interface to get mark coordinates."""
        return await self._get_mark_coordinates(name)

    async def get_all_marks(self) -> Dict[str, Dict[str, Any]]:
        """Get all marks formatted for UI display."""
        marks = await self._get_all_marks()
        return {name: {"name": name, "x": coords[0], "y": coords[1]} for name, coords in marks.items()}

    async def stop_service_tasks(self) -> None:
        """Allow pending mark writes to finish before shutdown continues."""
        await asyncio.sleep(self._config.mark.shutdown_grace_period_seconds)
        logger.debug("MarkService shutdown grace complete")

    # UI Event Handlers - simplified with unified storage
    async def _handle_get_all_request(self, _request: MarkGetAllRequestEventData) -> None:
        """Handle get all marks request."""
        marks = await self.get_all_marks()
        logger.debug("Mark get-all request: %s marks", len(marks))
        await self._event_bus.publish(MarksChangedEventData(marks=marks))

    async def _handle_create_mark_request(self, create_request: MarkCreateRequestEventData) -> None:
        """Handle create mark request from UI."""
        success, message = await self._add_mark(create_request.name, create_request.x, create_request.y)
        if success:
            await self._publish_marks_changed_event()
        logger.debug(f"Handled create mark request - {message}")

    async def _handle_delete_by_name_request(self, delete_request: MarkDeleteByNameRequestEventData) -> None:
        """Handle delete mark by name request."""
        success = await self._remove_mark(delete_request.name)
        if success:
            await self._publish_marks_changed_event()
        logger.debug(f"Handled delete mark request - {'success' if success else 'failed'}")

    async def _handle_delete_all_request(self, _request: MarkDeleteAllRequestEventData) -> None:
        """Handle delete all marks request."""
        num_cleared = await self._reset_all_marks()
        await self._publish_marks_changed_event()
        logger.debug(f"Handled delete all marks request - {num_cleared} marks cleared")

    async def _handle_execute_mark_request(self, execute_request: MarkExecuteRequestEventData) -> None:
        """Handle execute mark request."""
        label = str(execute_request.name_or_id)
        coords = await self._get_mark_coordinates(label)
        if coords:
            x, y = coords
            logger.debug("Execute mark UI request: %s at (%s, %s)", label, x, y)
            pyautogui.click(x, y)
            message = f"Navigated to mark '{label}' at ({x}, {y}) and clicked."
            await self._event_bus.publish(
                MarkOperationSuccessEventData(operation="execute", label=label, message=message, marks_data={"x": x, "y": y})
            )
        else:
            message = f"Mark '{label}' not found for execution"
            logger.warning(message)

    async def _handle_visualize_all_request(self, _request: MarkVisualizeAllRequestEventData) -> None:
        """Handle visualize all marks request."""
        await self.visualize_marks(True)

    async def _handle_visualize_cancel_request(self, _request: MarkVisualizeCancelRequestEventData) -> None:
        """Handle cancel visualization request."""
        await self.visualize_marks(False)
