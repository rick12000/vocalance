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
from vocalance.app.events.mark_events import MarksChangedEventData, MarkVisualizationStateChangedEventData
from vocalance.app.services.base_service import Service
from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.storage.storage_models import Coordinate, MarksData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class MarkService(Service):
    """Service for managing screen position marks with unified storage."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        storage: StorageService,
        protected_terms_validator: ProtectedTermsValidator,
    ) -> None:
        self._event_bus = event_bus
        self._config = config
        self._storage = storage
        self._protected_terms_validator = protected_terms_validator
        self._is_viz_active: bool = False
        event_bus.subscribe(MarkCommandParsedEvent, self._handle_mark_command_parsed)

    async def _handle_mark_command_parsed(self, parsed_mark_command: MarkCommandParsedEvent) -> None:
        command = parsed_mark_command.command
        logger.debug("MarkService received mark command: %s", type(command).__name__)

        if isinstance(command, MarkExecuteCommand):
            if not await self._mark_exists(command.label):
                logger.warning("MarkService: Mark '%s' does not exist, ignoring execute command", command.label)
                return

        await self._execute_mark_command(command)

    async def _mark_exists(self, label: str) -> bool:
        marks_data = await self._storage.read(model_type=MarksData)
        return label.lower().strip() in marks_data.marks

    async def _execute_mark_command(self, command: BaseCommand) -> None:
        if isinstance(command, MarkCreateCommand):
            ix, iy = int(round(command.x)), int(round(command.y))
            mark_created, create_msg = await self._add_mark(command.label, ix, iy)
            if mark_created:
                logger.info("Mark '%s' created at (%s, %s).", command.label, ix, iy)
                await self._publish_marks_changed()
            else:
                logger.warning("Failed to create mark '%s': %s", command.label, create_msg)

        elif isinstance(command, MarkExecuteCommand):
            coords = await self._get_mark_coordinates(command.label)
            if coords:
                x, y = coords
                pyautogui.click(x, y)
                logger.info("Navigated to mark '%s' at (%s, %s) and clicked.", command.label, x, y)
            else:
                logger.warning("Mark '%s' not found.", command.label)

        elif isinstance(command, MarkDeleteCommand):
            deleted = await self._remove_mark(command.label)
            if deleted:
                logger.info("Mark '%s' deleted.", command.label)
                await self._publish_marks_changed()
            else:
                logger.warning("Mark '%s' not found.", command.label)

        elif isinstance(command, MarkVisualizeCommand):
            await self._publish_marks_changed()
            await self.set_visualization(True)
            logger.info("Mark visualization activated.")

        elif isinstance(command, MarkResetCommand):
            num_cleared = await self._reset_all_marks()
            logger.info("All %s marks have been reset.", num_cleared)
            await self._publish_marks_changed()

        elif isinstance(command, MarkVisualizeCancelCommand):
            await self.set_visualization(False)
            logger.info("Mark visualization cancelled.")

        else:
            logger.error("Unknown mark command: %s", type(command))

    async def _is_label_valid(self, label: str) -> Tuple[bool, str]:
        is_valid, error_msg = await self._protected_terms_validator.validate_term(term=label)
        if not is_valid:
            return False, error_msg
        if await self._mark_exists(label.lower().strip()):
            return False, f"Mark label '{label}' is already in use."
        return True, ""

    async def _add_mark(self, label: str, x: int, y: int) -> Tuple[bool, str]:
        normalized_label = label.lower().strip()
        is_valid, reason = await self._is_label_valid(normalized_label)
        if not is_valid:
            logger.warning("Failed to add mark '%s': %s", label, reason)
            return False, reason

        marks_data = await self._storage.read(model_type=MarksData)
        marks_data.marks[normalized_label] = Coordinate(x=x, y=y)
        success = await self._storage.write(data=marks_data)

        if success:
            self._protected_terms_validator.invalidate_cache()
            logger.info("Added mark '%s' at (%s, %s)", normalized_label, x, y)
            return True, f"Mark '{normalized_label}' created."
        else:
            logger.error("Failed to save mark '%s' to storage", normalized_label)
            return False, "Failed to save mark to storage."

    async def _get_mark_coordinates(self, label: str) -> Optional[Tuple[int, int]]:
        marks_data = await self._storage.read(model_type=MarksData)
        mark_coord = marks_data.marks.get(label.lower().strip())
        if mark_coord:
            return (mark_coord.x, mark_coord.y)
        return None

    async def _get_all_marks(self) -> Dict[str, Tuple[int, int]]:
        marks_data = await self._storage.read(model_type=MarksData)
        return {name: (coord.x, coord.y) for name, coord in marks_data.marks.items()}

    async def _remove_mark(self, label: str) -> bool:
        normalized_label = label.lower().strip()
        marks_data = await self._storage.read(model_type=MarksData)
        if normalized_label in marks_data.marks:
            del marks_data.marks[normalized_label]
            success = await self._storage.write(data=marks_data)
            if success:
                self._protected_terms_validator.invalidate_cache()
                logger.info("Removed mark '%s'", normalized_label)
            return success
        logger.warning("Attempted to remove non-existent mark '%s'", normalized_label)
        return True

    async def _reset_all_marks(self) -> int:
        all_marks = await self._get_all_marks()
        num_cleared = len(all_marks)
        success = await self._storage.write(data=MarksData(marks={}))
        if success:
            self._protected_terms_validator.invalidate_cache()
            logger.info("All %s marks have been reset.", num_cleared)
        else:
            logger.error("Failed to reset marks in storage")
        return num_cleared

    async def _publish_marks_changed(self) -> None:
        all_marks = await self.get_all_marks()
        await self._event_bus.publish(MarksChangedEventData(marks=all_marks))

    async def set_visualization(self, show: bool) -> None:
        """Toggle mark visualization and broadcast the state change."""
        self._is_viz_active = show
        await self._event_bus.publish(MarkVisualizationStateChangedEventData(is_visible=show))

    # ── Public interface for direct callers (UI controllers) ──────────────────

    async def create_mark(self, name: Optional[str], x: int, y: int) -> Tuple[bool, str]:
        """Create a mark and broadcast the updated marks list. Returns (success, message)."""
        label = (name or "").lower().strip()
        success, msg = await self._add_mark(label, x, y)
        if success:
            await self._publish_marks_changed()
        return success, msg

    async def delete_mark(self, name: str) -> bool:
        """Delete a mark by name and broadcast the updated marks list."""
        success = await self._remove_mark(name)
        if success:
            await self._publish_marks_changed()
        return success

    async def delete_all_marks(self) -> int:
        """Delete all marks and broadcast the updated marks list. Returns count deleted."""
        num_cleared = await self._reset_all_marks()
        await self._publish_marks_changed()
        return num_cleared

    async def execute_mark(self, name_or_id: str) -> bool:
        """Click the screen position of a mark. Returns True if found and clicked."""
        coords = await self._get_mark_coordinates(str(name_or_id))
        if not coords:
            logger.warning("Mark '%s' not found for execution", name_or_id)
            return False
        x, y = coords
        pyautogui.click(x, y)
        logger.info("Executed mark '%s' at (%s, %s)", name_or_id, x, y)
        return True

    async def get_mark_coordinates(self, name: str) -> Optional[Tuple[int, int]]:
        return await self._get_mark_coordinates(name)

    async def get_all_marks(self) -> Dict[str, Dict[str, Any]]:
        marks = await self._get_all_marks()
        return {name: {"name": name, "x": coords[0], "y": coords[1]} for name, coords in marks.items()}

    async def shutdown(self) -> None:
        self._event_bus.unsubscribe(MarkCommandParsedEvent, self._handle_mark_command_parsed)
        await asyncio.sleep(self._config.mark.shutdown_grace_period_seconds)
