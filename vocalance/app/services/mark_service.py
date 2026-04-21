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
    MarksChangedEventData,
    MarkUiRequestEvent,
    MarkUiResponseEvent,
    MarkVisualizationStateChangedEventData,
)
from vocalance.app.services.base_service import Service
from vocalance.app.services.commands.utilities.input_executor import shared_input_executor
from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.storage.storage_models import Coordinate, MarksData
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class MarkService(Service):
    """Persist and execute screen marks, driven by parsed commands and UI requests."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        storage: StorageService,
        protected_terms_validator: ProtectedTermsValidator,
    ) -> None:
        self.event_bus = event_bus
        self.config = config
        self.storage = storage
        self.protected_terms_validator = protected_terms_validator
        self.is_viz_active: bool = False
        event_bus.subscribe(MarkCommandParsedEvent, self.handle_mark_command_parsed)
        event_bus.subscribe(MarkUiRequestEvent, self.handle_mark_ui_request)

    async def handle_mark_command_parsed(self, parsed_mark_command: MarkCommandParsedEvent) -> None:
        command: BaseCommand = parsed_mark_command.command

        if isinstance(command, MarkExecuteCommand):
            if not await self.mark_exists(command.label):
                logger.warning("Mark '%s' does not exist, ignoring execute command", command.label)
                return

        await self.execute_mark_command(command)

    async def mark_exists(self, label: str) -> bool:
        marks_data = await self.storage.read(model_type=MarksData)
        return label.lower().strip() in marks_data.marks

    async def execute_mark_command(self, command: BaseCommand) -> None:
        if isinstance(command, MarkCreateCommand):
            ix: int = int(round(command.x))
            iy: int = int(round(command.y))
            mark_created, create_msg = await self.add_mark(command.label, ix, iy)
            if mark_created:
                logger.info("Mark '%s' created at (%s, %s).", command.label, ix, iy)
                await self.publish_marks_changed()
            else:
                logger.warning("Failed to create mark '%s': %s", command.label, create_msg)

        elif isinstance(command, MarkExecuteCommand):
            coords = await self.get_mark_coordinates_internal(command.label)
            if coords:
                x, y = coords
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(shared_input_executor, pyautogui.click, x, y)
                logger.info("Navigated to mark '%s' at (%s, %s) and clicked.", command.label, x, y)
            else:
                logger.warning("Mark '%s' not found.", command.label)

        elif isinstance(command, MarkDeleteCommand):
            deleted = await self.remove_mark(command.label)
            if deleted:
                logger.info("Mark '%s' deleted.", command.label)
                await self.publish_marks_changed()
            else:
                logger.warning("Mark '%s' not found.", command.label)

        elif isinstance(command, MarkVisualizeCommand):
            await self.publish_marks_changed()
            await self.set_visualization(True)
            logger.info("Mark visualization activated.")

        elif isinstance(command, MarkResetCommand):
            num_cleared: int = await self.reset_all_marks()
            logger.info("All %s marks have been reset.", num_cleared)
            await self.publish_marks_changed()

        elif isinstance(command, MarkVisualizeCancelCommand):
            await self.set_visualization(False)
            logger.info("Mark visualization cancelled.")

        else:
            logger.error("Unknown mark command: %s", type(command))

    async def is_label_valid(self, label: str) -> Tuple[bool, str]:
        is_valid, error_msg = await self.protected_terms_validator.validate_term(term=label)
        if not is_valid:
            return False, error_msg
        if await self.mark_exists(label.lower().strip()):
            return False, f"Mark label '{label}' is already in use."
        return True, ""

    async def add_mark(self, label: str, x: int, y: int) -> Tuple[bool, str]:
        normalized_label: str = label.lower().strip()
        is_valid, reason = await self.is_label_valid(normalized_label)
        if not is_valid:
            logger.warning("Failed to add mark '%s': %s", label, reason)
            return False, reason

        marks_data = await self.storage.read(model_type=MarksData)
        marks_data.marks[normalized_label] = Coordinate(x=x, y=y)
        success: bool = await self.storage.write(data=marks_data)

        if success:
            self.protected_terms_validator.invalidate_cache()
            logger.info("Added mark '%s' at (%s, %s)", normalized_label, x, y)
            return True, f"Mark '{normalized_label}' created."
        logger.error("Failed to save mark '%s' to storage", normalized_label)
        return False, "Failed to save mark to storage."

    async def get_mark_coordinates_internal(self, label: str) -> Optional[Tuple[int, int]]:
        marks_data = await self.storage.read(model_type=MarksData)
        mark_coord = marks_data.marks.get(label.lower().strip())
        if mark_coord:
            return (mark_coord.x, mark_coord.y)
        return None

    async def get_all_marks_internal(self) -> Dict[str, Tuple[int, int]]:
        marks_data = await self.storage.read(model_type=MarksData)
        return {name: (coord.x, coord.y) for name, coord in marks_data.marks.items()}

    async def remove_mark(self, label: str) -> bool:
        normalized_label: str = label.lower().strip()
        marks_data = await self.storage.read(model_type=MarksData)
        if normalized_label in marks_data.marks:
            del marks_data.marks[normalized_label]
            success: bool = await self.storage.write(data=marks_data)
            if success:
                self.protected_terms_validator.invalidate_cache()
                logger.info("Removed mark '%s'", normalized_label)
            return success
        logger.warning("Attempted to remove non-existent mark '%s'", normalized_label)
        return True

    async def reset_all_marks(self) -> int:
        all_marks: Dict[str, Tuple[int, int]] = await self.get_all_marks_internal()
        num_cleared: int = len(all_marks)
        success: bool = await self.storage.write(data=MarksData(marks={}))
        if success:
            self.protected_terms_validator.invalidate_cache()
            logger.info("All %s marks have been reset.", num_cleared)
        else:
            logger.error("Failed to reset marks in storage")
        return num_cleared

    async def publish_marks_changed(self) -> None:
        all_marks: Dict[str, Dict[str, Any]] = await self.get_all_marks()
        await self.event_bus.publish(MarksChangedEventData(marks=all_marks))

    async def set_visualization(self, show: bool) -> None:
        """Toggle mark visualization and broadcast the state change.

        Args:
            show: When True, include current marks in the visualization event payload.
        """
        self.is_viz_active = show
        marks_payload: Optional[Dict[str, Dict[str, Any]]] = None
        if show:
            marks_payload = await self.get_all_marks()
        await self.event_bus.publish(MarkVisualizationStateChangedEventData(is_visible=show, marks=marks_payload))

    async def handle_mark_ui_request(self, event: MarkUiRequestEvent) -> None:
        op: str = event.op
        if op == "create":
            if event.x is None or event.y is None:
                return
            success, msg = await self.create_mark(event.name, event.x, event.y)
            await self.event_bus.publish(
                MarkUiResponseEvent(
                    kind="create_result",
                    success=success,
                    message=msg,
                    name=event.name or "",
                    x=event.x,
                    y=event.y,
                )
            )
        elif op == "delete" and event.mark_name:
            await self.delete_mark(event.mark_name)
        elif op == "delete_all":
            await self.delete_all_marks()
        elif op == "execute" and event.identifier:
            await self.execute_mark(event.identifier)
        elif op == "set_visualization" and event.visible is not None:
            await self.set_visualization(event.visible)
        elif op == "refresh_list":
            await self.publish_marks_changed()
        elif op == "prepare_overlay":
            marks: Dict[str, Dict[str, Any]] = await self.get_all_marks()
            await self.event_bus.publish(MarkUiResponseEvent(kind="overlay_marks", marks=marks))

    async def create_mark(self, name: Optional[str], x: int, y: int) -> Tuple[bool, str]:
        """Create a mark and broadcast the updated marks list.

        Returns:
            ``(success, message)`` for UI feedback.
        """
        label: str = (name or "").lower().strip()
        success, msg = await self.add_mark(label, x, y)
        if success:
            await self.publish_marks_changed()
        return success, msg

    async def delete_mark(self, name: str) -> bool:
        """Delete a mark by name and broadcast the updated marks list."""
        success: bool = await self.remove_mark(name)
        if success:
            await self.publish_marks_changed()
        return success

    async def delete_all_marks(self) -> int:
        """Delete all marks and broadcast the updated marks list.

        Returns:
            Number of marks cleared before persist (may differ if write fails).
        """
        num_cleared: int = await self.reset_all_marks()
        await self.publish_marks_changed()
        return num_cleared

    async def execute_mark(self, name_or_id: str) -> bool:
        """Move the cursor to a mark and click. Returns True if the mark existed."""
        coords = await self.get_mark_coordinates_internal(str(name_or_id))
        if not coords:
            logger.warning("Mark '%s' not found for execution", name_or_id)
            return False
        x, y = coords
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(shared_input_executor, pyautogui.click, x, y)
        logger.info("Executed mark '%s' at (%s, %s)", name_or_id, x, y)
        return True

    async def get_mark_coordinates(self, name: str) -> Optional[Tuple[int, int]]:
        """Return stored coordinates for ``name``, or None."""
        return await self.get_mark_coordinates_internal(name)

    async def get_all_marks(self) -> Dict[str, Dict[str, Any]]:
        """Return marks as UI-friendly dicts keyed by name."""
        marks: Dict[str, Tuple[int, int]] = await self.get_all_marks_internal()
        return {name: {"name": name, "x": coords[0], "y": coords[1]} for name, coords in marks.items()}

    async def shutdown(self) -> None:
        self.event_bus.unsubscribe(MarkCommandParsedEvent, self.handle_mark_command_parsed)
        self.event_bus.unsubscribe(MarkUiRequestEvent, self.handle_mark_ui_request)
        await asyncio.sleep(self.config.mark.shutdown_grace_period_seconds)
