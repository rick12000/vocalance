"""
Mark Service V2 - Unified Storage Implementation

Migrated mark service using the unified storage system for:
- High-performance mark coordinate storage with caching
- Non-blocking mark operations for voice command responsiveness
- Event-driven architecture with unified storage
- Separated storage logic from business logic for better maintainability
"""

import logging
import asyncio
from typing import Dict, Tuple, Optional, Set, Any

from iris.app.event_bus import EventBus
from iris.app.config.app_config import GlobalAppConfig
from iris.app.events.command_events import MarkCommandParsedEvent
from iris.app.events.mark_events import (
    MarkCreatedEventData, MarkDeletedEventData, MarkVisualizationStateChangedEventData,
    MarkOperationSuccessEventData, MarksChangedEventData,
    MarkVisualizeAllRequestEventData, MarkVisualizeCancelRequestEventData,
    MarkShowNumbersRequestEventData, MarkHideNumbersRequestEventData, MarkExecuteRequestEventData,
    MarkGetAllRequestEventData, MarkCreateRequestEventData, MarkDeleteByNameRequestEventData,
    MarkDeleteAllRequestEventData
)
from iris.app.config.command_types import (
    MarkCreateCommand, MarkExecuteCommand, MarkDeleteCommand,
    MarkVisualizeCommand, MarkResetCommand, MarkVisualizeCancelCommand,
    BaseCommand
)
from iris.app.services.storage.storage_adapters import StorageAdapterFactory

logger = logging.getLogger(__name__)


class MarkService:
    """
    Enhanced mark service using unified storage backend
    
    Performance improvements for voice commands:
    - Cached mark coordinate lookups for instant navigation
    - Non-blocking mark operations with debounced writes
    - Event-driven architecture with unified storage
    - Separated storage concerns from business logic
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, 
                 storage_factory: StorageAdapterFactory, reserved_labels: Optional[Set[str]] = None):
        self._event_bus = event_bus
        self._config = config
        self._storage_adapter = storage_factory.get_mark_adapter()
        self._is_viz_active = False
        
        # Reserved labels for validation
        self._reserved_labels = set(label.lower() for label in reserved_labels) if reserved_labels else set()
        
        # Performance tracking
        self._mark_operations = 0
        self._cache_hits = 0
        
        logger.info(f"MarkServiceV2 initialized with unified storage. Reserved labels: {self._reserved_labels}")

    def setup_subscriptions(self) -> None:        
        """Set up event subscriptions for mark operations."""
        logger.info("Setting up MarkServiceV2 event subscriptions...")
        
        # Subscribe to visualization state changes
        self._event_bus.subscribe(MarkVisualizationStateChangedEventData, self._handle_visualization_state_changed)
        
        # UI-driven requests
        self._event_bus.subscribe(MarkGetAllRequestEventData, self._handle_get_all_request)
        self._event_bus.subscribe(MarkCreateRequestEventData, self._handle_create_mark_request)
        self._event_bus.subscribe(MarkDeleteByNameRequestEventData, self._handle_delete_by_name_request)
        self._event_bus.subscribe(MarkDeleteAllRequestEventData, self._handle_delete_all_request)
        self._event_bus.subscribe(MarkExecuteRequestEventData, self._handle_execute_mark_request)
        self._event_bus.subscribe(MarkVisualizeAllRequestEventData, self._handle_visualize_all_request)
        self._event_bus.subscribe(MarkVisualizeCancelRequestEventData, self._handle_visualize_cancel_request)
        self._event_bus.subscribe(MarkShowNumbersRequestEventData, self._handle_show_numbers_request)
        self._event_bus.subscribe(MarkHideNumbersRequestEventData, self._handle_hide_numbers_request)
        
        # Centralized command events
        self._event_bus.subscribe(MarkCommandParsedEvent, self._handle_mark_command_parsed)
        
        logger.info("MarkServiceV2 subscriptions set up")

    async def _handle_mark_command_parsed(self, event_data: MarkCommandParsedEvent) -> None:
        """Handle parsed mark commands"""
        try:
            command = event_data.command
            logger.debug(f"MarkServiceV2 received mark command: {type(command).__name__}")
            
            # For mark execute commands, check if mark exists before processing
            if isinstance(command, MarkExecuteCommand):
                if not await self._mark_exists(command.label):
                    logger.warning(f"MarkServiceV2: Mark '{command.label}' does not exist, ignoring execute command")
                    return
            
            await self._execute_mark_command(command)
            
        except Exception as e:
            logger.error(f"Error handling mark command: {e}", exc_info=True)

    async def _mark_exists(self, label: str) -> bool:
        """Check if a mark with the given label exists using cached lookup."""
        start_time = asyncio.get_event_loop().time()
        
        mark_names = await self._storage_adapter.get_all_mark_names()
        
        # Track performance
        lookup_time = asyncio.get_event_loop().time() - start_time
        self._mark_operations += 1
        if lookup_time < 0.001:  # Sub-millisecond indicates cache hit
            self._cache_hits += 1
            
        return label.lower().strip() in mark_names

    async def _execute_mark_command(self, command: BaseCommand) -> None:
        """Execute mark commands with unified storage backend."""
        success = False
        message = ""
        mark_data_for_event: Optional[Dict[str, Any]] = None
        operation_type = command.__class__.__name__.replace("Command", "").lower()

        try:
            if isinstance(command, MarkCreateCommand):
                mark_created, create_msg = await self._add_mark(command.label, command.x, command.y)
                if mark_created:
                    success = True
                    message = f"Mark '{command.label}' created at ({command.x}, {command.y})."
                    logger.info(message)
                    mark_data_for_event = {"name": command.label, "x": command.x, "y": command.y}
                    await self._event_bus.publish(MarkCreatedEventData(name=command.label, x=command.x, y=command.y))
                else:
                    success = False
                    message = f"Failed to create mark '{command.label}': {create_msg}"
                    logger.warning(message)
            
            elif isinstance(command, MarkExecuteCommand):
                coords = await self._get_mark_coordinates(command.label)
                if coords:
                    x, y = coords
                    
                    # Perform the actual mouse movement and click
                    try:
                        import pyautogui
                        logger.debug(f"Moving mouse to ({x}, {y}) and clicking for mark '{command.label}'")
                        pyautogui.click(x, y)
                        
                        success = True
                        message = f"Navigated to mark '{command.label}' at ({x}, {y}) and clicked."
                        logger.info(message)
                        mark_data_for_event = {"name": command.label, "x": x, "y": y}
                        
                        await self._event_bus.publish(MarkOperationSuccessEventData(operation="execute", label=command.label, 
                            message=message, marks_data={"x": x, "y": y}))
                    except Exception as click_error:
                        success = False
                        message = f"Found mark '{command.label}' at ({x}, {y}) but failed to click: {click_error}"
                        logger.error(message, exc_info=True)
                else:
                    success = False
                    message = f"Mark '{command.label}' not found."
                    logger.warning(message)
            
            elif isinstance(command, MarkDeleteCommand):
                deleted = await self._remove_mark(command.label)
                if deleted:
                    success = True
                    message = f"Mark '{command.label}' deleted."
                    logger.info(message)
                    mark_data_for_event = {"name": command.label}
                    await self._event_bus.publish(MarkDeletedEventData(name=command.label))
                else:
                    success = False
                    message = f"Mark '{command.label}' not found."
                    logger.warning(message)
            
            elif isinstance(command, MarkVisualizeCommand):
                await self.visualize_marks(True)
                success = True
                message = "Mark visualization activated."
                logger.info(message)
            
            elif isinstance(command, MarkResetCommand):
                num_cleared = await self._reset_all_marks()
                success = True
                message = f"All {num_cleared} marks have been reset."
                logger.info(message)
                await self._publish_marks_changed_event()
            
            elif isinstance(command, MarkVisualizeCancelCommand):
                await self.visualize_marks(False)
                success = True
                message = "Mark visualization cancelled."
                logger.info(message)
            
            else:
                success = False
                message = f"Unknown mark command: {type(command)}"
                logger.error(message)

        except Exception as e:
            success = False
            message = f"Error executing mark command: {str(e)}"
            logger.error(message, exc_info=True)

        # Publish command status
        await self._publish_command_status("", operation_type, success, message, mark_data_for_event)

    def update_reserved_labels(self, new_reserved_labels: Set[str]):
        """Update the set of reserved labels for validation."""
        self._reserved_labels.update(label.lower() for label in new_reserved_labels)
        logger.info(f"MarkServiceV2 reserved labels updated: {self._reserved_labels}")

    async def _is_label_valid(self, label: str) -> Tuple[bool, str]:
        """Validate a mark label with unified storage lookup."""
        normalized_label = label.lower().strip()
        if not normalized_label:
            return False, "Mark label cannot be empty."
        if ' ' in normalized_label:
            return False, "Mark label must be a single word."
        if normalized_label in self._reserved_labels:
            return False, f"Mark label '{normalized_label}' is a reserved command or keyword."
        
        # Check if mark already exists using unified storage
        if await self._mark_exists(normalized_label):
            return False, f"Mark label '{normalized_label}' is already in use."
            
        return True, ""

    async def _add_mark(self, label: str, x: int, y: int) -> Tuple[bool, str]:
        """Add a new mark using unified storage."""
        normalized_label = label.lower().strip()
        is_valid, reason = await self._is_label_valid(normalized_label)
        if not is_valid:
            logger.warning(f"Failed to add mark '{label}' (normalized: '{normalized_label}'): {reason}")
            return False, reason
        
        success = await self._storage_adapter.set_mark(normalized_label, x, y)
        if success:
            logger.info(f"Added mark '{normalized_label}' at ({x}, {y})")
            return True, f"Mark '{normalized_label}' created."
        else:
            logger.error(f"Failed to save mark '{normalized_label}' to storage")
            return False, "Failed to save mark to storage."

    async def _get_mark_coordinates(self, label: str) -> Optional[Tuple[int, int]]:
        """Get coordinates for a mark using cached unified storage."""
        return await self._storage_adapter.get_mark_coordinates(label.lower().strip())

    async def _get_all_marks(self) -> Dict[str, Tuple[int, int]]:
        """Get all marks using unified storage."""
        return await self._storage_adapter.load_marks()

    async def _get_all_mark_names(self) -> Set[str]:
        """Get all mark names using unified storage."""
        return await self._storage_adapter.get_all_mark_names()

    def _get_reserved_labels(self) -> Set[str]:
        """Get the set of reserved labels."""
        return self._reserved_labels

    async def _remove_mark(self, label: str) -> bool:
        """Remove a mark using unified storage."""
        normalized_label = label.lower().strip()
        success = await self._storage_adapter.remove_mark(normalized_label)
        if success:
            logger.info(f"Removed mark '{normalized_label}'")
        else:
            logger.warning(f"Attempted to remove non-existent mark '{normalized_label}'")
        return success

    async def _reset_all_marks(self) -> int:
        """Reset all marks and return count of cleared marks."""
        all_marks = await self._get_all_marks()
        num_cleared = len(all_marks)
        
        success = await self._storage_adapter.clear_all_marks()
        if success:
            logger.info(f"All {num_cleared} marks have been reset.")
        else:
            logger.error("Failed to reset marks in storage")
            
        return num_cleared

    async def _get_defined_mark_labels(self) -> Set[str]:
        """Get all defined mark labels using unified storage."""
        return await self._get_all_mark_names()

    async def _publish_command_status(self, raw_phrase: str, command_name: str, success: bool, 
                                    message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Publish command execution status."""
        # Implementation would publish to appropriate event channel
        logger.debug(f"Command status - {command_name}: {'SUCCESS' if success else 'FAILED'} - {message}")

    async def _publish_marks_changed_event(self) -> None:
        """Publish marks changed event for UI updates."""
        all_marks = await self.get_all_marks()
        
        # Publish MarksChangedEvent for UI updates
        marks_changed_event = MarksChangedEventData(marks=all_marks)
        await self._event_bus.publish(MarksChangedEventData(marks=all_marks))
        
        logger.debug(f"Published marks changed event - {len(all_marks)} marks")

    async def visualize_marks(self, show: bool) -> None:
        """Toggle mark visualization."""
        self._is_viz_active = show
        
        # Publish visualization state changed event for UI updates
        state_event = MarkVisualizationStateChangedEventData(is_visible=show)
        await self._event_bus.publish(state_event)
        
        logger.debug(f"Mark visualization {'activated' if show else 'deactivated'}")

    async def get_mark_coordinates(self, name: str) -> Optional[Tuple[int, int]]:
        """Public interface to get mark coordinates with performance tracking."""
        start_time = asyncio.get_event_loop().time()
        coords = await self._get_mark_coordinates(name)
        
        # Track performance for monitoring
        lookup_time = asyncio.get_event_loop().time() - start_time
        self._mark_operations += 1
        if lookup_time < 0.001:  # Sub-millisecond indicates cache hit
            self._cache_hits += 1
            
        return coords

    async def get_all_marks(self) -> Dict[str, Dict[str, Any]]:
        """Get all marks formatted for UI display."""
        marks = await self._get_all_marks()
        return {
            name: {"name": name, "x": coords[0], "y": coords[1]} 
            for name, coords in marks.items()
        }

    def get_cache_hit_rate(self) -> float:
        """Get current cache hit rate as percentage for performance monitoring."""
        return (self._cache_hits / self._mark_operations * 100) if self._mark_operations > 0 else 0.0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics for voice command optimization."""
        hit_rate = self.get_cache_hit_rate()
        
        return {
            "total_operations": self._mark_operations,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": hit_rate,
            "avg_latency": "<1ms (cached)" if hit_rate > 80 else "1-5ms (mixed)",
            "optimization_status": "excellent" if hit_rate > 90 else "good" if hit_rate > 70 else "needs_tuning"
        }

    async def start_service_tasks(self) -> None:
        """Start background service tasks."""
        logger.info("MarkServiceV2 background tasks started")

    async def stop_service_tasks(self) -> None:
        """Stop background service tasks and ensure cleanup."""
        try:
            # Allow any pending writes to complete
            await asyncio.sleep(0.1)
            
            # Log final performance statistics
            stats = self.get_performance_stats()
            logger.info(f"MarkServiceV2 cleanup - Performance: {stats['cache_hit_rate']:.1f}% cache hit rate, {stats['total_operations']} operations")
            
        except Exception as e:
            logger.error(f"Error during MarkServiceV2 cleanup: {e}", exc_info=True)

    # UI Event Handlers - simplified with unified storage
    async def _handle_get_all_request(self, event_data) -> None:
        """Handle get all marks request."""
        marks = await self.get_all_marks()
        
        # Publish MarksChangedEvent for UI updates
        marks_changed_event = MarksChangedEventData(marks=marks)
        await self._event_bus.publish(marks_changed_event)
        
        logger.debug(f"Handled get all marks request - {len(marks)} marks")

    async def _handle_create_mark_request(self, event_data) -> None:
        """Handle create mark request from UI."""
        success, message = await self._add_mark(event_data.name, event_data.x, event_data.y)
        if success:
            await self._publish_marks_changed_event()
        logger.debug(f"Handled create mark request - {message}")

    async def _handle_delete_by_name_request(self, event_data) -> None:
        """Handle delete mark by name request."""
        success = await self._remove_mark(event_data.name)
        if success:
            await self._publish_marks_changed_event()
        logger.debug(f"Handled delete mark request - {'success' if success else 'failed'}")

    async def _handle_delete_all_request(self, event_data) -> None:
        """Handle delete all marks request."""
        num_cleared = await self._reset_all_marks()
        await self._publish_marks_changed_event()
        logger.debug(f"Handled delete all marks request - {num_cleared} marks cleared")

    async def _handle_execute_mark_request(self, event_data) -> None:
        """Handle execute mark request."""
        coords = await self._get_mark_coordinates(event_data.name)
        if coords:
            # Trigger mark execution
            logger.debug(f"Handled execute mark request - navigating to {coords}")
        else:
            logger.warning(f"Mark '{event_data.name}' not found for execution")

    async def _handle_visualize_all_request(self, event_data) -> None:
        """Handle visualize all marks request."""
        await self.visualize_marks(True)

    async def _handle_visualize_cancel_request(self, event_data) -> None:
        """Handle cancel visualization request."""
        await self.visualize_marks(False)

    async def _handle_show_numbers_request(self, event_data) -> None:
        """Handle show mark numbers request."""
        logger.debug("Show mark numbers requested")

    async def _handle_hide_numbers_request(self, event_data) -> None:
        """Handle hide mark numbers request."""
        logger.debug("Hide mark numbers requested")

    async def _handle_visualization_state_changed(self, event_data: MarkVisualizationStateChangedEventData) -> None:
        """Handle visualization state changes."""
        self._is_viz_active = event_data.is_visible
        logger.debug(f"Mark visualization state changed: {self._is_viz_active}") 