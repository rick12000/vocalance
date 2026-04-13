import logging
import threading
import time
from typing import Any, Dict, List

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.services.storage.storage_models import GridClickEvent, GridClicksData
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.utils.event_utils import EventSubscriptionManager, ThreadSafeEventPublisher

logger = logging.getLogger(__name__)


def prioritize_grid_rects(rect_details_with_clicks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort rectangles by click frequency (desc), then by stable screen position.

    Random tie-breaks caused visible renumbering when the grid repainted or when
    equal-click cells were re-sorted; geographic order is deterministic.
    """
    if not rect_details_with_clicks:
        return []

    def sort_key(rect_item: Dict[str, Any]) -> tuple:
        clicks = rect_item.get("clicks", 0)
        if not isinstance(clicks, (int, float)):
            clicks = 0
        data = rect_item.get("data") or {}
        try:
            x = float(data.get("x", 0))
            y = float(data.get("y", 0))
        except (TypeError, ValueError):
            x, y = 0.0, 0.0
        tie = rect_item.get("id", 0)
        try:
            tie_i = int(tie)
        except (TypeError, ValueError):
            tie_i = 0
        return (-float(clicks), y, x, tie_i)

    return sorted(rect_details_with_clicks, key=sort_key)


class ClickTrackerService:
    """Click tracking service with in-memory cache and startup/shutdown persistence.

    Architecture:
    - Maintains in-memory list of all clicks for fast access
    - Loads clicks from storage on initialization (startup)
    - Adds new clicks only to memory during session
    - Writes all clicks to storage only on shutdown

    This provides low-latency click tracking while ensuring persistence across sessions.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService) -> None:
        """Initialize click tracker service with dependencies.

        Args:
            event_bus: EventBus for pub/sub messaging.
            config: Global application configuration.
            storage: Storage service for persistent click data.
        """
        self._event_bus = event_bus
        self._config = config
        self._storage = storage

        # Thread-safe in-memory click cache
        self._lock = threading.RLock()
        self._clicks: List[GridClickEvent] = []
        self._loaded = False

        self.event_publisher = ThreadSafeEventPublisher(event_bus=event_bus)
        self.subscription_manager = EventSubscriptionManager(event_bus=event_bus, component_name="ClickTrackerService")

        logger.debug("ClickTrackerService initialized")

    async def initialize(self) -> None:
        """Load click history from storage into memory cache.

        Called once on application startup to populate the in-memory cache.
        """
        try:
            logger.info("Loading click history from storage...")
            clicks_data = await self._storage.read(model_type=GridClicksData)

            with self._lock:
                self._clicks = list(clicks_data.clicks)
                self._loaded = True

            logger.info(f"Loaded {len(self._clicks)} clicks from storage into memory cache")

        except Exception as e:
            logger.error(f"Failed to load click history from storage: {e}", exc_info=True)
            with self._lock:
                self._clicks = []
                self._loaded = True

    def setup_subscriptions(self) -> None:
        """Set up event subscriptions for click tracking."""
        subscriptions = [
            (PerformMouseClickEventData, self._handle_mouse_click),
        ]

        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)

        logger.debug("ClickTrackerService subscriptions set up")

    def _handle_mouse_click(self, click_request: PerformMouseClickEventData) -> None:
        """Handle mouse click event by adding to in-memory cache only.

        NO storage I/O here - just fast memory append.
        """
        timestamp = time.time()
        new_click = GridClickEvent(x=click_request.x, y=click_request.y, timestamp=timestamp, cell_id=None)

        with self._lock:
            self._clicks.append(new_click)
            click_count = len(self._clicks)

        logger.debug(f"Click logged to memory cache: ({click_request.x}, {click_request.y}) - total: {click_count}")

    def _calculate_click_counts(
        self, all_clicks: List[Dict[str, Any]], rect_definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate click counts per rectangle."""
        processed_rects = []

        for rect_def in rect_definitions:
            try:
                rect_x, rect_y = int(rect_def["x"]), int(rect_def["y"])
                rect_w, rect_h = int(rect_def["w"]), int(rect_def["h"])

                count = sum(1 for click in all_clicks if self._is_click_in_rect(click, rect_x, rect_y, rect_w, rect_h))

                processed_rects.append({"data": rect_def, "clicks": count})

            except (KeyError, ValueError, TypeError):
                processed_rects.append({"data": rect_def, "clicks": 0})

        return processed_rects

    def _is_click_in_rect(self, click: Dict[str, Any], rect_x: int, rect_y: int, rect_w: int, rect_h: int) -> bool:
        """Check if click is within rectangle bounds."""
        try:
            click_x, click_y = click.get("x", 0), click.get("y", 0)
            return rect_x <= click_x <= rect_x + rect_w and rect_y <= click_y <= rect_y + rect_h
        except (TypeError, ValueError):
            return False

    def get_all_clicks_sync(self) -> List[Dict[str, Any]]:
        """Get all clicks from memory cache synchronously.

        Used by grid view for prioritization - no async/storage overhead.

        Returns:
            List of click dictionaries with x, y, timestamp, cell_id
        """
        with self._lock:
            return [click.model_dump() for click in self._clicks]

    def get_click_statistics(self) -> Dict[str, Any]:
        """Get click statistics from in-memory cache."""
        with self._lock:
            all_clicks = [click.model_dump() for click in self._clicks]

        if not all_clicks:
            return {"total_clicks": 0}

        timestamps = [click.get("timestamp", 0) for click in all_clicks if click.get("timestamp")]
        sources = [click.get("source", "unknown") for click in all_clicks]

        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "total_clicks": len(all_clicks),
            "earliest_click": min(timestamps) if timestamps else 0,
            "latest_click": max(timestamps) if timestamps else 0,
            "source_distribution": source_counts,
        }

    async def shutdown(self) -> None:
        """Save all clicks from memory to storage on shutdown.

        Called once on application shutdown to persist all session data.
        """
        try:
            with self._lock:
                clicks_to_save = list(self._clicks)

            if not clicks_to_save:
                logger.info("No clicks to save on shutdown")
                return

            logger.info(f"Saving {len(clicks_to_save)} clicks to storage on shutdown...")

            clicks_data = GridClicksData(clicks=clicks_to_save)
            success = await self._storage.write(data=clicks_data)

            if success:
                logger.info(f"Successfully saved {len(clicks_to_save)} clicks to storage")
            else:
                logger.error("Failed to save clicks to storage on shutdown")

        except Exception as e:
            logger.error(f"Error saving clicks on shutdown: {e}", exc_info=True)

    async def cleanup(self) -> None:
        """Clean up resources and save data."""
        self.subscription_manager.unsubscribe_all()
        await self.shutdown()
        logger.info("ClickTrackerService cleanup complete")
