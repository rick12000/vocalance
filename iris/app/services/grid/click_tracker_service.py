import logging
import random
import time
from typing import Any, Dict, List

from iris.app.config.app_config import GlobalAppConfig
from iris.app.event_bus import EventBus
from iris.app.events.core_events import ClickLoggedEventData, PerformMouseClickEventData
from iris.app.events.grid_events import ClickCountsForGridEventData, RequestClickCountsForGridEventData
from iris.app.services.storage.storage_models import GridClickEvent, GridClicksData
from iris.app.services.storage.storage_service import StorageService
from iris.app.utils.event_utils import EventSubscriptionManager, ThreadSafeEventPublisher

logger = logging.getLogger(__name__)


def prioritize_grid_rects(rect_details_with_clicks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort rectangles by click frequency for grid optimization."""
    if not rect_details_with_clicks:
        return []

    for item in rect_details_with_clicks:
        item["rand_tiebreak"] = random.random()

    def sort_key(rect_item):
        clicks = rect_item.get("clicks", 0)
        if not isinstance(clicks, (int, float)):
            clicks = 0
        return (-clicks, rect_item["rand_tiebreak"])

    return sorted(rect_details_with_clicks, key=sort_key)


class ClickTrackerService:
    """Click tracking service with debounced storage for grid optimization.

    Records mouse clicks with position and timestamp, aggregates click counts per
    grid cell, and provides click frequency data for grid layout optimization.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService) -> None:
        self._event_bus = event_bus
        self._config = config
        self._storage = storage

        self.event_publisher = ThreadSafeEventPublisher(event_bus=event_bus)
        self.subscription_manager = EventSubscriptionManager(event_bus=event_bus, component_name="ClickTrackerService")

        logger.info("ClickTrackerService initialized")

    def setup_subscriptions(self) -> None:
        subscriptions = [
            (PerformMouseClickEventData, self._handle_mouse_click),
            (RequestClickCountsForGridEventData, self._handle_click_counts_request),
        ]

        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)

        logger.info("ClickTrackerService subscriptions set up")

    async def _handle_mouse_click(self, event_data: PerformMouseClickEventData) -> None:
        timestamp = time.time()

        # Load current clicks, append new one, save
        clicks_data = await self._storage.read(model_type=GridClicksData)
        new_click = GridClickEvent(x=event_data.x, y=event_data.y, timestamp=timestamp, cell_id=None)
        clicks_data.clicks.append(new_click)
        success = await self._storage.write(data=clicks_data)

        if success:
            click_logged_event = ClickLoggedEventData(x=event_data.x, y=event_data.y, timestamp=timestamp)
            self.event_publisher.publish(click_logged_event)
            logger.debug(f"Click logged: ({event_data.x}, {event_data.y})")

    async def _handle_click_counts_request(self, event_data: RequestClickCountsForGridEventData) -> None:
        try:
            clicks_data = await self._storage.read(model_type=GridClicksData)
            # Convert GridClickEvent objects to dictionaries for compatibility with existing logic
            all_clicks = [click.model_dump() for click in clicks_data.clicks]
            processed_rects = self._calculate_click_counts(all_clicks, event_data.rect_definitions)

            response_event = ClickCountsForGridEventData(
                request_id=event_data.request_id, processed_rects_with_clicks=processed_rects
            )

            self.event_publisher.publish(response_event)
            logger.debug(f"Published click counts for request {event_data.request_id}")

        except Exception as e:
            logger.error(f"Error processing click counts request: {e}", exc_info=True)

    def _calculate_click_counts(
        self, all_clicks: List[Dict[str, Any]], rect_definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
        try:
            click_x, click_y = click.get("x", 0), click.get("y", 0)
            return rect_x <= click_x <= rect_x + rect_w and rect_y <= click_y <= rect_y + rect_h
        except (TypeError, ValueError):
            return False

    async def get_click_statistics(self) -> Dict[str, Any]:
        try:
            clicks_data = await self._storage.read(model_type=GridClicksData)
            all_clicks = [click.model_dump() for click in clicks_data.clicks]

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

        except Exception as e:
            logger.error(f"Error getting click statistics: {e}", exc_info=True)
            return {"error": str(e)}

    async def cleanup(self) -> None:
        self.subscription_manager.unsubscribe_all()
        logger.info("ClickTrackerService cleanup complete")
