"""
Click Tracker Service V2 - Unified Storage Implementation

Migrated click tracker service using the unified storage system for:
- High-performance click logging with debounced writes
- Non-blocking operations for voice command responsiveness
- Cached click history for grid optimization
- Event-driven architecture with improved reliability
"""

import logging
import asyncio
import time
import random
from typing import List, Dict, Any

from iris.event_bus import EventBus
from iris.config.app_config import GlobalAppConfig
from iris.events.core_events import PerformMouseClickEventData, ClickLoggedEventData
from iris.events.grid_events import RequestClickCountsForGridEventData, ClickCountsForGridEventData
from iris.utils.event_utils import ThreadSafeEventPublisher, EventSubscriptionManager
from iris.services.storage.storage_adapters import StorageAdapterFactory

logger = logging.getLogger(__name__)


def prioritize_grid_rects(rect_details_with_clicks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort rectangles by click count (descending) with random tie-breaker."""
    if not rect_details_with_clicks:
        return []

    for item in rect_details_with_clicks:
        item['rand_tiebreak'] = random.random()

    def sort_key(rect_item):
        clicks = rect_item.get('clicks', 0)
        if not isinstance(clicks, (int, float)):
            clicks = 0
        return (-clicks, rect_item['rand_tiebreak'])

    return sorted(rect_details_with_clicks, key=sort_key)


class ClickTrackerService:
    """Click tracker service for grid optimization using cached click history."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage_factory: StorageAdapterFactory):
        self._event_bus = event_bus
        self._config = config
        self._storage_adapter = storage_factory.get_grid_click_adapter()
        
        self.event_publisher = ThreadSafeEventPublisher(event_bus)
        self.subscription_manager = EventSubscriptionManager(event_bus, "ClickTrackerService")
        
        logger.info("ClickTrackerService initialized")

    def setup_subscriptions(self) -> None:
        """Set up event subscriptions."""
        subscriptions = [
            (PerformMouseClickEventData, self._handle_mouse_click),
            (RequestClickCountsForGridEventData, self._handle_click_counts_request)
        ]
        
        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)
        
        logger.info("ClickTrackerService subscriptions set up")

    async def _handle_mouse_click(self, event_data: PerformMouseClickEventData) -> None:
        """Handle mouse click logging."""
        click_data = {
            "x": event_data.x,
            "y": event_data.y,
            "timestamp": time.time(),
            "source": event_data.source
        }
        
        success = await self._storage_adapter.append_click(click_data)
        
        if success:
            click_logged_event = ClickLoggedEventData(
                x=event_data.x, 
                y=event_data.y, 
                timestamp=click_data["timestamp"]
            )
            self.event_publisher.publish(click_logged_event)
            logger.debug(f"Click logged: ({event_data.x}, {event_data.y})")

    async def _handle_click_counts_request(self, event_data: RequestClickCountsForGridEventData) -> None:
        """Handle request for click counts in grid rectangles."""
        try:
            all_clicks = await self._storage_adapter.load_clicks()
            processed_rects = self._calculate_click_counts(all_clicks, event_data.rect_definitions)
            
            response_event = ClickCountsForGridEventData(
                request_id=event_data.request_id,
                processed_rects_with_clicks=processed_rects
            )
            
            self.event_publisher.publish(response_event)
            logger.debug(f"Published click counts for request {event_data.request_id}")
            
        except Exception as e:
            logger.error(f"Error processing click counts request: {e}", exc_info=True)

    def _calculate_click_counts(self, all_clicks: List[Dict[str, Any]], 
                              rect_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate click counts for each rectangle definition."""
        processed_rects = []
        
        for rect_def in rect_definitions:
            try:
                rect_x, rect_y = int(rect_def["x"]), int(rect_def["y"])
                rect_w, rect_h = int(rect_def["w"]), int(rect_def["h"])
                
                count = sum(
                    1 for click in all_clicks 
                    if self._is_click_in_rect(click, rect_x, rect_y, rect_w, rect_h)
                )
                
                processed_rects.append({"data": rect_def, "clicks": count})
                
            except (KeyError, ValueError, TypeError):
                processed_rects.append({"data": rect_def, "clicks": 0})
        
        return processed_rects

    def _is_click_in_rect(self, click: Dict[str, Any], rect_x: int, rect_y: int, 
                         rect_w: int, rect_h: int) -> bool:
        """Check if click falls within rectangle bounds."""
        try:
            click_x, click_y = click.get("x", 0), click.get("y", 0)
            return (rect_x <= click_x <= rect_x + rect_w and 
                    rect_y <= click_y <= rect_y + rect_h)
        except (TypeError, ValueError):
            return False

    async def get_click_statistics(self) -> Dict[str, Any]:
        """Get click statistics for monitoring."""
        try:
            all_clicks = await self._storage_adapter.load_clicks()
            
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
                "source_distribution": source_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting click statistics: {e}", exc_info=True)
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.subscription_manager.unsubscribe_all()
        logger.info("ClickTrackerService cleanup complete") 