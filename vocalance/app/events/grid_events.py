from typing import Any, Dict, Literal, Optional

from vocalance.app.events.base_event import BaseEvent


class GridStateEvent(BaseEvent):
    """Unified event for grid state changes and requests."""

    state: Literal["visible", "hidden", "config_updated", "interaction_request", "interaction_success", "interaction_failed"]
    config: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
