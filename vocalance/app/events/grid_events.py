from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from vocalance.app.events.base_event import BaseEvent
from vocalance.app.services.storage.storage_models import GridClickEvent


class GridClickHistoryChangedEvent(BaseEvent):
    """Published when grid click history changes; carries a full in-memory snapshot for UI."""

    clicks_snapshot: List[GridClickEvent] = Field(
        default_factory=list,
        description="Current grid click history.",
    )


class GridStateEvent(BaseEvent):
    """Unified event for grid state changes and requests."""

    state: Literal["visible", "hidden", "config_updated", "interaction_request", "interaction_success", "interaction_failed"]
    config: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
