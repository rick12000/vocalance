"""Abstract base class for all application services.

Lifecycle contract
------------------
1. ``__init__``  – Synchronous.  Wire ALL event-bus subscriptions here.  The bus
                   queues events until ``event_bus.start(loop)`` is called, so subscribing
                   in ``__init__`` is always safe regardless of init order.
2. ``initialize`` – Optional async startup (storage reads, heavy imports, etc.).
                    Runs after construction but before the bus is started.
3. ``shutdown``  – Async teardown.  Unsubscribe handlers and release resources.
                   Must be idempotent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Service(ABC):
    """Contract every application service must satisfy."""

    async def initialize(self) -> bool:
        """Optional async initialisation.  Returns True on success."""
        return True

    @abstractmethod
    async def shutdown(self) -> None:
        """Tear down; unsubscribe handlers and release resources."""
