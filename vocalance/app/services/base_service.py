from __future__ import annotations

from abc import ABC, abstractmethod


class Service(ABC):
    """Contract every application service must satisfy.

    Subclasses subscribe to the event bus in ``__init__``, optionally implement
    ``initialize`` for async setup, and implement ``shutdown`` for teardown.
    """

    async def initialize(self) -> bool:
        """Optional async startup (e.g. storage reads, heavy imports).

        Returns:
            True when startup succeeded; services may use False to signal failure.
        """
        return True

    @abstractmethod
    async def shutdown(self) -> None:
        """Tear down: unsubscribe handlers and release resources. Must be idempotent."""
