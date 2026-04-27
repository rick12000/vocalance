from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

from vocalance.app.event_bus import EventBus
from vocalance.app.lifecycle import run_blocking
from vocalance.app.services.base_service import Service

T = TypeVar("T")


class KeyboardInputService(Service):
    """Serialises OS-level input injection (pyautogui) using an asyncio lock + ``run_blocking``.

    Each call hops to a worker thread (so pyautogui never blocks the loop) but the
    asyncio lock guarantees strict FIFO ordering, preventing OS-level races between
    mouse clicks and keystrokes.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus)
        self._serial = asyncio.Lock()

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute ``fn`` off the loop, serialised against other ``run`` calls."""
        async with self._serial:
            return await run_blocking(fn, *args, name="vocalance-input", **kwargs)
