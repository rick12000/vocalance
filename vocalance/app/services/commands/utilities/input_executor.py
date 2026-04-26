from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from vocalance.app.event_bus import EventBus
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)

T = TypeVar("T")


class KeyboardInputService(Service):
    """Serialises OS-level input injection (pyautogui) on a single dedicated worker thread.

    A single-worker pool guarantees that mouse clicks and keystrokes execute strictly
    in the order requested, preventing OS-level race conditions, while keeping the
    async/Qt event loop unblocked. The pool is created on construction and torn down
    deterministically by ``AppLifecycle`` via ``shutdown``.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vocalance-input")

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute ``fn`` on the input worker and await its result.

        Args:
            fn: Synchronous callable performing input injection.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            Whatever ``fn`` returns.
        """
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))
        return await loop.run_in_executor(self._executor, fn, *args)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Fire-and-forget submission for callers that cannot await (e.g. Qt widgets)."""
        if kwargs:
            self._executor.submit(lambda: fn(*args, **kwargs))
        else:
            self._executor.submit(fn, *args)

    async def shutdown(self) -> None:
        """Drain the input worker so no pyautogui call outlives the lifecycle."""
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        except Exception as exc:
            logger.warning("KeyboardInputService executor shutdown raised: %s", exc)
        await super().shutdown()
