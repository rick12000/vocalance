from __future__ import annotations

import asyncio
import threading
from typing import Callable, Set


class CancellationToken:
    """Single source of truth for shutdown across sync and async worlds.

    A token wraps a ``threading.Event`` (polled by sync worker threads) and an
    ``asyncio.Event`` (awaited by coroutines) that are flipped together. Setting
    the token from any thread is idempotent and safe.

    Per-operation cancellation events (e.g. a user pressing "Cancel" on a
    download) can be linked via ``link_event``: app shutdown automatically
    propagates into those events so no in-flight blocking work outlives the
    lifecycle, while the operation can still complete or be cancelled
    independently.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._sync_event = threading.Event()
        self._async_event = asyncio.Event()
        self._linked: Set[threading.Event] = set()
        self._lock = threading.Lock()

    def set(self) -> None:
        """Mark the token as cancelled; thread-safe and idempotent."""
        if self._sync_event.is_set():
            return
        self._sync_event.set()
        with self._lock:
            linked = list(self._linked)
            self._linked.clear()
        for child in linked:
            child.set()
        try:
            self._loop.call_soon_threadsafe(self._async_event.set)
        except RuntimeError:
            pass

    def is_set(self) -> bool:
        """Return True once cancellation has been requested."""
        return self._sync_event.is_set()

    def threading_event(self) -> threading.Event:
        """Expose the underlying ``threading.Event`` for sync workers to poll."""
        return self._sync_event

    def link_event(self, event: threading.Event) -> Callable[[], None]:
        """Mirror this token into ``event``; return an unlink callable.

        If the token is already set, ``event`` is set immediately. Otherwise
        the event is added to the link set and will be set when this token is.
        Callers should invoke the returned unlink callable once the event is
        no longer needed (typically when the operation finishes) to avoid
        retaining stale references.
        """
        if self._sync_event.is_set():
            event.set()
            return lambda: None

        with self._lock:
            self._linked.add(event)

        def unlink() -> None:
            with self._lock:
                self._linked.discard(event)

        return unlink

    async def wait(self) -> None:
        """Await cancellation on the asyncio side."""
        await self._async_event.wait()
