from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import threading
from types import FrameType
from typing import Any, Callable, Coroutine, List, Optional, Protocol, Set, TypeVar, runtime_checkable

from PySide6.QtCore import QTimer

from vocalance.app.lifecycle.cancellation import CancellationToken
from vocalance.app.lifecycle.worker import run_blocking as _run_blocking

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class AsyncCloseable(Protocol):
    """Resource that the lifecycle can tear down via ``shutdown`` (sync or async)."""

    async def shutdown(self) -> None:
        ...


_BACKGROUND_GRACE_S = 3.0
_INIT_GRACE_S = 2.0
_RESOURCE_GRACE_S = 5.0
_DEFAULT_EXECUTOR_GRACE_S = 5.0
_SIGNAL_POLL_MS = 100


class AppLifecycle:
    """Owns process-wide lifecycle state and coordinates shutdown.

    Holds the cancellation token, asyncio shutdown event, initialization task,
    tracked background tasks, and the LIFO stack of registered resources.
    Exposes one ``teardown`` method that is idempotent and safe to call from any
    error path.
    """

    def __init__(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as e:
            raise RuntimeError("AppLifecycle must be constructed from a coroutine running on the target loop.") from e

        self.cancel_token = CancellationToken(self._loop)

        self._coord_lock = threading.Lock()
        self._requested = False
        self._teardown_done = False

        self._shutdown_event = asyncio.Event()
        self._init_task: Optional[asyncio.Task[Any]] = None
        self._background_tasks: Set[asyncio.Task[Any]] = set()
        self._resources: List[AsyncCloseable] = []
        self._signal_timer: Optional[QTimer] = None
        self._signal_event: Optional[threading.Event] = None

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers that request shutdown via a Qt poll timer."""
        self._signal_event = threading.Event()

        def on_os_signal(signum: int, _frame: Optional[FrameType]) -> None:
            logger.info("Received signal %s; initiating graceful shutdown", signum)
            assert self._signal_event is not None
            self._signal_event.set()

        signal.signal(signal.SIGINT, on_os_signal)
        signal.signal(signal.SIGTERM, on_os_signal)

        timer = QTimer()
        timer.timeout.connect(self._poll_signal_event)
        timer.start(_SIGNAL_POLL_MS)
        self._signal_timer = timer

    def _poll_signal_event(self) -> None:
        if self._signal_event is not None and self._signal_event.is_set():
            self.request_shutdown(reason="System signal received", source="signal_handler")

    def request_shutdown(self, *, reason: str, source: str) -> bool:
        """Request a clean shutdown. Thread-safe and idempotent.

        Args:
            reason: Human-readable reason for diagnostic logs.
            source: Component that originated the request.

        Returns:
            True on the first request, False on subsequent calls.
        """
        with self._coord_lock:
            if self._requested:
                return False
            self._requested = True

        logger.info("Shutdown requested by %s: %s", source, reason)

        self.cancel_token.set()

        if self._init_task is not None and not self._init_task.done():
            self._init_task.cancel()

        try:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
        except RuntimeError:
            pass
        return True

    def is_shutdown_requested(self) -> bool:
        """Return whether shutdown has been requested."""
        with self._coord_lock:
            return self._requested

    async def wait(self) -> None:
        """Block until shutdown has been requested."""
        await self._shutdown_event.wait()

    def register_init_task(self, task: asyncio.Task[Any]) -> None:
        """Track the initialization task so it is cancelled on shutdown."""
        self._init_task = task

    def clear_init_task(self) -> None:
        """Drop the init task reference once initialization has completed."""
        self._init_task = None

    def spawn(self, coro: Coroutine[Any, Any, Any], *, name: str = "task") -> asyncio.Task[Any]:
        """Create, track, and observe a background task.

        Combines ``loop.create_task`` with lifecycle tracking and a done-callback
        that logs any unhandled exception so a fire-and-forget task can never
        silently swallow errors. The done-callback also drops the task from the
        tracking set, so callers that rotate spawned tasks (e.g. click-tracker
        debounce) don't accumulate references to completed tasks.

        Args:
            coro: Coroutine to schedule on this lifecycle's loop.
            name: Diagnostic task name.

        Returns:
            The created task.
        """
        task = self._loop.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Background task %r failed: %s", task.get_name(), exc, exc_info=exc)

    def register_resource(self, resource: AsyncCloseable) -> None:
        """Register a resource to be closed (LIFO) during teardown."""
        self._resources.append(resource)

    async def run_blocking(
        self,
        fn: Callable[..., T],
        *args: Any,
        name: str = "vocalance-blocking",
        **kwargs: Any,
    ) -> T:
        """Run ``fn`` on a daemon worker thread, observing this lifecycle's cancel token."""
        return await _run_blocking(fn, *args, cancel_token=self.cancel_token, name=name, **kwargs)

    async def teardown(self) -> None:
        """Tear down everything. Idempotent and safe from any error path.

        Order:
            1. Set cancel token (cooperative blocking work returns).
            2. Cancel and await the initialization task.
            3. Cancel and await tracked background tasks.
            4. Close registered resources in reverse registration order.
            5. Drain the asyncio default executor (covers ``asyncio.to_thread`` /
               ``loop.run_in_executor(None, ...)`` worker threads).
            6. Stop the signal poll timer.
        """
        if self._teardown_done:
            return
        self._teardown_done = True

        logger.info("Lifecycle teardown starting")
        self.cancel_token.set()

        await self._cancel_and_await_init()
        await self._cancel_and_await_background()
        await self._close_resources()
        await self._shutdown_default_executor()
        self._stop_signal_timer()

        logger.info("Lifecycle teardown complete")

    async def _cancel_and_await_init(self) -> None:
        task = self._init_task
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, timeout=_INIT_GRACE_S)

    async def _cancel_and_await_background(self) -> None:
        if not self._background_tasks:
            return
        tasks = list(self._background_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=_BACKGROUND_GRACE_S,
            )
        except asyncio.TimeoutError:
            logger.warning("Background tasks did not settle within %.1fs after cancellation", _BACKGROUND_GRACE_S)
        finally:
            self._background_tasks.clear()

    async def _close_resources(self) -> None:
        for resource in reversed(self._resources):
            name = type(resource).__name__
            try:
                result = resource.shutdown()
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, timeout=_RESOURCE_GRACE_S)
            except asyncio.TimeoutError:
                logger.warning("%s.shutdown() exceeded %.1fs", name, _RESOURCE_GRACE_S)
            except Exception as exc:
                logger.error("%s.shutdown() failed: %s", name, exc, exc_info=True)
        self._resources.clear()

    async def _shutdown_default_executor(self) -> None:
        """Drain the asyncio default executor so its non-daemon workers do not outlive us."""
        try:
            await asyncio.wait_for(self._loop.shutdown_default_executor(), timeout=_DEFAULT_EXECUTOR_GRACE_S)
        except asyncio.TimeoutError:
            logger.warning("Default executor did not drain within %.1fs", _DEFAULT_EXECUTOR_GRACE_S)
        except Exception as exc:
            logger.warning("Failed to shut down default executor: %s", exc)

    def _stop_signal_timer(self) -> None:
        if self._signal_timer is None:
            return
        with contextlib.suppress(Exception):
            self._signal_timer.stop()
        self._signal_timer = None
