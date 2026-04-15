import asyncio
import concurrent.futures
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


def _log_future_exception(fut: concurrent.futures.Future[Any]) -> None:
    try:
        exc = fut.exception()
        if exc is not None:
            logger.error("GUI-loop scheduled coroutine failed", exc_info=(type(exc), exc, exc.__traceback__))
    except Exception:
        logger.exception("GUI-loop future inspection failed")


class GuiAsyncBridge:
    """Marshals work onto the single GUI asyncio loop from any OS thread.

    Used by services and Qt controllers the same way ``AudioService`` receives
    ``main_event_loop``: construction-time wiring in ``qt_main._construct_services``,
    never optional post-hoc injection from the UI layer.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def schedule_coro(self, coro: Awaitable[Any]) -> concurrent.futures.Future[Any]:
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        fut.add_done_callback(_log_future_exception)
        return fut

    def invoke_on_gui_loop(self, callback: Callable[[], None]) -> None:
        self._loop.call_soon_threadsafe(callback)
