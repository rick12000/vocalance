from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Optional, TypeVar

from vocalance.app.lifecycle.cancellation import CancellationToken

T = TypeVar("T")


async def run_blocking(
    fn: Callable[..., T],
    *args: Any,
    cancel_token: Optional[CancellationToken] = None,
    name: str = "vocalance-blocking",
    **kwargs: Any,
) -> T:
    """Run a synchronous callable on a daemon thread and await its result.

    The thread is daemon so any non-cooperating call cannot keep the interpreter
    alive past process exit. If the awaiting coroutine is cancelled,
    ``cancel_token`` (when supplied) is set so cooperating worker code can
    return at its next checkpoint.

    Args:
        fn: Synchronous callable to execute.
        *args: Positional arguments forwarded to ``fn``.
        cancel_token: Optional token set on awaiter cancellation so ``fn`` can
            observe cooperative shutdown.
        name: Thread name used for diagnostics.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        BaseException: Whatever ``fn`` raises is re-raised on the awaiter.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def _set_result(value: T) -> None:
        if not future.done():
            future.set_result(value)

    def _set_exception(exc: BaseException) -> None:
        if not future.done():
            future.set_exception(exc)

    def worker() -> None:
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            loop.call_soon_threadsafe(_set_exception, exc)
        else:
            loop.call_soon_threadsafe(_set_result, result)

    threading.Thread(target=worker, daemon=True, name=name).start()

    try:
        return await future
    except asyncio.CancelledError:
        if cancel_token is not None:
            cancel_token.set()
        raise
