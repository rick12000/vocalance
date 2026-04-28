from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List

import pytest

from vocalance.app.lifecycle.lifecycle import AppLifecycle
from vocalance.app.lifecycle.registry import ServiceSpec, build_services, register_services_for_teardown


class _RecordingResource:
    """AsyncCloseable that appends its tag to a shared log on shutdown."""

    def __init__(self, tag: str, sink: List[str]) -> None:
        self.tag = tag
        self.sink = sink
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.sink.append(self.tag)


class _SyncResource:
    """Resource whose ``shutdown`` is synchronous (lifecycle must accept both)."""

    def __init__(self, sink: List[str]) -> None:
        self.sink = sink

    def shutdown(self) -> None:
        self.sink.append("sync")


class _SlowResource:
    """Resource that exceeds the per-resource grace period."""

    async def shutdown(self) -> None:
        await asyncio.sleep(60)


@pytest.mark.asyncio
async def test_resources_torn_down_in_lifo_order() -> None:
    lifecycle = AppLifecycle()
    sink: List[str] = []
    for tag in ("first", "second", "third"):
        lifecycle.register_resource(_RecordingResource(tag, sink))

    await lifecycle.teardown()

    assert sink == ["third", "second", "first"]


@pytest.mark.asyncio
async def test_teardown_is_idempotent() -> None:
    lifecycle = AppLifecycle()
    sink: List[str] = []
    resource = _RecordingResource("only", sink)
    lifecycle.register_resource(resource)

    await lifecycle.teardown()
    await lifecycle.teardown()

    assert resource.shutdown_calls == 1
    assert sink == ["only"]


@pytest.mark.asyncio
async def test_teardown_handles_sync_and_slow_resources() -> None:
    lifecycle = AppLifecycle()
    sink: List[str] = []
    lifecycle.register_resource(_SlowResource())
    lifecycle.register_resource(_SyncResource(sink))
    lifecycle.register_resource(_RecordingResource("recorded", sink))

    start = time.monotonic()
    await lifecycle.teardown()
    elapsed = time.monotonic() - start

    assert sink == ["recorded", "sync"]
    assert elapsed < 10.0


@pytest.mark.asyncio
async def test_run_blocking_threads_do_not_outlive_teardown() -> None:
    lifecycle = AppLifecycle()
    started = threading.Event()
    cancelled = threading.Event()

    def cooperative_worker() -> str:
        started.set()
        token = lifecycle.cancel_token.threading_event()
        while not token.is_set():
            time.sleep(0.01)
        cancelled.set()
        return "done"

    assert await asyncio.get_running_loop().run_in_executor(None, started.wait, 2.0)

    pre_thread_names = {t.name for t in threading.enumerate() if t.is_alive() and t.name == "coop"}
    assert pre_thread_names == {"coop"}

    await lifecycle.teardown()

    assert lifecycle.cancel_token.is_set()
    assert cancelled.wait(2.0), "Worker did not observe cancel token"

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not any(t.name == "coop" for t in threading.enumerate() if t.is_alive()):
            break
        await asyncio.sleep(0.05)
    leaked = [t.name for t in threading.enumerate() if t.is_alive() and t.name == "coop"]
    assert not leaked, f"Daemon worker threads leaked past teardown: {leaked}"


@pytest.mark.asyncio
async def test_teardown_cancels_init_task() -> None:
    lifecycle = AppLifecycle()
    init_started = asyncio.Event()

    async def init_routine() -> None:
        init_started.set()
        await asyncio.sleep(60)

    task = asyncio.create_task(init_routine())
    lifecycle.register_init_task(task)

    await init_started.wait()
    await lifecycle.teardown()

    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_request_shutdown_unblocks_wait() -> None:
    lifecycle = AppLifecycle()

    waiter = asyncio.create_task(lifecycle.wait())
    await asyncio.sleep(0)

    lifecycle.request_shutdown(reason="test", source="unit")
    await asyncio.wait_for(waiter, timeout=1.0)

    assert lifecycle.is_shutdown_requested()
    assert lifecycle.cancel_token.is_set()

    await lifecycle.teardown()


@pytest.mark.asyncio
async def test_service_spec_registry_drives_lifo_teardown() -> None:
    """Registry order is construction order; lifecycle tears down in reverse."""
    lifecycle = AppLifecycle()
    sink: List[str] = []

    def factory(tag: str):
        def _build(_ctx: Dict[str, Any]) -> _RecordingResource:
            return _RecordingResource(tag, sink)

        return _build

    specs = [ServiceSpec(name=tag, factory=factory(tag)) for tag in ("alpha", "beta", "gamma")]
    ctx: Dict[str, Any] = {}
    build_services(specs, ctx)
    register_services_for_teardown(specs, ctx, lifecycle)

    assert {tag: type(ctx[tag]).__name__ for tag in ("alpha", "beta", "gamma")} == {
        "alpha": "_RecordingResource",
        "beta": "_RecordingResource",
        "gamma": "_RecordingResource",
    }

    await lifecycle.teardown()

    assert sink == ["gamma", "beta", "alpha"]
