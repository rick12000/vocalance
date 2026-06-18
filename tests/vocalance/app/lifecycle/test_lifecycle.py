from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict

import pytest

from vocalance.app.lifecycle.registry import ServiceSpec, build_services, register_services_for_teardown
from vocalance.app.lifecycle.worker import run_blocking


async def test_resources_torn_down_in_lifo_order(app_lifecycle, recording_resource_factory, teardown_sink):
    for tag in ("first", "second", "third"):
        app_lifecycle.register_resource(recording_resource_factory(tag))

    await app_lifecycle.teardown()

    assert teardown_sink == ["third", "second", "first"]


async def test_teardown_is_idempotent(app_lifecycle, recording_resource_factory, teardown_sink):
    resource = recording_resource_factory("only")
    app_lifecycle.register_resource(resource)

    await app_lifecycle.teardown()
    await app_lifecycle.teardown()

    assert resource.shutdown_calls == 1
    assert teardown_sink == ["only"]


async def test_teardown_isolates_slow_and_sync_resources(app_lifecycle, recording_resource_factory, teardown_sink):
    app_lifecycle.register_resource(recording_resource_factory("slow", mode="slow"))
    app_lifecycle.register_resource(recording_resource_factory("sync", mode="sync"))
    app_lifecycle.register_resource(recording_resource_factory("recorded"))

    start = time.monotonic()
    await app_lifecycle.teardown()
    elapsed = time.monotonic() - start

    assert teardown_sink == ["recorded", "sync"]
    assert elapsed < 10.0


async def test_teardown_cancels_init_task(app_lifecycle):
    init_started = asyncio.Event()

    async def init_routine() -> None:
        init_started.set()
        await asyncio.sleep(60)

    task = asyncio.create_task(init_routine())
    app_lifecycle.register_init_task(task)

    await init_started.wait()
    await app_lifecycle.teardown()

    assert task.cancelled()


async def test_spawn_tracked_task_is_cancelled_on_teardown(app_lifecycle):
    running = asyncio.Event()

    async def background() -> None:
        running.set()
        await asyncio.sleep(60)

    task = app_lifecycle.spawn(background(), name="bg")
    await running.wait()

    await app_lifecycle.teardown()

    assert task.cancelled()


async def test_request_shutdown_is_idempotent_and_unblocks_wait(app_lifecycle):
    waiter = asyncio.create_task(app_lifecycle.wait())
    await asyncio.sleep(0)

    assert app_lifecycle.request_shutdown(reason="test", source="unit") is True
    assert app_lifecycle.request_shutdown(reason="again", source="unit") is False

    await asyncio.wait_for(waiter, timeout=1.0)

    assert app_lifecycle.is_shutdown_requested()
    assert app_lifecycle.cancel_token.is_set()


async def test_cancellation_token_set_flips_sync_and_async_events(cancellation_token):
    assert not cancellation_token.is_set()

    cancellation_token.set()

    assert cancellation_token.is_set()
    assert cancellation_token.threading_event().is_set()
    await asyncio.wait_for(cancellation_token.wait(), timeout=1.0)


async def test_link_event_mirrors_later_cancellation(cancellation_token):
    child = threading.Event()
    cancellation_token.link_event(child)
    assert not child.is_set()

    cancellation_token.set()

    assert child.is_set()


async def test_link_event_sets_immediately_when_already_cancelled(cancellation_token):
    cancellation_token.set()

    child = threading.Event()
    cancellation_token.link_event(child)

    assert child.is_set()


async def test_unlink_stops_mirroring(cancellation_token):
    child = threading.Event()
    unlink = cancellation_token.link_event(child)
    unlink()

    cancellation_token.set()

    assert not child.is_set()


async def test_run_blocking_returns_callable_result():
    result = await run_blocking(lambda a, b: a + b, 2, 3)

    assert result == 5


async def test_run_blocking_propagates_exception():
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await run_blocking(boom)


async def test_run_blocking_sets_cancel_token_on_cancellation(cancellation_token):
    started = threading.Event()

    def blocker() -> None:
        started.set()
        cancellation_token.threading_event().wait(2.0)

    task = asyncio.create_task(run_blocking(blocker, cancel_token=cancellation_token))
    assert await asyncio.get_running_loop().run_in_executor(None, started.wait, 2.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancellation_token.is_set()


def test_build_services_builds_in_order_and_passes_partial_context():
    specs = [
        ServiceSpec(name="a", factory=lambda ctx: f"a:{sorted(ctx)}"),
        ServiceSpec(name="b", factory=lambda ctx: f"b:{sorted(ctx)}"),
    ]
    ctx: Dict[str, Any] = {"base": 1}

    result = build_services(specs, ctx)

    assert result is ctx
    assert result["a"] == "a:['base']"
    assert result["b"] == "b:['a', 'base']"


async def test_registry_drives_lifo_teardown(app_lifecycle, recording_resource_factory, teardown_sink):
    specs = [ServiceSpec(name=tag, factory=lambda ctx, t=tag: recording_resource_factory(t)) for tag in ("alpha", "beta", "gamma")]
    ctx: Dict[str, Any] = {}
    build_services(specs, ctx)
    register_services_for_teardown(specs, ctx, app_lifecycle)

    assert [ctx[tag].tag for tag in ("alpha", "beta", "gamma")] == ["alpha", "beta", "gamma"]

    await app_lifecycle.teardown()

    assert teardown_sink == ["gamma", "beta", "alpha"]
