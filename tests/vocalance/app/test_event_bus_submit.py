import asyncio

from vocalance.app.event_bus import EventBus
from vocalance.app.events.base_event import BaseEvent


class SampleEvent(BaseEvent):
    value: int


async def test_submit_enqueues_for_worker_without_create_task():
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.start(loop)
    seen: list[int] = []

    def on_event(event: SampleEvent) -> None:
        seen.append(event.value)

    bus.subscribe(SampleEvent, on_event)
    bus.submit(SampleEvent(value=7))
    await asyncio.sleep(0.05)
    assert seen == [7]
