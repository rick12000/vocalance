Event bus
#########

The feature chapters described publishers and subscribers without
saying *how* the bus delivers an event between them. This chapter
fills that gap: what it means to "deliver an event", what
guarantees the bus makes about ordering and concurrency, and how
the small subscription-tracking helper keeps the codebase free of
manual unsubscribe calls.

Surface
=======

The bus is one object, ``EventBus``
(``vocalance/app/event_bus.py``), constructed once and shared
across the application:

.. code-block:: python

   class EventBus:
       def subscribe(self, event_type, handler) -> None: ...
       def unsubscribe(self, event_type, handler) -> None: ...
       async def publish(self, event) -> None: ...

A handler is any callable. Synchronous and asynchronous handlers
are both accepted; the bus inspects each with
``asyncio.iscoroutinefunction`` and dispatches accordingly. A
handler subscribed to a parent event type also receives every
subclass event, because dispatch walks ``type(event).__mro__`` to
collect handlers.

Queue and worker
================

A naive bus would invoke every subscriber the moment ``publish``
is called. That couples the publisher's runtime to the slowest
subscriber and loses ordering as soon as a subscriber awaits
anything. The Vocalance bus separates publishing from delivery
with a queue.

.. mermaid::

   flowchart LR
       Pub[Publishers] -->|publish: enqueue| Q[(Bounded<br/>asyncio.Queue<br/><i>cap = 500</i>)]
       Q -->|one event<br/>at a time| Worker[Single worker task]
       Worker -->|dispatch| Subs[All subscribers<br/>of this event type]

Three properties follow:

- ``publish`` returns as soon as the event is on the queue.
- A single worker task pulls events one at a time and dispatches
  each to its handlers.
- Sequential delivery between events is therefore guaranteed:
  while the worker is dispatching event A, it is not dispatching
  event B.

If the queue cap is reached, ``publish`` blocks until the worker
drains. Events are never silently discarded while the bus is
running — the bus uses backpressure, not drops.

Within a single event
---------------------

Sequential between events does not preclude concurrency *within*
an event. The worker splits subscribers into two groups and
treats them differently.

.. mermaid::

   flowchart LR
       Worker[Worker pulls<br/>event from queue] --> Split{Handler kind?}
       Split -->|sync| Sync[Run inline,<br/>one after another]
       Split -->|async| Async[asyncio.gather all,<br/>await as a group]
       Sync --> Done[Wait until all<br/>handlers finished]
       Async --> Done
       Done --> Next[Pull next event]

The worker only proceeds to the next event after every handler
of the current one has finished. The bus is sequential
*between* events, concurrent *within* one. A failing handler is
logged and skipped; it cannot prevent the other handlers — or
the next event — from running.

Subscription tracking
=====================

Services and UI controllers subscribe to many events between
them. Releasing those subscriptions on shutdown by hand would be
a chore and a source of bugs. The bus ships a helper:
``SubscriptionTracker``.

.. code-block:: python

   class SubscriptionTracker:
       def subscribe(self, event_type, handler) -> None:
           self.event_bus.subscribe(event_type, handler)
           self._subscriptions.append((event_type, handler))

       def unsubscribe_all(self) -> None:
           for event_type, handler in self._subscriptions:
               self.event_bus.unsubscribe(event_type, handler)
           self._subscriptions.clear()

The base ``Service`` class owns one of these and exposes
``self.subscribe(event_type, handler)``. Every service registers
its handlers through that helper at construction;
``await super().shutdown()`` calls ``unsubscribe_all`` on the way
out. The result is that no service in the codebase contains a
manual ``event_bus.unsubscribe`` call.

The same helper is used by ``QtBaseController`` for UI
controllers, so every Qt controller's teardown follows the same
pattern.

The audio stream is on the bus too
==================================

There is no shortcut path that bypasses the bus. The captured
audio stream — about thirty events per second, one per microphone
buffer — flows through the same queue as every other event. The
segmenters, the dictation coordinator, and the popup wave meter
are all ordinary subscribers to ``AudioChunkCapturedEvent``. The
queue's 500-event cap and sequential dispatch handle that traffic
comfortably, and treating audio like any other event removes the
only plausible reason a service might own a callback registry of
its own.

Where to read next
==================

The bus delivers events sequentially on a single worker task
that runs on a single thread. *Which* thread that is, why every
other service runs on the same one, and how the few services
that genuinely need a different thread cooperate, is the subject
of :doc:`concurrency`.
