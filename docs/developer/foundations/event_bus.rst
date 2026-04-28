Event bus
#########

Every feature chapter so far described publishers and subscribers
without saying anything about *how* the bus actually delivers an
event between them. This chapter fills that gap. By the end of it
you will know what it means for the bus to "deliver an event", what
guarantees the bus makes about ordering and concurrency, and how the
small subscription-tracking helper that services use under the hood
keeps the codebase free of manual unsubscribe calls.

The bus contract
================

The bus is a single object, ``EventBus``
(``vocalance/app/event_bus.py``), constructed once and shared across
the whole application. It exposes three operations:

.. code-block:: python

   class EventBus:
       def subscribe(self, event_type, handler) -> None: ...
       def unsubscribe(self, event_type, handler) -> None: ...
       async def publish(self, event) -> None: ...

A handler is any callable. Synchronous and asynchronous handlers are
both accepted; the bus inspects each one with
``asyncio.iscoroutinefunction`` and dispatches accordingly. A handler
subscribed to a parent class type also receives every subclass event,
because dispatch walks ``type(event).__mro__`` to collect handlers.

The queue
=========

A naive bus would invoke every subscriber's handler the moment
``publish`` is called. That would couple the publisher's runtime to
the speed of the slowest subscriber and would lose ordering as soon
as a subscriber awaited anything. The Vocalance bus therefore
separates publishing from delivery with a queue.

.. mermaid::

   flowchart LR
       Pub[Publishers] -->|publish| Q[Bounded asyncio.Queue]
       Q --> Worker[Single worker task]
       Worker --> H1[Handler 1]
       Worker --> H2[Handler 2]
       Worker --> H3[Handler 3]

The pieces:

- ``publish`` puts the event onto a bounded ``asyncio.Queue``.
- A single long-running worker task pulls events off the queue, one
  at a time, and dispatches each one to its handlers.
- ``publish`` returns as soon as the event is on the queue.

Two consequences:

- **Sequential dispatch.** While the worker is dispatching event A,
  it is not dispatching event B. The next event waits its turn,
  which is what gives every feature chapter the right to talk about
  state changes as if they were ordered — they really are.
- **Backpressure, not drops.** The queue has a cap (currently 500).
  If the system is so overloaded that the cap is reached, ``publish``
  blocks until the worker drains. Events are never silently
  discarded while the bus is running.

Within one event
================

Sequential delivery between events does not preclude concurrency
within a single event. When the worker dispatches an event, it walks
the subscriber list and splits them in two:

- **Synchronous handlers** are invoked immediately, one after the
  other, on the worker's thread.
- **Asynchronous handlers** are scheduled together with
  ``asyncio.gather`` and awaited as a group.

The worker only proceeds to the next event after every async handler
of the current one has finished. The result is sequential *between*
events and concurrent *within* an event, with one bus-wide guarantee
that one event finishes before the next begins.

A single failing handler is logged and skipped; it cannot prevent
the other handlers — or the next event — from running. The bus is
forgiving by design.

Subscription tracking
=====================

Services and UI controllers subscribe to a lot of events between
them. Releasing those subscriptions on shutdown by hand would be a
chore and a source of bugs. The bus ships a small helper to handle
the chore automatically: ``SubscriptionTracker``.

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
``self.subscribe(event_type, handler)``. Every service registers its
handlers through that helper at construction; ``await
super().shutdown()`` calls ``unsubscribe_all`` on the way out. The
result is that no service in the codebase contains a manual
``event_bus.unsubscribe`` call.

The same helper is used by ``QtBaseController`` for UI controllers,
so every Qt controller's teardown follows the same pattern.

Including the audio stream
==========================

There is no shortcut path that bypasses the bus. The captured audio
stream — about thirty events per second, one per microphone buffer —
flows through exactly the same queue as every other event. The
segmenters, the dictation coordinator, and the popup wave-meter are
all ordinary subscribers to ``AudioChunkCapturedEvent``. The queue's
500-event cap and sequential dispatch handle that traffic
comfortably, and treating audio like any other event removes the only
plausible reason a service might own a callback registry of its own.

Where to read next
==================

The bus delivers events sequentially on a single worker task that
runs on a single thread. Which thread that is, why every other
service runs on the same one, and how the few services that
genuinely *do* need a different thread cooperate, is the subject of
:doc:`concurrency`.
