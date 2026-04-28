Event Bus
#########

.. sectnum::

The event bus is the only communication channel between services. It is the
mechanism that lets every service in :doc:`../features/capture`,
:doc:`../features/command_flow`, :doc:`../features/dictation_flow`, and the UI
layer interact without holding references to each other.

This chapter explains how the bus delivers events: the queue that separates
publishing from dispatching, the ordering guarantees that follow, and the
subscription tracking that makes teardown automatic.

The Publish / Subscribe Model
==============================

The bus is one object, ``EventBus`` (``vocalance/app/event_bus.py``), shared
across the application:

.. code-block:: python

   class EventBus:
       def subscribe(self, event_type, handler) -> None: ...
       def unsubscribe(self, event_type, handler) -> None: ...
       async def publish(self, event) -> None: ...

A **handler** is any callable — synchronous or asynchronous. The bus inspects
each handler with ``asyncio.iscoroutinefunction`` at subscription time and
stores the kind alongside the callable so dispatch does not have to re-inspect.

Event types form a hierarchy via normal Python inheritance. A handler subscribed
to ``BaseEvent`` receives every event in the system; a handler subscribed to
``CommandParsedEvent`` receives every event that is a subclass of it. The bus
implements this by walking ``type(event).__mro__`` at dispatch time and
collecting all handlers registered for any type in the chain.

The Queue and the Worker
=========================

Naive Publish Semantics and Their Problems
-------------------------------------------

A straightforward bus would call every subscriber synchronously inside
``publish``. Two problems arise:

- **Coupling.** If subscriber B is slow — it awaits a disk read, for example —
  publisher A is stuck waiting for B to finish before it can continue. The
  publisher's execution is coupled to the slowest subscriber.
- **Ordering.** If a handler awaits anything, another event can be published
  and start dispatching before the first event's handlers have all finished.
  Two events published in order A then B can interleave in ways that make final
  state depend on scheduling, not publication order.

Queue-Backed Delivery
----------------------

The Vocalance bus solves both problems by inserting a queue between publishing
and dispatching.

.. mermaid::

   flowchart LR
       Pub[Publishers] -->|enqueue| Q[Bounded asyncio.Queue<br/>cap = 500]
       Q --> Worker[Worker task<br/>pulls one at a time]
       Worker --> Dispatch[Dispatch to all<br/>subscribers of this type]
       Dispatch --> Next[Pull next event]

``publish`` places the event on the queue and returns immediately. The caller
is not blocked by anything a subscriber does. A single **worker task** runs in
an asyncio loop, pulling one event at a time and dispatching it to all matching
subscribers. Only after every subscriber has finished handling that event does
the worker pull the next one.

This gives a hard guarantee: **event A is fully dispatched before event B
begins**. Causal ordering is preserved regardless of what subscribers do
internally.

The queue's capacity is capped at 500 events. If publishing ever fills the
queue — which would indicate a serious backlog — ``publish`` blocks until the
worker drains a slot rather than silently discarding events. The 500-event cap
is large enough to absorb the audio stream (~30 events/second) indefinitely
under normal load.

Within-Event Concurrency
--------------------------

Sequential delivery *between* events does not prevent concurrency *within* an
event. When the worker dispatches an event, it splits the subscriber list by
handler type and treats the two groups differently.

Synchronous handlers run one after another in registration order. Each must
complete before the next begins.

Asynchronous handlers are gathered with ``asyncio.gather`` and awaited as a
group. All async handlers for a given event can interleave at their respective
``await`` points — they are concurrent with each other. However, the worker does
not pull the next event until ``asyncio.gather`` returns, which is after the
slowest async handler finishes. The event is still fully dispatched before the
next begins.

.. mermaid::

   flowchart LR
       Worker[Worker pulls event] --> Split{Handler kind?}
       Split -->|sync| Sync[Run sequentially]
       Split -->|async| Async[asyncio.gather all]
       Sync --> Done[All done]
       Async --> Done
       Done --> Next[Pull next event]

A handler that raises an exception is logged and skipped. It cannot prevent
remaining handlers from running, and it cannot prevent the worker from pulling
the next event.

Audio on the Bus
-----------------

The audio stream — roughly thirty ``AudioChunkCapturedEvent`` messages per
second — travels through the same queue as every other event. There is no
priority lane, no shortcut, no separate audio channel. The 500-event buffer is
large enough that audio events do not back up even when recognition takes longer
than a single chunk interval.

Subscription Tracking
======================

Services and UI controllers together hold many active subscriptions. Releasing
each one manually at shutdown is error-prone; forgotten subscriptions cause
handlers to fire on a half-torn-down service. The bus provides a helper that
tracks and bulk-releases subscriptions automatically.

The ``SubscriptionTracker``
----------------------------

.. code-block:: python

   class SubscriptionTracker:
       def subscribe(self, event_type, handler) -> None:
           self.event_bus.subscribe(event_type, handler)
           self._subscriptions.append((event_type, handler))

       def unsubscribe_all(self) -> None:
           for event_type, handler in self._subscriptions:
               self.event_bus.unsubscribe(event_type, handler)
           self._subscriptions.clear()

Every call to ``subscribe`` is recorded in a list. ``unsubscribe_all`` releases
all of them in one call. The base ``Service`` class owns a
``SubscriptionTracker`` and exposes ``self.subscribe(event_type, handler)``
as a shortcut. ``await super().shutdown()`` calls ``unsubscribe_all`` on the
way out. There is not a single manual ``event_bus.unsubscribe`` call anywhere
in the codebase. The same tracker is embedded in ``QtBaseController`` for UI
controllers.

Understanding the bus's delivery guarantees is the foundation for understanding
the concurrency model. The next chapter — :doc:`concurrency` — explains how the
event loop, threads, and blocking work interact beneath the surface of every
``publish`` and ``await``.
