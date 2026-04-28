Lifecycle
#########

The application starts, runs for a while, and stops. Between
those three moments the lifecycle controls every resource
Vocalance owns: the services, the background tasks, the asyncio
executor, the signal handlers.

Lifecycle at a glance
=====================

One object — ``AppLifecycle``
(``vocalance/app/lifecycle/lifecycle.py``) — owns four phases.

.. mermaid::

   flowchart LR
       Specs[Service specs<br/><i>declarative list</i>] --> Build[Build phase<br/>realize specs in order]
       Build --> Init[Init phase<br/>await initialize() per service]
       Init --> Run[Run phase<br/>application is live]
       Run -->|trigger| Tear[Teardown phase<br/>LIFO shutdown]

The lifecycle exposes the operations each phase needs.

===================================  =========================================================
Operation                            Purpose
===================================  =========================================================
``register_resource``                Add a service or other resource that must be torn down.
``register_init_task``               Track the in-flight initialization task.
``spawn``                            Create and track a long-running background task.
``request_shutdown`` / ``teardown``  Drive the orderly shutdown.
===================================  =========================================================

Holding all four in one object is what allows the rest of the
application to ignore lifetime entirely. A service registers
itself once, opts into an init task once, and never thinks about
teardown again.

The service spec
================

Construction is *declarative*. ``qt_main`` does not instantiate
services in a hand-written sequence; it builds a list of
``ServiceSpec`` records and lets the lifecycle realize them in
order.

.. code-block:: python

   ServiceSpec(name="audio_capture", factory=lambda c: AudioCaptureService(...))

Each spec is a name and a factory. The factory receives a
container with previously built services so that dependencies
resolve by name (``c["event_bus"]``, ``c["config"]``,
``c["storage"]``).

The full list lives in ``qt_main._service_specs`` and is the
source of truth for the service graph. Three guarantees follow
from a single declarative list:

- Construction order is the order specs appear.
- Teardown order is the reverse (LIFO over registration).
- Adding a service is one edit: appending a spec.

Background tasks
================

Some services need long-running asyncio tasks: the click tracker
debounces persistence, the dictation coordinator monitors silence
in Type mode, the LLM service streams tokens. Each is started
with ``lifecycle.spawn``.

.. code-block:: python

   def spawn(self, coro, *, name: str = "task") -> asyncio.Task:
       task = self._loop.create_task(coro, name=name)
       task.add_done_callback(self._log_task_exception)
       self._background_tasks.append(task)
       return task

Three properties:

- The task is created on the lifecycle's loop.
- A done-callback logs any unhandled exception, so a
  fire-and-forget task cannot silently swallow errors.
- The task is recorded for cooperative cancellation during
  teardown.

Cancellation token
==================

Shutdown is *cooperative*. The lifecycle does not kill threads;
it asks running work to return. The mechanism is a
``CancellationToken``
(``vocalance/app/lifecycle/cancellation.py``) with two faces — an
``asyncio.Event`` and a ``threading.Event`` — that mirror each
other.

================  ============================================================
Caller            Side it polls
================  ============================================================
Awaiting code     ``asyncio.Event`` (``await token.wait_async()``)
Daemon workers    ``threading.Event`` (``token.is_set()`` between iterations)
================  ============================================================

Setting the token from any thread wakes both sides.

The lifecycle holds one token, exposes it as ``cancel_token``,
and sets it as the very first step of teardown.

Teardown order
==============

Teardown is a fixed six-step sequence. Each step is run to
completion before the next begins.

.. mermaid::

   flowchart TD
       T0[teardown called] --> T1[1. Set cancel token<br/><i>cooperating workers wind down</i>]
       T1 --> T2[2. Cancel and await init task]
       T2 --> T3[3. Cancel and await background tasks]
       T3 --> T4[4. Close registered resources<br/><i>LIFO over registration</i>]
       T4 --> T5[5. Drain asyncio default executor]
       T5 --> T6[6. Stop signal-poll timer]

In code:

.. code-block:: python

   async def teardown(self) -> None:
       if self._teardown_done:
           return
       self._teardown_done = True

       self.cancel_token.set()
       await self._cancel_and_await_init()
       await self._cancel_and_await_background()
       await self._close_resources()
       await self._shutdown_default_executor()
       self._stop_signal_timer()

``teardown`` is idempotent: a second call returns immediately.
Every error path (signal, init failure, user-initiated quit)
funnels into the same method.

Why LIFO matters
----------------

Reverse-registration teardown is a correctness requirement, not
aesthetics. Consider the capture path:

- ``AudioCaptureService`` is registered early because the
  segmenters and the dictation coordinator subscribe to its bus
  event during construction.
- ``CommandSegmenterService``, ``SoundSegmenterService``, and
  ``DictationCoordinator`` are registered later.

LIFO teardown means the segmenters and the coordinator shut
down *first*. The base ``Service.shutdown`` releases every bus
subscription each one holds, so by the time the capture service
itself shuts down there are no subscribers left for
``AudioChunkCapturedEvent``. The capture service then stops the
PortAudio stream and the path is fully torn down.

Reverse the order and the capture service would stop first,
leaking chunks onto the bus while subscribers are mid-teardown
— exactly the kind of race LIFO is designed to avoid.

Triggers
========

Three things can start a shutdown.

================================  ===========================================================
Trigger                           Path
================================  ===========================================================
User closes the main window       Qt's ``lastWindowClosed`` calls ``request_shutdown``.
``SIGINT`` / ``SIGTERM``          Python signal handler sets a ``threading.Event`` polled
                                  by a Qt ``QTimer`` every 100 ms.
Init failure                      ``initialize`` returns ``False`` or raises;
                                  ``qt_main`` calls ``request_shutdown`` itself.
================================  ===========================================================

The signal-via-timer indirection exists because
``signal.signal`` callbacks are not allowed to touch the asyncio
loop directly.

All three funnel through ``request_shutdown``, which sets the
cancel token and signals the asyncio shutdown event. Whatever
was awaiting ``lifecycle.wait()`` wakes up and runs ``teardown``.

Where to read next
==================

The lifecycle owns *resources*; some of those resources own
*persisted state*. The storage and configuration layer that sits
underneath the services — atomic JSON, a TTL-cached reader, a
live configuration store — is the subject of :doc:`storage`.
