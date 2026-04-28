Lifecycle
#########

The application starts, runs for a while, and stops. Between those
three moments the lifecycle controls every resource Vocalance owns:
the services, the background tasks, the asyncio executor, the
signal handlers. This chapter describes the pieces that orchestrate
that work.

The contract
============

The lifecycle is a single object, ``AppLifecycle``
(``vocalance/app/lifecycle/lifecycle.py``), constructed once at
startup. It exposes four kinds of operations:

===================================  =========================================================
Operation                            Purpose
===================================  =========================================================
``register_resource``                Add a service or other resource that must be torn down.
``register_init_task``               Track the in-flight initialization task.
``spawn``                            Create and track a long-running background task.
``request_shutdown`` / ``teardown``  Drive the orderly shutdown.
===================================  =========================================================

Holding all four in one object is what allows the rest of the
application to ignore lifetime entirely. A service registers itself
once, opts into an init task once, and never thinks about teardown
again.

The service spec
================

Construction is *declarative*. ``qt_main`` does not instantiate
services in a hand-written sequence; it builds a list of
``ServiceSpec`` records and lets the lifecycle realize them in
order:

.. code-block:: python

   ServiceSpec(name="audio_capture", factory=lambda c: AudioCaptureService(...))

Each spec is a name and a factory function. The factory receives a
container with previously built services so that dependencies can be
resolved by name (``c["event_bus"]``, ``c["config"]``,
``c["storage"]``).

The full list of specs lives in ``qt_main._service_specs`` and is
the source of truth for the service graph. Three guarantees follow
from a single declarative list:

- Construction order is the order specs appear.
- Teardown order is the reverse (LIFO over registration).
- Adding a service is one edit: appending a spec.

Background tasks
================

Some services need long-running asyncio tasks: the click tracker
debounces persistence, the dictation coordinator monitors silence
in Type mode, the LLM service streams tokens from the model. Each
of those tasks is started with ``lifecycle.spawn``:

.. code-block:: python

   def spawn(self, coro, *, name: str = "task") -> asyncio.Task:
       task = self._loop.create_task(coro, name=name)
       task.add_done_callback(self._log_task_exception)
       self._background_tasks.append(task)
       return task

Three properties:

- The task is created on the lifecycle's loop.
- A done-callback logs any unhandled exception, so a fire-and-forget
  task cannot silently swallow errors.
- The task is recorded for cooperative cancellation during teardown.

Cancellation
============

Shutdown is *cooperative*. The lifecycle does not kill threads; it
asks running work to return.

The mechanism is the ``CancellationToken``
(``vocalance/app/lifecycle/cancellation.py``). The token has two
faces — an ``asyncio.Event`` and a ``threading.Event`` — that mirror
each other. Awaiting code polls the asyncio side; daemon-thread
workers (running STT, LLM, ``pyautogui`` calls) poll the threading
side. Setting the token from any thread wakes both sides.

The lifecycle holds one token, exposes it as ``cancel_token``, and
sets it as the very first step of teardown:

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

The order is fixed:

1. **Set the cancel token.** Every cooperative worker starts winding
   down on its next checkpoint.
2. **Cancel the init task.** If startup is still in progress, it
   stops being in progress.
3. **Cancel and await background tasks.** Any task started with
   ``spawn`` is cancelled; the lifecycle gathers them with a timeout.
4. **Close resources LIFO.** Each registered service's ``shutdown``
   is awaited, in reverse registration order.
5. **Drain the asyncio default executor.** Workers from
   ``asyncio.to_thread`` and ``loop.run_in_executor(None, ...)``
   finish before the loop closes.
6. **Stop the signal-poll timer.** The Qt timer that watches for
   ``SIGINT`` / ``SIGTERM`` is released.

``teardown`` is idempotent: a second call returns immediately. Every
error path (signal, init failure, user-initiated quit) funnels into
the same method.

Why LIFO matters
----------------

Reverse-registration teardown is not aesthetic; it is a correctness
requirement. Consider the capture path:

- ``AudioCaptureService`` is registered early, because the
  segmenters and the dictation coordinator subscribe to its bus
  event during construction.
- ``CommandSegmenterService``, ``SoundSegmenterService``, and
  ``DictationCoordinator`` are registered later.

LIFO teardown means the segmenters and the coordinator shut down
*first*. The base ``Service.shutdown`` releases every bus
subscription each one holds, so by the time the capture service
itself shuts down there are no subscribers left for
``AudioChunkCapturedEvent``. The capture service then stops the
PortAudio stream and the path is fully torn down.

Reverse the order and the capture service would stop first, leaking
chunks onto the bus while subscribers are mid-teardown — exactly the
kind of race LIFO is designed to avoid.

Triggers
========

Three things can trigger a shutdown:

- **The user closes the main window.** Qt fires its lastWindowClosed
  signal, which calls ``request_shutdown`` directly.
- **An OS signal.** ``SIGINT`` (Ctrl-C) and ``SIGTERM`` are caught
  by a Python signal handler that sets a ``threading.Event``. A Qt
  ``QTimer`` polls that event every 100 ms and calls
  ``request_shutdown`` when it fires. The poll-via-timer indirection
  exists because ``signal.signal`` callbacks are not allowed to
  touch the asyncio loop directly.
- **An unrecoverable startup failure.** If a service's ``initialize``
  returns ``False`` or raises, ``qt_main`` calls
  ``request_shutdown`` itself.

All three funnel through ``request_shutdown``, which sets the cancel
token and signals the asyncio shutdown event. Whatever was awaiting
``lifecycle.wait()`` wakes up and runs ``teardown``.

Where to read next
==================

The lifecycle owns *resources*; some of those resources own
*persisted state*. The storage and configuration layer that sits
underneath the services — atomic JSON, a TTL-cached reader, a live
configuration store — is the subject of :doc:`storage`.
