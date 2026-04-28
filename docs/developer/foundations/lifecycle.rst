Lifecycle
#########

.. sectnum::

:doc:`concurrency` described the threading model. This chapter describes what
wraps around it: how every service, background task, and thread is constructed,
initialized, and torn down in a coordinated sequence.

The lifecycle owns every resource Vocalance allocates: services, background
tasks, threads, and OS signal handlers. It controls the order in which they are
created and the order in which they are destroyed. Both orders matter —
constructing in the wrong order produces services that depend on something not
yet ready; destroying in the wrong order produces races between components that
are still running and components that have already released their resources.

``AppLifecycle`` (``vocalance/app/lifecycle/lifecycle.py``) implements the
four phases: build, initialize, run, and teardown.

Building the Application
=========================

The Service Spec Pattern
-------------------------

Service construction is **declarative**. ``qt_main`` builds a list of
``ServiceSpec`` objects — each a name and a factory function — and hands the
list to the lifecycle. The lifecycle calls the factories in order, passing each
one a *container* dictionary of already-constructed services so that
dependencies resolve by name:

.. code-block:: python

   ServiceSpec(
       name="command_speech",
       factory=lambda c: CommandSpeechService(
           event_bus=c["event_bus"],
           config=c["config"],
           vosk_engine=c["vosk_engine"],
       )
   )

The full spec list in ``qt_main._service_specs`` is the single source of truth
for the service graph. Every service's dependencies are expressed as string keys
into the container; there are no circular imports, no global singletons, and no
implicit coupling between service files. Adding a new service is one edit:
append a ``ServiceSpec``.

Construction order is the order specs appear. Teardown order is the reverse.
This guarantee is the foundation that makes LIFO teardown work correctly.

Initialization
---------------

After all services are constructed, each one's ``initialize`` coroutine runs in
the same order as construction:

.. code-block:: python

   async def initialize(self) -> bool: ...

``initialize`` is where async setup lives — loading a model file from disk,
opening a device stream, reading stored configuration. These operations cannot
run in ``__init__`` because they are either coroutines (requiring an event loop)
or blocking (requiring a background thread). Any service whose
``initialize`` returns ``False`` or raises triggers a shutdown before the
application reaches the run phase.

Background Tasks
-----------------

Some services need long-running tasks that continue for the life of the
application. These are registered with ``lifecycle.spawn``:

.. code-block:: python

   def spawn(self, coro, *, name: str = "task") -> asyncio.Task:
       task = self._loop.create_task(coro, name=name)
       task.add_done_callback(self._log_task_exception)
       self._background_tasks.append(task)
       return task

The done-callback logs any unhandled exception, so a fire-and-forget task cannot
silently fail. The task handle is stored so teardown can cancel it.

Shutting Down
==============

Three paths lead to shutdown. All three converge on the same teardown function.

Shutdown Triggers
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :class: uniform-rows

   * - Trigger
     - Path
   * - User closes the main window
     - Qt's ``lastWindowClosed`` signal calls ``lifecycle.request_shutdown``.
   * - ``SIGINT`` / ``SIGTERM``
     - A Python signal handler sets a ``threading.Event``. A Qt ``QTimer``
       polls that event every 100 ms and calls ``lifecycle.request_shutdown``
       when it is set. (Signal handlers cannot touch the asyncio loop directly.)
   * - Init failure
     - A service's ``initialize`` returns ``False`` or raises; ``qt_main``
       calls ``lifecycle.request_shutdown``.

``request_shutdown`` sets the cancellation token, signals the asyncio shutdown
event, and returns. ``teardown`` is called by whatever is awaiting
``lifecycle.wait()``.

Cooperative Cancellation
-------------------------

Shutdown is cooperative — the lifecycle signals threads to stop rather than
killing them. The signal is a ``CancellationToken`` (described fully in
:doc:`concurrency`). The lifecycle holds one token for the entire application
and exposes it as ``cancel_token``. Setting it is the first step of teardown.

Any service or background task that needs to run code during shutdown should
check ``cancel_token.is_set()`` in its loop, or ``await cancel_token.wait()``
at a suspension point. The lifecycle does not provide any stronger guarantee
than "the token will be set before resources are released".

The Teardown Sequence
----------------------

Teardown runs six steps, each completing fully before the next begins:

.. mermaid::

   flowchart TD
       T0[teardown called] --> T1[1. Set cancel token]
       T1 --> T2[2. Cancel init task, await it]
       T2 --> T3[3. Cancel background tasks, await all]
       T3 --> T4[4. Close resources in LIFO order]
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

``teardown`` is idempotent — a second call returns immediately. All three
shutdown triggers funnel into the same method, so whichever fires first runs
teardown, and any subsequent trigger is a no-op.

Why LIFO Order Matters
~~~~~~~~~~~~~~~~~~~~~~

Destroying resources in reverse-construction order is a correctness requirement.
Consider the capture path. ``AudioCaptureService`` is registered early because
the segmenters and dictation coordinator subscribe to its bus event during their
own construction — they need the bus and the event type definition to exist, but
they do not need the capture service to exist yet. ``CommandSegmenterService``,
``SoundSegmenterService``, and ``DictationCoordinator`` are registered later.

LIFO teardown means the segmenters and coordinator shut down *first*. Each
``Service.shutdown`` releases all bus subscriptions through the
``SubscriptionTracker``. By the time ``AudioCaptureService`` shuts down,
no subscriber is left for ``AudioChunkCapturedEvent``. The capture service stops
the PortAudio stream and exits cleanly.

If the order were reversed — capture service first — the PortAudio stream would
stop while the segmenters and coordinator were still alive and subscribed.
Further, if any of those services published an event during their own shutdown,
the bus would attempt to deliver it to components that have already released
their resources.

This pattern generalises: any service that *produces* data should be shut down
*after* all services that *consume* its data. LIFO over construction order
satisfies this requirement automatically, as long as producers are constructed
before consumers — which is exactly what the dependency-ordered spec list
guarantees.

The final foundations chapter — :doc:`storage` — covers how state persists
between sessions: the typed JSON persistence layer and the live configuration
store that keeps every service's parameters in sync with the user's settings.
