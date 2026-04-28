Concurrency
###########

The feature chapters spoke about events being delivered,
callbacks being invoked, and OS calls being made, without saying
*which thread* any of it ran on. This chapter answers that
question.

Threading model at a glance
===========================

Almost everything in Vocalance runs on one thread. The two
exceptions are the audio device callback and any synchronous
work that would block the main thread for too long.

.. mermaid::

   flowchart LR
       subgraph Main["Main thread<br/>(Qt + asyncio)"]
           Bus((Event bus))
           Svcs[All services]
           UI[Qt views]
       end
       PA[PortAudio thread<br/><i>OS-owned</i>] -->|call_soon_threadsafe| Bus
       Heavy[Daemon threads<br/><i>spawned per call</i>] -->|call_soon_threadsafe| Bus
       Bus -. dispatch .-> Svcs
       Bus -. dispatch .-> UI
       Svcs -->|run_blocking| Heavy

The single main thread is what makes the bus contract from the
previous chapter useful. "Event A finishes before event B begins"
is automatic on a single thread: no two handlers can run at the
same instant in the first place.

The single-thread model
=======================

For the vast majority of the code, Vocalance is single-threaded.
One operating-system thread runs both the Qt event loop and the
asyncio event loop in cooperation. Almost everything happens on
this thread:

- Every event-bus dispatch.
- Every service handler.
- Every UI controller and Qt signal.
- Every ``await``.

Qt and asyncio share the thread because they are integrated
through PySide6's ``QtAsyncio``: ``QtAsyncio.run(...)`` schedules
the asyncio worker on Qt's main event loop. The two loops yield
to each other; from the application's perspective there is only
one place work happens.

Crossing into the main thread
=============================

Both kinds of foreign thread (PortAudio and the daemon spawn)
eventually need to invoke code that expects to run on the main
thread (a service handler, a bus publish, a callback). The
mechanism for that crossing is one asyncio primitive:

.. code-block:: python

   loop.call_soon_threadsafe(callable, *args)

It does two things:

- Records the callable in a queue the loop reads from.
- Wakes the loop if it is currently idle.

When the loop next ticks, it picks up the callable and runs it
on its own thread. From the callable's perspective, it is back
on the main thread; nothing it touches has to be thread-safe.

Two helpers in
``vocalance/app/lifecycle/worker.py`` wrap this primitive:

.. code-block:: python

   def schedule_on_loop(loop, coro) -> None:
       loop.call_soon_threadsafe(loop.create_task, coro)

   def schedule_on_loop_callback(loop, fn, *args) -> None:
       loop.call_soon_threadsafe(fn, *args)

The first runs a coroutine, the second a synchronous callable.
Together they cover every "I am on a foreign thread and I need
to hand something to the main thread" case in the codebase.

The PortAudio crossing
----------------------

The audio capture service is the only component that runs on
PortAudio's thread. Its callback is invoked by the audio driver
directly. The callback's job is to do the bare minimum on the
foreign thread and hop the result over to the main thread.

.. mermaid::

   sequenceDiagram
       participant Drv as PortAudio driver
       participant CB as _portaudio_callback
       participant Loop as asyncio loop
       participant Pub as _publish_chunk
       participant Bus as Event bus

       Drv->>CB: input buffer
       CB->>CB: copy bytes,<br/>take timestamp
       CB->>Loop: call_soon_threadsafe(_publish_chunk, ...)
       CB-->>Drv: return
       Loop->>Pub: invoke on main thread
       Pub->>Bus: publish AudioChunkCapturedEvent

Three properties of this callback matter:

- It copies the bytes (it must — PortAudio's buffer is only
  valid for the duration of the callback). After return,
  nothing on the audio thread retains a reference.
- ``_publish_chunk`` runs on the main thread; it is what
  actually publishes the bus event.
- The callback returns within microseconds.

Anything heavier on the audio thread — running a model, taking
a lock, allocating a large object — would risk dropping audio.
Rule of thumb: copy and schedule, nothing else.

Heavy synchronous work
======================

The other category of off-main-thread work is the opposite
direction: code that *runs on* the main thread but cannot afford
to *stay* there. Vosk recognition takes hundreds of
milliseconds. ``pyautogui.click`` blocks until the OS confirms.
LLM token generation blocks for as long as the model takes. Any
of that on the main thread freezes the UI loop.

The helper for this case is ``run_blocking``
(``vocalance/app/lifecycle/worker.py``):

.. code-block:: python

   async def run_blocking(fn, *args, cancel_token=None, name=..., **kwargs) -> T:
       loop = asyncio.get_running_loop()
       future = loop.create_future()

       def worker() -> None:
           try:
               result = fn(*args, **kwargs)
           except BaseException as exc:
               loop.call_soon_threadsafe(future.set_exception, exc)
           else:
               loop.call_soon_threadsafe(future.set_result, result)

       threading.Thread(target=worker, daemon=True, name=name).start()
       return await future

The mechanics are tight:

- A fresh daemon thread is spawned for the call.
- The synchronous function runs on that thread.
- The result hops back to the main thread via the same
  ``call_soon_threadsafe`` primitive.
- The caller's coroutine awaits a future that completes when the
  result lands.

Two design choices justify themselves:

- **Daemon threads.** A daemon thread cannot keep the
  interpreter alive past process exit. If a native call fails to
  return, it cannot stop the application from shutting down —
  Python kills the thread when the process exits.
- **One thread per call.** No shared pool. A pool would
  introduce the question of pool starvation; one-thread-per-call
  has no upper bound on concurrency, and the cost of spawning a
  thread is negligible compared to the work that follows.

If the awaiting coroutine is cancelled (typically during
shutdown), ``run_blocking`` sets a cancellation token before
re-raising. Cooperating workers — the speech and LLM engines —
poll the token and return early when it is set. The token
mechanism itself belongs to :doc:`lifecycle`.

Ordering OS input
=================

A subtle problem follows from "one thread per blocking call".
The grid service, the mark service, and the automation service
can all reach for ``pyautogui`` at roughly the same time. If
three calls each spawn their own thread, the OS receives the
clicks in whatever order the three threads land — which is not
the order the user spoke the commands.

The fix is one shared service, ``KeyboardInputService``
(``vocalance/app/services/keyboard_input_service.py``):

.. code-block:: python

   class KeyboardInputService(Service):
       def __init__(self, event_bus: EventBus) -> None:
           super().__init__(event_bus)
           self._serial = asyncio.Lock()

       async def run(self, fn, *args, **kwargs):
           async with self._serial:
               return await run_blocking(fn, *args, name="vocalance-input", **kwargs)

Two ideas combine in those four lines:

- ``run_blocking`` provides the off-main-thread execution.
- The ``asyncio.Lock`` is acquired *before* the thread is
  spawned, released *after* it joins. Only one ``run`` call is
  in flight at a time, in strict FIFO order.

Every executor that touches ``pyautogui`` routes through this
service. A sequence of "click, click, scroll up" reaches the OS
in that order, even though three different services made the
calls.

Three rules
===========

Every line of concurrent code in the application reduces to
three rules.

#. Almost everything runs on the main thread, which hosts both
   the Qt loop and the asyncio loop.
#. Anything arriving from a foreign thread (the audio device)
   re-enters the main thread via ``loop.call_soon_threadsafe``.
#. Anything synchronous that would block the main thread for too
   long is dispatched to a daemon thread via ``run_blocking``,
   and its result re-enters via the same primitive.

The remaining foundations chapters use those rules without
re-explaining them. :doc:`lifecycle` describes the orchestration
that builds the services and tears them down; :doc:`storage`
describes the persistence layer underneath them.
