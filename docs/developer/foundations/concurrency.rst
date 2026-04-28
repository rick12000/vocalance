Concurrency
###########

The feature chapters spoke about events being delivered, callbacks
being invoked, and OS calls being made, without saying *which thread*
any of it ran on. This chapter answers that question. It is the only
chapter where the two execution contexts that actually run the
application are described; once you have read it, every "the
service does X" sentence in the rest of the guide can be re-read
with a precise picture of which thread X happens on.

The single-thread model
=======================

For the vast majority of the code, Vocalance is a single-threaded
application. There is one operating-system thread that runs both the
Qt event loop and the asyncio event loop, in cooperation. That
thread is the **main thread**, and almost everything in the code
runs on it:

- Every event-bus dispatch.
- Every service handler.
- Every UI controller and every Qt signal.
- Every ``await``.

The Qt and asyncio loops share the thread because they are
integrated through PySide6's ``QtAsyncio`` module — calling
``QtAsyncio.run(...)`` schedules the asyncio worker on Qt's main
event loop. The two loops cooperate by yielding to each other; from
the application's perspective there is only one place work happens.

The single-thread guarantee is what makes the bus contract from the
previous chapter useful. "Event A finishes before event B begins"
would be hard to enforce across multiple threads; on a single thread
it is automatic, because no two handlers can run at the same instant
in the first place.

The two exceptions
==================

Two pieces of work do not run on the main thread, and they are the
only two:

1. **The audio device callback** — the OS-level callback that hands
   raw microphone buffers to the application — runs on a thread
   that the audio library, PortAudio, owns. We do not control it.
2. **Heavy synchronous work** — running an STT model, generating
   LLM tokens, calling ``pyautogui`` to inject input — is pushed to
   short-lived daemon threads so it does not block the main thread.
   We *do* control the spawn, but the work itself runs elsewhere.

The rest of this chapter is about those two cases: how each one
crosses back into the main thread, and how a small set of helpers
makes those crossings consistent across the codebase.

Crossing into the main thread
=============================

Both kinds of foreign thread eventually need to invoke code that
expects to run on the main thread (a service handler, an event bus
publish, a callback). The mechanism for that crossing is a single
asyncio primitive:

.. code-block:: python

   loop.call_soon_threadsafe(callable, *args)

``call_soon_threadsafe`` is the only asyncio API that is safe to
call from a different thread. It does two things:

- Records the callable in a queue the loop reads from.
- Wakes the loop if it is currently idle.

When the loop next ticks, it picks up the callable and runs it on
its own thread. From the callable's perspective, it is back on the
main thread; nothing it touches has to be thread-safe.

Two helpers in
``vocalance/app/lifecycle/worker.py`` wrap this primitive:

.. code-block:: python

   def schedule_on_loop(loop, coro) -> None:
       loop.call_soon_threadsafe(loop.create_task, coro)

   def schedule_on_loop_callback(loop, fn, *args) -> None:
       loop.call_soon_threadsafe(fn, *args)

The first runs a coroutine, the second runs a synchronous callable.
Together they cover every "I am on a foreign thread and I need to
hand something to the main thread" case in the codebase.

The microphone crossing
-----------------------

The audio capture service is the one component that actually runs on
PortAudio's thread. Its callback is invoked by the audio driver
directly, not by anything the application owns. The callback's job
is therefore to do the bare minimum on the foreign thread and hop
the result over to the main thread:

.. code-block:: python

   def _portaudio_callback(self, indata, frames, time_info, status):
       pcm_bytes = indata.tobytes()
       timestamp = time.time()
       self.loop.call_soon_threadsafe(self._publish_chunk, pcm_bytes, timestamp)

   def _publish_chunk(self, pcm_bytes, timestamp):
       asyncio.create_task(
           self.event_bus.publish(
               AudioChunkCapturedEvent(
                   pcm_bytes=pcm_bytes, timestamp=timestamp, sample_rate=self.sample_rate
               )
           )
       )

Three properties of this callback matter:

- It copies the bytes (it must — PortAudio's buffer is only valid
  for the duration of the callback). After the call returns, nothing
  on the audio thread retains a reference to the buffer.
- ``_publish_chunk`` runs on the main thread; it is what actually
  publishes the bus event.
- The callback returns quickly. The audio thread is back to
  listening for the next buffer within microseconds.

Anything heavier on the audio thread — running a model, taking a
lock, allocating a large object — would risk dropping audio. That
is the rule of thumb: the audio thread copies and schedules,
nothing else.

Heavy synchronous work
======================

The second category of off-main-thread work is the opposite
direction: code that runs on the main thread but cannot afford to
*stay* there. Running a Vosk recognition takes hundreds of
milliseconds. Calling ``pyautogui.click`` blocks until the OS
confirms the click. Generating an LLM token blocks for as long as
the model takes. Doing any of that on the main thread would freeze
the UI loop.

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
  ``call_soon_threadsafe`` mechanism the audio recorder uses.
- The caller's coroutine awaits a future that completes when the
  result lands.

Two design choices justify themselves:

- **Daemon threads.** A daemon thread cannot keep the interpreter
  alive past process exit. If a native call fails to return, it
  cannot stop the application from shutting down — Python kills the
  thread when the process exits.
- **One thread per call.** There is no shared thread pool. A pool
  would introduce the question of pool starvation; one-thread-per-call
  has no upper bound on concurrency, and the cost of spawning a
  thread is negligible compared to the work that follows.

Cancellation
------------

If the awaiting coroutine is cancelled (typically during application
shutdown), ``run_blocking`` sets a cancellation token before
re-raising. Cooperating workers — the speech and LLM engines — poll
the token and return early when it is set. The token mechanism
itself belongs to the lifecycle and is detailed in
:doc:`lifecycle`.

Ordering OS input
=================

A subtle problem follows from the "spawn one thread per blocking
call" rule. The grid service, the mark service, and the automation
service can all reach for ``pyautogui`` at roughly the same time.
If three calls each spawn their own thread, the OS receives the
clicks in whatever order the three threads happen to land — which
is not the order the user spoke the commands.

The fix is a single shared service, ``KeyboardInputService``
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
- The ``asyncio.Lock`` is acquired *before* the thread is spawned,
  released *after* it joins. Only one ``run`` call is in flight at a
  time, in strict FIFO order.

Every executor that touches ``pyautogui`` routes through this
service. A sequence of "click, click, scroll up" therefore reaches
the OS in that order, even though three different services made the
calls.

Summary
=======

Three rules cover every line of concurrent code in the application:

#. Almost everything runs on the main thread, which hosts both the
   Qt loop and the asyncio loop.
#. Anything that arrives from a foreign thread (the audio device)
   re-enters the main thread via ``loop.call_soon_threadsafe``.
#. Anything synchronous that would block the main thread for too
   long is dispatched to a daemon thread via ``run_blocking``, and
   its result re-enters via the same primitive.

The remaining foundations chapters use those rules without
re-explaining them. :doc:`lifecycle` describes the orchestration
that builds the services and tears them down; :doc:`storage`
describes the persistence layer that sits underneath them.
