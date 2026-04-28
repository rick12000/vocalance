Concurrency
###########

.. sectnum::

:doc:`event_bus` described what the bus guarantees — sequential delivery,
within-event concurrency — but not *how* those guarantees are achievable given
that some work takes hundreds of milliseconds and the audio device invokes
callbacks on a thread the application does not own. This chapter explains the
concurrency model from first principles.

Vocalance has three hard timing constraints that cannot all be satisfied on a
single naive thread:

- The microphone delivers audio **every ~30 ms**. A late read drops frames.
- The UI must repaint at **~60 fps**, meaning no single task can take more than
  ~16 ms between repaints.
- Speech recognition takes **100–300 ms**. LLM generation takes **seconds**.

This chapter explains, from first principles, the two concurrency tools Python
provides, why Vocalance chooses one as the default and uses the other
selectively, and how the two communicate safely.

The Two Concurrency Models
===========================

asyncio: One Thread, Many Tasks
---------------------------------

asyncio is a *cooperative* concurrency system. It runs entirely on a single OS
thread. On that thread lives an **event loop** that maintains a list of
*coroutines* — functions defined with ``async def`` that can be suspended and
resumed. The loop picks one coroutine, runs it until it encounters an ``await``,
suspends it, picks another, and so on.

The critical property is where suspensions happen: **only at ``await``**. A
coroutine that never awaits runs to completion without interruption. Nothing
else can execute while it runs. This makes it safe to read and write shared
state between two ``await`` points without any locking — no other coroutine
can observe partial state during that window.

The flip side: a coroutine that never yields *blocks everything else*. A tight
loop that runs for 300 ms freezes the event loop — and everything that depends
on it, including the UI — for the entire 300 ms. asyncio gives safety and
simplicity for anything that spends most of its time waiting; it is harmful for
anything that spends most of its time computing.

Threads: Multiple Threads, Preemptive Switching
-------------------------------------------------

Threads are the OS's mechanism for running multiple execution flows. Unlike
asyncio, the OS scheduler decides when to switch between threads and does so
without any cooperation from the running code — a thread can be preempted
mid-statement.

The advantage is that multiple threads can genuinely run in parallel on separate
CPU cores. Python's **Global Interpreter Lock** (GIL) limits this: the GIL
allows only one thread to execute Python bytecode at a time. However, the GIL
is released for blocking system calls (file I/O, network I/O) and for most
C-extension calls. Every heavy model used in Vocalance — Vosk, YAMNet, Moonshine,
and the LLM — is a C extension that releases the GIL during inference. Threads
running those models run in true parallel with the main Python thread.

The cost is that preemptive switching makes shared mutable state dangerous.
Two threads reading and writing the same variable without coordination can
produce results that depend on thread scheduling. Synchronization primitives
(locks, queues, futures) are needed wherever threads share state.

The Vocalance Concurrency Model
=================================

The Default: asyncio on One Thread
------------------------------------

Almost everything in Vocalance runs on a single OS thread shared by two
cooperating event loops: the Qt event loop (which drives the UI) and the asyncio
event loop (which drives all services and the bus). The integration is one call:

.. code-block:: python

   QtAsyncio.run(start_app())

``QtAsyncio`` (from PySide6) installs an asyncio event loop implementation that
schedules its turns through Qt's existing event loop. The two loops interleave
on one thread: Qt paints a frame, yields to asyncio, asyncio runs a few
coroutines, yields back to Qt, and so on.

.. mermaid::

   flowchart LR
       Thread[Single OS Thread]
       Thread --> Qt[Qt event loop<br/>paints, signals]
       Thread --> Aio[asyncio event loop<br/>tasks, awaits]
       Qt -. yields to .- Aio
       Aio -. yields to .- Qt

Every bus dispatch, every service handler, every Qt signal, every coroutine
suspension happens on this one thread, in a single total order. This is what
gives the bus's sequential-delivery guarantee its meaning: only one thing can
execute at a time.

The Exceptions: Two Categories of Off-Thread Work
---------------------------------------------------

Two categories of work cannot run on the main thread without violating the
timing constraints.

**Category 1: A foreign thread the application does not own.** The PortAudio
audio driver invokes its callback on a thread it manages. The application cannot
avoid this; it is how the audio API works. The callback must copy the audio
buffer and return in microseconds, or the driver's internal buffer overflows
and frames are dropped.

**Category 2: CPU-heavy synchronous work.** Speech recognition, sound
embedding, LLM generation, and OS input calls are all blocking operations that
take far longer than a UI frame. Running them on the main thread would freeze the
loop.

Both categories require getting work *off* the main thread and getting results
*back* to it.

The CPU-Heavy Jobs
===================

The following table lists every operation in the application that is too slow to
run on the main thread and must be dispatched to a background thread.

.. list-table::
   :widths: 27 27 23 23
   :header-rows: 1
   :class: uniform-rows

   * - Job
     - Where in the code
     - Cost
     - Frequency
   * - Vosk command recognition
     - ``VoskEngine.recognize``
     - 100–300 ms (C++ Kaldi decoder)
     - Once per spoken command clip
   * - YAMNet sound embedding
     - ``SoundRecognizer.recognize_sound``
     - 50–100 ms (TensorFlow CPU)
     - Once per sound clip
   * - LLM token generation
     - ``LLMService`` (llama-cpp-python)
     - Seconds total, token-by-token streaming
     - Once per Smart / Amend session
   * - ``pyautogui`` OS input
     - ``KeyboardInputService``
     - 5–50 ms per call (OS-blocking)
     - Per executed command
   * - Storage I/O
     - ``StorageService``
     - Low single-digit ms
     - Per JSON read or write
   * - Model loading (all four)
     - Service ``initialize``
     - Seconds to tens of seconds
     - Once at startup

One notable absence from the table is **Moonshine streaming inference**.
``MoonshineStreamSession.add_audio_pcm16`` runs on the main thread. The
Moonshine native library releases the GIL during inference, so it does not block
other Python threads, but the main thread is occupied for a few milliseconds per
chunk. The chunks are small enough that this is below the frame budget.

The Thread Crossing Primitives
================================

Moving Results from a Foreign Thread to the Main Thread
---------------------------------------------------------

The only asyncio API that is safe to call from a thread other than the one
running the event loop is ``loop.call_soon_threadsafe``:

.. code-block:: python

   loop.call_soon_threadsafe(callable, *args)

When called from any thread, it places ``callable`` into a thread-safe internal
queue that the event loop reads on its next tick, and wakes the loop if it is
currently idle. The callable runs on the main thread, in the same single-thread
context as everything else. The foreign thread returns immediately after
scheduling; it does not wait for the callable to execute.

Two wrappers in ``vocalance/app/lifecycle/worker.py`` cover the two typical
cases:

.. code-block:: python

   def schedule_on_loop(loop, coro) -> None:
       loop.call_soon_threadsafe(loop.create_task, coro)

   def schedule_on_loop_callback(loop, fn, *args) -> None:
       loop.call_soon_threadsafe(fn, *args)

The first wraps a coroutine in a task; the second schedules a synchronous
callable. Together they cover every "I am on a foreign thread and need to hand
something to the main thread" case in the codebase.

The PortAudio Crossing
-----------------------

The capture service is the one place where a foreign thread must hand data to
the main thread on every audio frame — roughly thirty times per second.

.. mermaid::

   flowchart LR
       subgraph Foreign["Driver Thread (PortAudio)"]
           Drv[PortAudio driver] -->|PCM buffer| CB[_portaudio_callback<br/>copy bytes, record timestamp]
       end
       subgraph Main["Main Thread"]
           Pub[_publish_chunk] -->|AudioChunkCapturedEvent| Bus((Event bus))
       end
       CB -.->|call_soon_threadsafe| Pub

The callback's three-step contract: copy the bytes (the buffer pointer is only
valid for the duration of the callback), record the timestamp, and schedule
``_publish_chunk`` on the main thread. The callback returns in microseconds.
``_publish_chunk`` runs on the main thread, constructs the event, and publishes
to the bus. All subscriber dispatch happens on the main thread with no
threading hazards.

``run_blocking``: Dispatching Heavy Work to a Background Thread
================================================================

When the main thread needs to run blocking work, the pattern is the inverse of
the audio crossing: spawn a thread, let it run the blocking call, and have it
hand the result back to the main thread when done.

``run_blocking`` (``vocalance/app/lifecycle/worker.py``) implements this:

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

       try:
           return await future
       except asyncio.CancelledError:
           if cancel_token is not None:
               cancel_token.set()
           raise

The calling coroutine creates a ``Future`` — a placeholder that will be resolved
when the result is ready. A daemon thread is spawned to run the blocking
function. When the thread finishes, it calls ``call_soon_threadsafe`` to resolve
the future on the main thread. The calling coroutine was suspended on
``await future``; resolving the future schedules it to resume on the next loop
tick. From the caller's perspective this looks like a normal ``await``.

.. mermaid::

   flowchart LR
       Caller[Caller coroutine<br/>main thread] -->|1. create future, spawn thread| W[Daemon thread<br/>runs fn]
       W -->|2. call_soon_threadsafe<br/>resolve future| Loop[Event loop<br/>main thread]
       Loop -->|3. resume caller| Caller

Why Daemon Threads
-------------------

``daemon=True`` means the interpreter does not wait for these threads to finish
before exiting. If a C-extension call misbehaves — segfault, deadlock, infinite
loop — it cannot keep the process alive past shutdown. The graceful path is
cooperative cancellation via ``CancellationToken`` (below). Daemon is the safety
net when graceful cancellation does not complete.

Why One Thread Per Call
------------------------

The alternative to one-per-call is a shared thread pool
(``ThreadPoolExecutor``). The codebase avoids this for two reasons.

First, **pool starvation**. If the LLM occupies all pool workers for thirty
seconds, every other blocking call queues behind it. With one-per-call there is
no upper bound on concurrency — Vosk, YAMNet, and a click can all start
simultaneously on their own threads.

Second, **simplicity of reasoning**. Each ``run_blocking`` call is independent;
there is no shared pool state to reason about, no worker lifecycle to manage,
and no risk of one call affecting another's latency.

The cost is slightly higher thread-spawn overhead (~tens of microseconds per
call), which is negligible compared to the hundreds of milliseconds of work each
thread performs.

Serializing OS Input
---------------------

One consequence of one-thread-per-call is that multiple callers can dispatch
``pyautogui`` calls simultaneously. If three spoken commands each spawn a thread
for OS input, the OS receives them in thread-scheduling order — not speech order.

``KeyboardInputService``
(``vocalance/app/services/keyboard_input_service.py``) solves this by combining
``run_blocking`` with an ``asyncio.Lock``:

.. code-block:: python

   class KeyboardInputService(Service):
       def __init__(self, event_bus: EventBus) -> None:
           super().__init__(event_bus)
           self._serial = asyncio.Lock()

       async def run(self, fn, *args, **kwargs):
           async with self._serial:
               return await run_blocking(fn, *args, name="vocalance-input", **kwargs)

The lock is acquired by a coroutine on the main thread before the daemon thread
is spawned, and released after the future resolves. At any moment, at most one
``pyautogui`` call is in flight. ``asyncio.Lock`` is used rather than
``threading.Lock`` because the contention is between coroutines on the main
thread, not between threads; an asyncio lock is compatible with
``async with`` and does not block the event loop when waiting.

Cancellation
=============

During shutdown, background threads must stop cooperatively. The mechanism is
``CancellationToken`` (``vocalance/app/lifecycle/cancellation.py``), an object
that wraps both an ``asyncio.Event`` and a ``threading.Event`` and keeps them
in sync.

.. mermaid::

   flowchart LR
       subgraph CT["CancellationToken"]
           AE[asyncio.Event]
           TE[threading.Event]
       end
       Aio[Awaiting coroutine] -->|await token.wait| AE
       Worker[Daemon worker] -->|token.is_set or<br/>threading_event().wait| TE
       Set[token.set] --> AE
       Set --> TE

Setting the token from any thread sets both internal events simultaneously.
Coroutines waiting on ``await token.wait()`` unblock on the next loop tick.
Daemon workers polling ``token.is_set()`` between iterations see the change on
their next poll. Workers that are blocking on a long operation can use
``token.threading_event().wait(timeout)`` to block for at most ``timeout``
seconds before checking again.

``run_blocking`` propagates cancellation automatically: if the awaiting coroutine
is cancelled while the daemon thread is still running, ``run_blocking`` calls
``cancel_token.set()`` so the daemon thread can observe the cancellation and
return early.

Summary: Three Rules
=====================

Every concurrency decision in the codebase follows three rules:

1. **Default: asyncio on one thread.** Services, handlers, UI controllers, and
   the bus all run on the asyncio loop, sharing one OS thread with Qt.

2. **Foreign thread → main thread via ``call_soon_threadsafe``.** The audio
   driver's callback is the only foreign thread. It copies, schedules, and
   returns in microseconds.

3. **Heavy blocking work → daemon thread via ``run_blocking``.** Model
   inference, LLM generation, and OS input calls run on per-call daemon
   threads. Results return to the main thread through a resolved future.

With the concurrency model established, the next chapter — :doc:`lifecycle` —
explains how all of these services, tasks, and threads are constructed, started,
and torn down in the correct order.
