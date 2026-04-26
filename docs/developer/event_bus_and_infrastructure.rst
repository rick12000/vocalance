Event Bus & Infrastructure
############################

Vocalance's infrastructure enables complex coordination through an event bus, thread pool architecture, and service lifecycle management. This foundation allows the application to remain responsive while performing audio capture, speech recognition, LLM inference, and UI updates.

Why Event-Driven Architecture?
==============================

Vocalance must handle multiple concurrent operations: capturing audio in real time, recognizing speech, parsing commands, coordinating dictation state, updating the UI, and more. Traditional architectures would have these components calling each other directly, creating tight coupling and potential deadlocks when threads compete for locks.

An event-driven architecture decouples these components: each publishes events describing what it did, and others subscribe to events they care about. This loose coupling means components can be developed, tested, and modified independently. It also provides natural serialization: events arrive at the event bus in order, are processed sequentially, and handlers can safely assume they won't race with other operations on the same event type.

The Event Bus: Central Nervous System
=======================================

The ``EventBus`` is the central routing system. All components—audio service, STT service, command parser, controllers, services—communicate by publishing and subscribing to events through the bus. The bus guarantees:

- **Sequential processing**: Only one event is processed at a time, preventing race conditions and guaranteeing causal ordering.
- **Async concurrency**: Within a single event, synchronous handlers run sequentially, and all asynchronous handlers run concurrently via ``asyncio.gather`` for maximum efficiency.
- **Backpressure management**: The event queue has a bounded size (maxsize=500). If the system is catastrophically overloaded, publishing will block, applying backpressure to the publisher.
- **Thread-safe subscriptions**: Components can subscribe from any thread without contention.

Publishing an Event
--------------------

Publishing is simple and non-blocking under normal conditions. A component creates an event (a subclass of ``BaseEvent``) and calls ``await event_bus.publish(event)``:

.. code-block:: python

   event = MarkCommandParsedEvent(command=MarkCreateCommand(label="home", ...))
   await event_bus.publish(event)

The ``publish`` method adds the event to the queue and returns. It will only block if the event queue is full, preventing memory exhaustion.

Subscribing to Events
---------------------

During initialization, services and controllers subscribe to event types they handle:

.. code-block:: python

   def setup_subscriptions(self):
       self.event_bus.subscribe(
           event_type=MarkCommandParsedEvent,
           handler=self._handle_mark_command_parsed
       )

The ``subscribe`` method registers a callable (sync or async function) to be invoked whenever an event of that type (or a subclass, via MRO dispatch) is published.

Event Processing Flow
------------------------------------

The event bus runs a background worker task that continuously dequeues events and invokes handlers. This worker is the single point of serialization:

1. The event type is matched against registered subscriptions (including parent classes via MRO).
2. All handlers for that event type are collected.
3. Synchronous handlers are executed sequentially and immediately.
4. Asynchronous handlers are collected into tasks.
5. All asynchronous handlers are executed concurrently using ``asyncio.gather``. The bus waits for all async handlers to finish before moving to the next event, ensuring strict inter-event ordering.
6. Exceptions in handlers are caught, logged, and don't prevent other handlers from running.

Threading Architecture
======================

Vocalance must remain responsive to user input while handling real-time audio capture and performing CPU-intensive operations like speech recognition and LLM inference. This requires careful threading:

**Main Thread (Qt + Asyncio)**: Vocalance uses ``PySide6.QtAsyncio`` to integrate the asyncio event loop with the Qt event loop on the **same main thread**. All widget creation, UI updates, signal/slot handling, event bus processing, and async service operations occur here. This thread must never block—if a handler blocks for too long, the UI becomes unresponsive.

**Audio Capture Thread**: The recorder thread continuously reads ~30 ms PCM frames via PortAudio callbacks. Raw dictation PCM is forwarded synchronously to a coordinator callback; command/sound VAD is fed via thread-safe calls to the main asyncio loop. This path must stay lightweight so the input device buffer does not overrun.

**Thread Pools**: CPU-intensive or blocking operations are offloaded to thread pools using ``loop.run_in_executor()``. For example, PyAutoGUI automation commands, file I/O for saving models, and LLM inference run in background threads to avoid blocking the main Qt/Asyncio thread.

Cross-Thread Communication
---------------------------

With multiple threads, special care is needed when one thread needs to communicate with another:

**Publishing events from native threads**: Audio chunks captured on the PortAudio thread are scheduled to be processed on the main asyncio loop using ``loop.call_soon_threadsafe()``.

**UI Updates**: Because the asyncio event loop and Qt event loop share the main thread, async event handlers can safely update the UI directly or emit Qt Signals. However, any background thread (like the automation executor) must not touch the UI directly.

State Management and Locking
=============================

Shared state accessed from multiple threads must be protected by synchronization primitives.

**threading.RLock and threading.Lock**: Used to protect state that might be accessed by both the main thread and background threads (e.g., the audio capture thread or the automation thread pool).

.. code-block:: python

   class DictationCoordinator:
       def __init__(self):
           self.state_lock = threading.RLock()
           self.current_state = DictationState.IDLE

       def set_state(self, new_state: DictationState) -> None:
           with self.state_lock:
               # Validate and transition state
               self.current_state = new_state

**Fine-Grained Locking**: Locks should be held for as little time as possible. Compute-intensive operations should happen outside the lock.

Service Lifecycle: Initialization and Shutdown
================================================

Services are initialized in stages, activated, and finally shut down gracefully. The lifecycle is coordinated in ``qt_main.py``.

Initialization Sequence
---------------------

1. **Configuration and Logging**: Load ``GlobalAppConfig`` and set up logging.
2. **UI Setup**: Initialize Qt application, load fonts, apply stylesheet, and show the ``StartupWindow``.
3. **Service Construction**: Instantiate all services (Storage, Grid, Automation, Marks, Audio, STT, Dictation, etc.) and wire configuration listeners.
4. **Event Bus Start**: Start the event bus worker task on the asyncio loop.
5. **Service Initialization**: Run async ``initialize()`` methods on services. This includes loading user settings, initializing STT models (Vosk, Moonshine), and loading sound recognition models (YAMNet).
6. **Main Window**: Create and show the ``VocalanceMainWindow``, then close the startup window.

Progress Tracking
-----------------

During initialization, the ``StartupProgressTracker`` updates the startup window with status and progress bars. This provides visual feedback during the startup sequence and prevents the UI from appearing frozen while models load.

Graceful Shutdown
------------------

Shutdown is managed by ``AppLifecycle`` (``vocalance.app.lifecycle``), which owns the cancellation token, the asyncio shutdown event, the initialization task, every tracked background task, and a LIFO stack of registered ``AsyncCloseable`` resources.

1. **Request Shutdown**: Triggered by the user closing the main window, OS signals (SIGINT/SIGTERM), or any failure path; ``AppLifecycle.request_shutdown`` is thread-safe and idempotent.
2. **Cancel Token**: ``CancellationToken`` is set, propagating to every cooperating sync worker and any per-operation events linked via ``link_event``.
3. **Cancel Init Task**: An in-flight ``initialize`` coroutine is cancelled and awaited under a short grace period.
4. **Cancel Background Tasks**: Every task tracked via ``track_background_task`` is cancelled and awaited.
5. **Close Resources (LIFO)**: ``AsyncCloseable.shutdown`` is invoked on each registered resource in reverse registration order, so audio and dictation are torn down before the engines they depend on.
6. **Drain Default Executor**: ``loop.shutdown_default_executor`` is awaited so non-daemon worker threads from ``asyncio.to_thread``/``run_in_executor`` cannot outlive the lifecycle.
7. **Stop Signal Timer**: The Qt poll timer that bridges OS signals onto the GUI loop is stopped.

**Signal Handlers**: When the user presses Ctrl+C or the OS sends SIGTERM, the handlers set an internal event that the Qt poll timer flips into ``AppLifecycle.request_shutdown``, ensuring shutdown runs on the GUI loop.

Infrastructure Summary
======================

The infrastructure provides a foundation for responsive coordination:

1. **Event bus**: Asynchronous pub/sub with sequential processing and concurrent async handlers.
2. **Threading model**: Integrated Qt/Asyncio main thread, dedicated audio capture thread, and thread pools for blocking operations.
3. **State management**: Locks protect shared state, atomic transitions prevent race conditions.
4. **Service lifecycle**: Coordinated initialization with progress tracking; reverse-order shutdown ensures clean resource release.

This foundation enables Vocalance to perform real-time audio capture, concurrent speech recognition, LLM inference, and responsive UI updates without the complexities of traditional multi-threaded programming. The event-driven design means components are loosely coupled, testable independently, and easily extended with new functionality.
