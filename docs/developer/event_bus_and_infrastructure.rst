Event Bus & Infrastructure
############################

Vocalance's infrastructure enables complex, multi-threaded coordination through an event bus, thread pool architecture, and service lifecycle management. This foundation allows the application to remain responsive while performing audio capture, speech recognition, LLM inference, and UI updates—all simultaneously.

Why Event-Driven Architecture?
==============================

Vocalance must handle multiple concurrent operations: capturing audio in real time, recognizing speech, parsing commands, coordinating dictation state, updating the UI, and more. Traditional architectures would have these components calling each other directly, creating tight coupling and potential deadlocks when threads compete for locks.

An event-driven architecture decouples these components: each publishes events describing what it did, and others subscribe to events they care about. This loose coupling means components can be developed, tested, and modified independently. It also provides natural serialization: events arrive at the event bus in order, are processed sequentially by priority, and handlers can safely assume they won't race with other operations on the same event type.

The Event Bus: Central Nervous System
=======================================

The `EventBus` is the central routing system. All components—audio service, STT service, command parser, controllers, services—communicate by publishing and subscribing to events through the bus. The bus guarantees:

- **Priority-based ordering**: Events are processed in priority order (CRITICAL before HIGH before NORMAL before LOW), ensuring critical operations aren't delayed by background tasks.
- **Sequential processing**: Only one handler runs at a time per event type, preventing race conditions.
- **Async support**: Handlers can be either sync or async, allowing the system to use both blocking and non-blocking code.
- **Backpressure management**: When the event queue fills up, low-priority events are dropped to prevent memory buildup.
- **Thread-safe subscriptions**: Components can subscribe from any thread without contention.

Publishing an Event
--------------------

Publishing is simple and non-blocking. A component creates an event (a subclass of `BaseEvent`) and calls `await event_bus.publish(event)`:

.. code-block:: python

   event = MarkCreateRequestEventData(name="home", x=100, y=200)
   await event_bus.publish(event)

The `publish` method validates the event, adds it to the priority queue with its priority level, and returns immediately. The caller doesn't wait for the event to be processed. This is crucial: if publishing were blocking, the audio thread would pause while the event bus processed events, causing audio drops.

**Backpressure Handling**: The event queue has a maximum size (default 200 events). When approaching capacity, the system takes action:

- At 50% capacity (100 events): info-level log message
- At 75% capacity (150 events): warning-level log message
- At max capacity and full: LOW and NORMAL priority events are dropped; HIGH and CRITICAL events are forced through but logged as an error indicating system overload

This ensures critical operations (shutdown, UI updates, audio processing) never get dropped while preventing unbounded memory growth under extreme load.

Subscribing to Events
---------------------

During initialization, services and controllers subscribe to event types they handle:

.. code-block:: python

   def setup_subscriptions(self):
       self.event_bus.subscribe(
           event_type=MarkCreateRequestEventData,
           handler=self._handle_mark_create_request
       )
       self.event_bus.subscribe(
           event_type=MarkCreatedEventData,
           handler=self._handle_mark_created
       )

The `subscribe` method registers a callable (sync or async function) to be invoked whenever an event of that type (or a subclass) is published. Subscriptions are thread-safe—a component can subscribe from any thread without locking issues.

Event Processing: Priority and Flow
------------------------------------

The event bus runs a background worker task that continuously dequeues events and invokes handlers. This worker is the single point of serialization: only one handler executes at a time per event type, but handlers for different event types can run concurrently if the system uses task scheduling (which Vocalance doesn't—it processes events sequentially for simplicity).

**Processing Sequence**: When an event is dequeued:

1. The event type is matched against registered subscriptions
2. All handlers for that event type are collected (or handlers for parent types if using inheritance)
3. Handlers execute sequentially in registration order
4. Both sync and async handlers are supported
5. Exceptions in handlers are caught, logged, and don't propagate
6. Slow handlers (>100ms) trigger a warning log
7. After all handlers complete, the worker applies an adaptive sleep based on event priority and queue depth

**Adaptive Sleep**: The worker doesn't busy-wait. Instead, it sleeps between events:

- CRITICAL priority: no sleep (immediate processing)
- Queue empty: 10ms sleep (low CPU usage, reduced latency)
- Queue light (< 10 events): no sleep (process immediately)
- Queue moderate (< 50 events): 1ms sleep
- Queue heavy (50+ events): 10ms sleep

This balances throughput (low latency when busy) with efficiency (low CPU when idle).

Event Types and Priority Levels
================================

All events inherit from `BaseEvent`, which defines a priority field. Priority is an enum with four levels:

- **CRITICAL** (value 10): Shutdown requests, safety-critical operations. Highest priority, never dropped.
- **HIGH** (value 20): User input, real-time audio processing, UI updates. Important and responsive.
- **NORMAL** (value 50): Command execution, STT results, standard events. Default priority.
- **LOW** (value 80): Logging, analytics, diagnostics. Can be dropped under load.

Events are ordered in the queue by priority, so a CRITICAL event published after 100 NORMAL events will be processed first.

Event Type Hierarchy
---------------------

Events are organized by functional area. The core events related to audio and recognition include `AudioChunkEvent` (50ms audio segments from the recorder), `CommandTextRecognizedEvent` (recognized command text from Vosk), `DictationTextRecognizedEvent` (recognized dictation from Whisper), `CustomSoundRecognizedEvent` (custom sound detection), and `CommandExecutedStatusEvent` (feedback on command execution results).

Command events (`AutomationCommandParsedEvent`, `MarkCommandParsedEvent`, etc.) represent parsed commands ready for execution. Dictation events coordinate the dictation workflow: `DictationStatusChangedEvent` indicates active/inactive state, `PartialDictationTextEvent` and `FinalDictationTextEvent` provide accumulated text, and `LLMTokenGeneratedEvent` delivers streaming LLM output for smart dictation modes.

Management events (`CommandMappingsUpdatedEvent`, `SettingsUpdatedEvent`, etc.) communicate configuration changes and operational state to services that need to adapt their behavior.

Threading Architecture
======================

Vocalance must remain responsive to user input while handling real-time audio capture and performing CPU-intensive operations like speech recognition and LLM inference. This requires careful threading:

**Main Thread (Tkinter)**: The primary thread running the Tkinter event loop. All widget creation, update, and event handling occurs here. When the user clicks a button, types text, or resizes the window, handlers fire on this thread. This thread must never block—if a handler blocks for too long, the UI becomes unresponsive.

**GUI Event Loop Thread**: A dedicated daemon thread running an asyncio event loop. This is where the event bus worker task runs, where async service operations execute, and where event handlers run. This thread is separate from the main thread because Tkinter and asyncio both need to run event loops, and they can't share one.

**Audio Thread**: Created by the audio service, this thread continuously captures audio in 50ms chunks and publishes them as events. It runs a tight loop and must not be preempted—any delay causes audio data to be dropped from the input buffer.

Together, these threads enable Vocalance to capture audio continuously (audio thread), process events asynchronously (GUI event loop thread), and remain responsive to user input (main thread).

Cross-Thread Communication
---------------------------

With multiple threads, special care is needed when one thread needs to communicate with another:

**Publishing Events**: The audio thread publishes `AudioChunkEvent` events via `await event_bus.publish(event)`. Because the event bus uses asyncio queues and thread-safe locks, this works safely from any thread.

**UI Updates from Event Handlers**: Event handlers run in the GUI event loop thread but must update the UI (which runs on the main thread). Controllers use `schedule_ui_update(callback, *args)`, which schedules the callback to run on the main thread via Tkinter's `root.after()` method. This ensures thread safety without explicit locks.

**Service Shutdown**: When the application shuts down, multiple threads need to be coordinated. The main thread requests shutdown, which signals the audio thread to stop, cancels tasks in the GUI event loop, and stops the GUI event loop thread. The shutdown coordinator manages this sequence.

State Management and Locking
=============================

Shared state accessed from multiple threads must be protected by synchronization primitives.

**asyncio.Lock**: Used in async code within the GUI event loop thread. These locks protect async-specific state without blocking the event loop:

.. code-block:: python

   class AudioListener:
       def __init__(self):
           self._state_lock = asyncio.Lock()
           self._audio_buffer = []

       async def _handle_audio_chunk(self, event):
           energy = calculate_energy(event.audio_chunk)
           async with self._state_lock:
               if not self._recording and energy > threshold:
                   self._recording = True
                   self._audio_buffer.append(event.audio_chunk)

**threading.RLock**: Used in sync code or when both sync and async code access the same state. RLock allows the same thread to acquire the lock multiple times:

.. code-block:: python

   class DictationCoordinator:
       def __init__(self):
           self._state_lock = threading.RLock()
           self._current_mode = DictationMode.INACTIVE

       def _start_dictation(self):
           with self._state_lock:
               self._current_mode = DictationMode.STANDARD
               self._current_session = DictationSession()

**Fine-Grained Locking**: Locks should be held for as little time as possible. Compute-intensive operations should happen outside the lock:

.. code-block:: python

   # WRONG: Computing inside lock
   async with self._lock:
       energy = calculate_energy(chunk)  # CPU-intensive, blocks others
       self._buffer.append(chunk)

   # RIGHT: Computing outside lock
   energy = calculate_energy(chunk)  # No lock, can do long operations
   async with self._lock:
       self._buffer.append(chunk)  # Brief lock, minimal contention

**Atomic State Transitions**: State machines enforce valid transitions using locks:

.. code-block:: python

   _VALID_TRANSITIONS = {
       State.IDLE: {State.RECORDING},
       State.RECORDING: {State.PROCESSING, State.IDLE},
       State.PROCESSING: {State.IDLE},
   }

   async def _transition_to(self, new_state):
       async with self._state_lock:
           if new_state not in _VALID_TRANSITIONS[self._current_state]:
               logger.error(f"Invalid transition: {self._current_state} → {new_state}")
               return False
           self._current_state = new_state
           return True

This ensures state never enters an invalid configuration even under concurrent events.

Service Lifecycle: Initialization and Shutdown
================================================

Services are initialized in stages based on dependencies, then activated, and finally shut down gracefully. The lifecycle is coordinated by `FastServiceInitializer` in `main.py`.

Initialization Stages
---------------------

**Stage 1 - Core Services**: GridService and AutomationService have no dependencies.

**Stage 2 - Storage Services**: StorageService (provides to others), then in parallel: SettingsService, CommandManagementService, MarkService, ClickTrackerService.

**Stage 3 - Audio Services**: AudioService (base audio capture), then in parallel: SoundService, SpeechToTextService, CentralizedCommandParser, DictationCoordinator (loads LLM model), MarkovCommandService.

**Stage 4 - UI Components**: FontService, then AppControlRoom (main window).

**Stage 5 - Activation**: Call `setup_subscriptions()` on each service to enable event processing.

Within each stage, services without mutual dependencies initialize in parallel using `asyncio.gather()`, reducing startup time from ~5-7 seconds to ~3-4 seconds.

Parallel Initialization Example
---------------------------------

.. code-block:: python

   async def _init_storage_services(self):
       storage = StorageService(config=self.config)

       async def init_settings():
           settings = SettingsService(storage=storage)
           await settings.initialize()

       async def init_commands():
           command_mgmt = CommandManagementService(storage=storage)

       async def init_marks():
           marks = MarkService(storage=storage)

       # Run in parallel
       await asyncio.gather(
           init_settings(),
           init_commands(),
           init_marks()
       )

Progress Tracking
-----------------

During initialization, the `StartupProgressTracker` updates the startup window with status and progress bars. This provides visual feedback during the ~3-4 second startup sequence and prevents the UI from appearing frozen.

Graceful Shutdown
------------------

Shutdown follows the reverse order of initialization:

1. **Stop audio processing**: Halt audio capture to end real-time operations
2. **Wait 300ms**: Allow pending audio chunks to be processed
3. **Stop event bus worker**: Prevent new events from being processed
4. **Cancel pending tasks**: Cancel all async tasks in the GUI event loop
5. **Stop GUI event loop**: Halt the GUI event loop thread
6. **Shutdown services in order**: Call `shutdown()` on each service in reverse initialization order
7. **Memory cleanup**: Run garbage collection and attempt to return memory to the OS
8. **Exit process**: Terminate cleanly with `os._exit(0)`

**Shutdown Coordinator**: The `ShutdownCoordinator` manages shutdown from multiple sources (user click, system signal, critical error). It ensures shutdown happens once and coordinated, collecting all errors and logging them.

**Signal Handlers**: When the user presses Ctrl+C or the OS sends SIGTERM, signal handlers request graceful shutdown. If shutdown doesn't complete in 5 seconds (e.g., a service hangs), a timeout forces the process to exit to prevent the application from becoming unresponsive.

Performance Monitoring and Diagnostics
========================================

The event bus provides statistics and diagnostic information for monitoring performance:

.. code-block:: python

   stats = await event_bus.get_stats()
   # {
   #   "queue_size": 12,
   #   "max_queue_size": 200,
   #   "queue_utilization": "6.0%",
   #   "events_dropped": 0,
   #   "subscribers": {
   #     "MarkCreateRequestEventData": 1,
   #     "AudioChunkEvent": 3,
   #     ...
   #   },
   #   "worker_status": "running",
   #   "is_shutting_down": False,
   # }

These stats help diagnose bottlenecks:

- **High queue utilization**: Event handlers are slow or many events are being published rapidly
- **Events dropped**: System is under heavy load and can't keep up
- **Slow handler warnings**: Individual handlers taking >100ms are logged, allowing identification of bottlenecks
- **Queue depth monitoring**: Progressive warnings at 50%, 75%, 90% capacity help catch load issues early

Infrastructure Summary
======================

The infrastructure provides a foundation for responsive, multi-threaded coordination:

1. **Event bus**: Asynchronous pub/sub with priority ordering, backpressure management, and thread-safe subscriptions
2. **Threading model**: Three threads (main, GUI event loop, audio) with clear responsibilities and safe cross-thread communication
3. **State management**: Locks protect shared state, atomic transitions prevent race conditions
4. **Service lifecycle**: Staged initialization with parallel setup reduces startup time; reverse-order shutdown ensures clean resource release
5. **Performance monitoring**: Event bus stats and slow handler detection help diagnose issues

This foundation enables Vocalance to perform real-time audio capture, concurrent speech recognition, LLM inference, and responsive UI updates without the complexities of traditional multi-threaded programming. The event-driven design means components are loosely coupled, testable independently, and easily extended with new functionality.
