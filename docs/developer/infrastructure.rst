Infrastructure & Supporting Systems
=====================================

The preceding sections covered the voice command processing pipeline. This section describes the infrastructure that enables inter-service communication, manages concurrency, and provides data persistence.

Event Bus
----------

All Vocalance services communicate via the ``EventBus``, a priority queue-based message broker that enables loose coupling between components.

Architecture and Design
~~~~~~~~~~~~~~~~~~~~~~~~

The EventBus acts as a central message broker. Publishers can be on any thread, but all subscribers run in the GUI event loop thread. This diagram shows the publish-subscribe architecture:

.. mermaid::

   flowchart LR
       subgraph Publishers["Publishers (Any Thread)"]
           P1[Audio Recorder Thread]
           P2[STT Service]
           P3[Command Parser]
       end

       subgraph EventBus
           Q[asyncio.PriorityQueue]
           W[Worker Task]
       end

       subgraph Subscribers["Subscribers (GUI Loop)"]
           S1[STT Service]
           S2[Command Parser]
           S3[Automation Service]
       end

       P1 -->|publish| Q
       P2 -->|publish| Q
       P3 -->|publish| Q

       Q --> W
       W -->|dispatch| S1
       W -->|dispatch| S2
       W -->|dispatch| S3

**Core Characteristics**:

- **Priority-based**: Events sorted by priority value (CRITICAL=0, HIGH=10, NORMAL=20, LOW=30)
- **Thread-safe publish**: Any thread can publish via ``asyncio.run_coroutine_threadsafe()``
- **Single-threaded consume**: Worker task runs in GUI event loop
- **Asynchronous dispatch**: Subscribers invoked via ``await`` or direct call

Publishing Events
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # From any thread
   async def publish(self, event: BaseEvent) -> None:
       await self._event_queue.put((event.priority, next(self._counter), event))

The ``_counter`` ensures stable sorting (FIFO within same priority).

Subscribing to Events
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def subscribe(self, event_type: Type[BaseEvent], handler: Callable) -> None:
       with self._subscribers_lock:
           self._subscribers[event_type].append(handler)

Subscribers are matched via ``isinstance()`` check, allowing inheritance-based subscriptions.

Threading Model
----------------

Vocalance uses multiple threads to prevent blocking and maintain responsiveness.

Thread Overview
~~~~~~~~~~~~~~~

Vocalance runs multiple threads concurrently. This diagram shows the thread architecture and how threads communicate:

.. mermaid::

   flowchart TD
       Main[Main Thread<br/>Tkinter mainloop]
       GUI[GUI Event Loop Thread<br/>asyncio event loop<br/>Event bus worker]

       CMD[Command Recorder Thread]
       DICT[Dictation Recorder Thread]

       STORAGE[Storage Pool<br/>File I/O]
       AUTO[Automation Pool<br/>PyAutoGUI]

       CMD -->|publish events| GUI
       DICT -->|publish events| GUI
       GUI -->|run_in_executor| STORAGE
       GUI -->|run_in_executor| AUTO
       GUI -->|call_soon_threadsafe| Main

Thread Responsibilities
~~~~~~~~~~~~~~~~~~~~~~~

**Main Thread**:

- Runs ``app_tk_root.mainloop()`` (Tkinter)
- Processes UI events and renders widgets
- Must never block - delegates all heavy work

**GUI Event Loop Thread**:

Created at startup:

.. code-block:: python

   # From main.py _setup_infrastructure()
   gui_event_loop = asyncio.new_event_loop()
   gui_thread = threading.Thread(
       target=lambda: (asyncio.set_event_loop(gui_event_loop),
                      gui_event_loop.run_forever()),
       daemon=False,
       name="GUIEventLoop"
   )
   gui_thread.start()

Runs the event bus worker and all service async methods. Separate from main thread because Tkinter's mainloop can't run inside an asyncio event loop.

**Audio Recorder Threads**:

Two dedicated threads run ``AudioRecorder._recording_thread()``:

- ``AudioRecorder_command``: Command mode VAD
- ``AudioRecorder_dictation``: Dictation mode VAD

Thread lifecycle:

.. code-block:: python

   self._thread = threading.Thread(
       target=self._recording_thread,
       daemon=False,
       name=f"AudioRecorder_{self.mode}"
   )
   self._thread.start()

Thread safety via ``threading.Lock`` for state flags (``_is_active``, ``_is_recording``).

**Thread Pools**:

.. code-block:: python

   # Storage (file I/O)
   ThreadPoolExecutor(max_workers=2, thread_name_prefix="Storage")

   # Automation (PyAutoGUI)
   ThreadPoolExecutor(max_workers=app_config.automation_service.thread_pool_max_workers)

Used via ``loop.run_in_executor()`` to avoid blocking the async event loop.

Cross-Thread Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**From recorder thread to GUI event loop**:

.. code-block:: python

       asyncio.run_coroutine_threadsafe(
           self._event_bus.publish(event_data),
           self._main_event_loop  # GUI event loop
       )

Asynchronous Programming Patterns
-----------------------------------

Vocalance uses ``async/await`` to manage I/O-bound operations without blocking. Understanding these patterns is critical for maintaining system responsiveness and preventing deadlocks.

The Hybrid Pattern
~~~~~~~~~~~~~~~~~~~

Most services use an "async shell, sync core" pattern to offload blocking work. The async wrapper provides a non-blocking interface, while the sync core does the actual work in a thread pool:

.. code-block:: python

   # From STT engine
   async def recognize(self, audio_bytes, sample_rate):
       """Async interface for callers"""
       loop = asyncio.get_running_loop()
       return await loop.run_in_executor(
           self._thread_pool,
           self._recognize_sync,  # Sync method
           audio_bytes,
           sample_rate
       )

   def _recognize_sync(self, audio_bytes, sample_rate):
       """Sync implementation - blocks during Vosk processing"""
       result = self.recognizer.AcceptWaveform(audio_bytes)
       return json.loads(self.recognizer.Result())["text"]

This keeps the event loop responsive while CPU-bound work runs in a thread pool.

Best Practices
~~~~~~~~~~~~~~~

**Rule 1**: Never block the event loop with CPU-bound or I/O operations

.. code-block:: python

   # BAD
   async def process(self):
       data = json.load(open("file.json"))  # Blocks!

   # GOOD
   async def process(self):
       loop = asyncio.get_event_loop()
       data = await loop.run_in_executor(self._pool, self._read_file)

**Rule 2**: Use ``run_coroutine_threadsafe`` from non-async threads

.. code-block:: python

   # From audio recorder thread
   def on_audio_captured(self, audio_bytes):
       asyncio.run_coroutine_threadsafe(
           self._event_bus.publish(event),
           self._gui_event_loop
       )

Data Persistence
-----------------

The ``StorageService`` provides crash-safe, thread-safe file persistence for application data.

**Key Features**:

- **Type-safe**: All data validated via Pydantic models
- **Atomic writes**: Temp file + rename pattern prevents corruption
- **Cached**: TTL-based in-memory cache reduces disk I/O
- **Async**: File operations run in thread pool
- **Thread-safe**: RLock protects cache access

Atomic Write Pattern
~~~~~~~~~~~~~~~~~~~~~

Writes are atomic at the OS level to prevent corruption. The pattern uses a temp file + atomic rename to ensure that either the complete old file or complete new file exists, never a partially-written file:

.. code-block:: python

   # From StorageService._write_json()
   def _write_json(self, path: Path, data: Dict) -> bool:
       # Write to temp file
       temp_path = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
       with open(temp_path, "w", encoding="utf-8") as f:
           json.dump(data, f, indent=2)

       # Atomic rename
       if path.exists():
           backup_path = path.with_suffix(".backup")
           os.replace(path, backup_path)  # Backup existing
           try:
               os.replace(temp_path, path)  # Atomic replace
               os.remove(backup_path)
           except Exception:
               os.replace(backup_path, path)  # Restore on error
               raise
       else:
           os.replace(temp_path, path)

If the process crashes mid-write, either the old file or new file exists (never corrupted).

Caching Strategy
~~~~~~~~~~~~~~~~~

The service caches read data with TTL expiration:

.. code-block:: python

   # From StorageService.read()
   with self._lock:
       if cache_key in self._cache:
           entry = self._cache[cache_key]
           if not entry.is_expired(self._cache_ttl):  # Default: 300s
               return entry.data

Writes update the cache immediately (write-through caching).
