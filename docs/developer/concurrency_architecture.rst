Concurrency & Asynchronous Architecture
#######################################

Vocalance employs a **Unified Hybrid Concurrency Model**. This architecture heavily favors asynchronous, sequential, and tightly coupled code on a single unified event loop, reserving background threads and asynchronous I/O strictly for operations where concurrency is explicitly justified.

This document explains exactly how our concurrency architecture is structured, why it is designed this way, and how data flows safely between different parts of the system.

The Golden Rule: Don't Block the Main Thread
============================================

The core of Vocalance's architecture is the unification of the Qt GUI event loop and Python's ``asyncio`` event loop onto the **Main Thread** using ``PySide6.QtAsyncio``.

Because both the UI and the asynchronous backend services share this single thread, they can interact seamlessly without complex thread marshaling or race conditions. However, this introduces a strict golden rule: **You must never block the Main Thread.**

If a task takes too long to execute synchronously on the Main Thread, the entire application freezes—the UI stops responding, animations stutter, and audio buffers overflow. We only use background threads when an operation would violate this rule.

The Asynchronous Event Bus
==========================

The ``EventBus`` is the central nervous system of Vocalance. It facilitates communication between decoupled components (e.g., Audio Listeners -> STT Service -> Dictation Coordinator -> UI Controllers).

To minimize overhead and complexity while preventing blocking, the EventBus operates as a **non-blocking, sequential dispatcher**.

When a component calls ``await event_bus.publish(event)``:

1. The bus immediately places the event into an ``asyncio.Queue`` and returns. It does **not** wait for subscribers to process the event.
2. A background worker task (``_process_queue``) continuously dequeues events one by one.
3. For each event, it executes synchronous handlers sequentially, and then executes all asynchronous handlers **concurrently** using ``asyncio.gather``.
4. The worker waits for all async handlers of the current event to finish before moving to the next event in the queue.

This design ensures highly predictable state transitions (events are processed in order) and eliminates race conditions between different events, while maximizing efficiency within a single event by running its async handlers concurrently. The entire subscriber dictionary and queue operations are protected by a highly efficient ``threading.Lock``, making it safe to subscribe or publish from any thread.

Why Background Threads Exist
============================

Because of Python's Global Interpreter Lock (GIL), pure Python code can only execute on one thread at a time. Multi-threading in pure Python does not yield true parallelism for CPU-bound tasks.

However, Vocalance uses background threads for heavy CPU-bound tasks like Speech-to-Text (Moonshine) and LLM generation (llama.cpp). Why? Because these tasks rely on **C/C++ Extensions**.

When a Python background thread calls into these highly optimized C++ libraries, the library immediately **releases the Python GIL**. This allows the C++ code to max out the CPU cores in true parallel, while the Python Main Thread remains completely free to keep the UI buttery smooth and capture audio.

We use background threads in specific, justified scenarios:

1. **Hardware Audio Capture (sounddevice)**: A dedicated C-level thread provided by PortAudio waits for the hardware buffer to fill every ~30ms.
2. **LLM Inference (llama.cpp)**: Offloaded to a thread pool to prevent the UI from freezing during heavy matrix math.
3. **Speech-to-Text (Vosk/Moonshine)**: Offloaded using ``asyncio.to_thread()`` or dedicated C++ threads to prevent blocking during transcription.
4. **File I/O and Automation (PyAutoGUI)**: Disk reads/writes and synchronous PyAutoGUI calls (which can block for 50-500ms) are offloaded via ``asyncio.to_thread()`` or a shared thread pool executor to prevent micro-stutters.

Cross-Thread Communication
==========================

Because the UI and the EventBus live on the Main Thread, background threads **cannot** safely touch them directly. Doing so would cause segmentation faults or state corruption.

To solve this, we use a bridge: ``loop.call_soon_threadsafe()``.

Whenever a background thread finishes a task (e.g., capturing an audio chunk or generating an LLM token), it packages that data into a function call and hands it to ``call_soon_threadsafe``. On its very next cycle, the Main Thread picks up this function and executes it safely within its own environment.

For events, background threads can safely call ``asyncio.run_coroutine_threadsafe(event_bus.publish(event), loop)`` to publish events from outside the main asyncio loop.

End-to-End Flows
================

The following diagrams illustrate how these concurrency principles apply to real-world use cases in Vocalance.

Command Recognition Flow
------------------------

In standard command mode, audio is buffered silently on the Main Thread until the user stops speaking. Only then is the complete audio segment sent to the background thread for processing.

.. mermaid::

   sequenceDiagram
       participant Mic as sounddevice C-Thread
       participant Main as Main Thread (Segmenter & Bus)
       participant STT as Vosk Background Thread
       participant UI as UI Controllers

       loop Every 30ms
           Mic->>Main: call_soon_threadsafe(audio_chunk)
           Note over Main: Segmenter buffers audio silently
       end
       Note over Main: User stops speaking (Silence detected)
       Main->>Main: Publish CommandAudioSegmentReadyEvent
       Main->>STT: asyncio.to_thread(process_audio)
       Note over STT: C++ engine releases GIL & processes
       STT-->>Main: publish(CommandTextRecognizedEvent)
       Main->>UI: Event Dispatch
       Note over UI: UI updates instantly

Dictation Flow
--------------

In dictation mode, audio chunks are streamed continuously to the background STT engine. The engine processes them and streams partial/final text back to the Main Thread.

.. mermaid::

   sequenceDiagram
       participant Mic as sounddevice C-Thread
       participant Main as Main Thread (Coordinator)
       participant STT as Moonshine Background Thread

       Main->>STT: Open Dictation Stream
       loop Continuous Streaming
           Mic->>Main: call_soon_threadsafe(audio_chunk)
           Main->>STT: Feed Chunk (C++ Buffer)
           Note over STT: Processes chunks asynchronously
           STT-->>Main: publish(PartialTextEvent)
           Note over Main: UI updates with partial text
       end
       Note over Main: Stop word detected or user stops
       Main->>STT: Stop Stream
       STT-->>Main: publish(FinalTextEvent)

LLM Smart Dictation Flow
------------------------

When using Smart Dictation, the finalized text from the STT engine is sent to the local LLM for grammar correction or transformation.

.. mermaid::

   sequenceDiagram
       participant Main as Main Thread (Coordinator)
       participant LLM as llama.cpp Background Thread
       participant UI as UI Controllers

       Note over Main: Dictation finishes, final text ready
       Main->>LLM: loop.run_in_executor(create_chat_completion)
       Note over LLM: C++ engine releases GIL & generates
       loop Token Generation
           LLM->>Main: call_soon_threadsafe(token_queue.put, token)
           Note over Main: Async generator yields token
           Main->>UI: Update UI with new token
       end
       LLM-->>Main: Completion
       Note over Main: Final transformed text pasted to active window
