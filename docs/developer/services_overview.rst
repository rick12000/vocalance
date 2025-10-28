Vocalance Services Overview
###########################

Introduction
============

Vocalance is a voice-controlled automation application that transforms spoken commands into keyboard and mouse actions.
This document provides a comprehensive overview of how the application works, from audio capture to command execution.

**Target Audience**: New developers joining the project who need to understand the system architecture and component interactions.

**Document Structure**: This guide follows the voice command processing pipeline sequentially, from microphone input through recognition and parsing to final execution. Supporting infrastructure (event bus, threading, storage) is covered at the end.

System Architecture Overview
==============================

Vocalance processes voice commands through a sequential pipeline from microphone input to screen action. This high-level flowchart shows the major components and decision points in the system:

.. mermaid::

   flowchart TD
       A[Microphone Input] --> B[AudioRecorder VAD]
       B --> C[Speech Detection]
       C --> D[AudioService]

       D --> E{Current Mode?}
       E -->|Command Mode| F[Command Recorder Active]
       E -->|Dictation Mode| G[Both Recorders Active]

       F --> H[CommandAudioSegmentReady Event]
       G --> I[DictationAudioSegmentReady Event]
       G --> J[CommandAudioSegmentReady Event<br/>for stop words]

       H --> K[SpeechToTextService]
       I --> K
       J --> K

       K --> L{Engine Selection}
       L -->|Command Audio| M[Vosk Engine]
       L -->|Dictation Audio| N[Whisper Engine]
       L -->|Empty Result| O[SoundService]

       M --> P[CommandTextRecognized Event]
       N --> Q[DictationTextRecognized Event]
       O --> R[CustomSoundRecognized Event]

       P --> S[CentralizedCommandParser]
       Q --> T[DictationCoordinator]
       R --> S

       S --> U{Command Type?}
       U -->|Automation| V[AutomationService]
       U -->|Mark| W[MarkService]
       U -->|Grid| X[GridService]
       U -->|Dictation| T

       V --> Y[pyautogui execution]
       W --> Z[Mouse jump]
       X --> AA[Grid UI display]
       T --> AB[Text output]


Audio Capture & Voice Activity Detection
==========================================

The foundation of Vocalance is its audio capture system, which continuously monitors microphone input and detects when speech occurs. This section covers the voice activity detection engine and the dual-recorder coordination system.

AudioRecorder: The Voice Detection Engine
-------------------------------------------

The ``AudioRecorder`` class is the entry point for all voice input. It implements Voice Activity Detection (VAD) to determine when speech begins and ends.

Core Responsibilities
~~~~~~~~~~~~~~~~~~~~~

- Continuously monitors microphone input via ``sounddevice``
- Detects speech onset using energy-based thresholding
- Captures complete utterances with pre-roll buffering
- Adapts to ambient noise using dynamic threshold adjustment
- Publishes audio segments via callbacks

VAD State Machine
~~~~~~~~~~~~~~~~~

The recorder implements a three-state machine:

.. mermaid::

   stateDiagram-v2
       [*] --> Waiting: recorder starts
       Waiting --> Recording: energy > energy_threshold
       Recording --> Waiting: silent_chunks >= silent_chunks_for_end
       Recording --> Waiting: duration >= max_duration
       Waiting --> Waiting: buffer pre-roll chunks<br/>update noise floor
       Recording --> Recording: collect audio chunks<br/>track silence

**State: Waiting**

- Reads audio chunks from microphone
- Calculates RMS energy per chunk
- Maintains pre-roll buffer (circular queue of recent chunks)
- Updates noise floor estimation with low-energy samples
- Transitions to Recording when ``energy > energy_threshold``

**State: Recording**

- Prepends pre-roll buffer to capture speech onset
- Continues collecting chunks
- Counts consecutive silent chunks (``energy < silence_threshold``)
- Ends when ``silent_chunks_count >= silent_chunks_for_end`` OR ``duration >= max_duration``
- Invokes ``on_audio_segment`` callback with complete audio bytes
- Returns to Waiting state

**Pre-Roll Mechanism**:

The pre-roll buffer solves the speech onset problem. By the time VAD detects "this is speech," the first syllable has already occurred. The recorder maintains a circular buffer (default 3-5 chunks) of recent audio. When speech is detected, these buffered chunks are prepended to the recording, capturing the full utterance.

Adaptive Noise Floor
~~~~~~~~~~~~~~~~~~~~

The recorder updates its noise floor estimate to handle varying acoustic environments:

.. code-block:: python

   # From AudioRecorder._recording_thread()
   if energy <= self.energy_threshold and len(self._noise_samples) < self._max_noise_samples:
       self._update_noise_floor(energy)

   # _update_noise_floor() method
   self._noise_samples.append(energy)
   if len(self._noise_samples) >= self.app_config.vad.max_noise_samples:
       self._noise_floor = np.percentile(self._noise_samples,
                                         self.app_config.vad.noise_floor_percentile)

When enough low-energy samples are collected, the noise floor is recalculated, and thresholds can be adapted if the environment is consistently noisy.

AudioService: Dual-Recorder Coordination
------------------------------------------

The ``AudioService`` coordinates two ``AudioRecorder`` instances running simultaneously in separate threads:

- ``_command_recorder``: Optimized for command mode (low-latency, short utterances)
- ``_dictation_recorder``: Optimized for dictation mode (longer duration, pause-tolerant)

Both recorders are always **running** (threads alive), but their VAD can be **active** or **inactive**:

- ``set_active(True)``: Recorder performs VAD and captures audio
- ``set_active(False)``: Recorder thread sleeps, no audio processing

This design allows instant mode switching without thread startup overhead.

Mode Switching Sequence
~~~~~~~~~~~~~~~~~~~~~~~~

This sequence diagram illustrates the complete event flow when switching between command and dictation modes. It shows how the system activates/deactivates recorders and handles the "stop dictation" trigger:

.. mermaid::

   sequenceDiagram
       participant User
       participant CmdRec as Command Recorder
       participant DictRec as Dictation Recorder
       participant EventBus
       participant STT as STT Service
       participant Parser as Command Parser

       Note over CmdRec,DictRec: Initial: Command Mode
       Note over CmdRec: VAD Active
       Note over DictRec: VAD Inactive

       User->>CmdRec: Says "dictate"
       CmdRec->>EventBus: CommandAudioSegmentReady
       EventBus->>STT: Process audio
       STT->>EventBus: CommandTextRecognized("dictate")
       EventBus->>Parser: Parse command
       Parser->>EventBus: DictationStartCommand
       EventBus->>CmdRec: AudioModeChangeRequest(mode="dictation")
       EventBus->>DictRec: AudioModeChangeRequest(mode="dictation")

       Note over CmdRec,DictRec: Now: Dictation Mode
       Note over CmdRec: VAD Active (stop words)
       Note over DictRec: VAD Active (full capture)

       User->>DictRec: Says "Hello world..."
       DictRec->>EventBus: DictationAudioSegmentReady

       User->>CmdRec: Says "stop dictation"
       CmdRec->>EventBus: CommandAudioSegmentReady
       EventBus->>STT: Process audio
       STT->>STT: _is_stop_trigger() returns True
       STT->>EventBus: CommandTextRecognized("stop dictation")
       EventBus->>Parser: Parse command
       Parser->>EventBus: DictationStopCommand
       EventBus->>CmdRec: AudioModeChangeRequest(mode="command")
       EventBus->>DictRec: AudioModeChangeRequest(mode="command")

       Note over CmdRec,DictRec: Back to: Command Mode


Speech-to-Text Processing
===========================

Once audio is captured, it must be converted to text. Vocalance uses a dual-engine approach: a fast Vosk engine for commands and an accurate Whisper engine for dictation. This section also covers auxiliary recognition systems for non-speech sounds and predictive execution.

SpeechToTextService: Dual-Engine Recognition
----------------------------------------------

The ``SpeechToTextService`` converts audio bytes into text using two specialized STT engines:

**Vosk Engine (``VoskSTT``)** (used for command audio):

- Model: Lightweight Kaldi-based model (vosk-model-small-en-us)
- Latency: Fast recognition suitable for real-time commands
- Accuracy: Good for command vocabulary, less accurate for natural prose
- Execution: Runs efficiently on CPU
- Use Case: Command mode where speed matters

**Whisper Engine (``WhisperSTT``)** (used for dictation audio):

- Model: OpenAI Whisper (configurable: tiny/base/small/medium, default: base)
- Latency: Slower than Vosk, varies with audio length and model size
- Accuracy: Excellent, includes automatic punctuation
- Execution: CPU or CUDA (configurable via ``whisper_device`` in STTConfig)
- Use Case: Dictation mode where accuracy matters

Audio Processing Flow
~~~~~~~~~~~~~~~~~~~~~

The STT service has different processing paths depending on the current mode and audio type. This flowchart shows how the service routes audio events to the appropriate engine and handles mode-specific logic:

.. mermaid::

   flowchart TD
       A[Audio Events] --> B{Event Type?}
       B -->|CommandAudioSegmentReady| C[_handle_command_audio_segment]
       B -->|DictationAudioSegmentReady| D[_handle_dictation_audio_segment]

       C --> E{dictation_active?}
       E -->|True| F[Vosk: Check stop trigger only]
       E -->|False| G[Vosk: Full recognition]

       F --> H{Is stop trigger?}
       H -->|Yes| I[Publish CommandTextRecognized]
       H -->|No| J[Ignore silently]

       G --> K{Has text?}
       K -->|Yes| L[Check duplicate filter]
       K -->|No| M[Publish ProcessAudioChunkForSound]

       L --> N{Is duplicate?}
       N -->|No| I
       N -->|Yes| J

       D --> O[Whisper: Recognize dictation]
       O --> P{Has text?}
       P -->|Yes| Q[Check duplicate filter]
       P -->|No| R[End quietly]

       Q --> S{Is duplicate?}
       S -->|No| T[Publish DictationTextRecognized]
       S -->|Yes| R


The service uses a ``DuplicateTextFilter`` to prevent duplicate recognition results (sometimes STT engines return the same result twice):

- Cache size: 5 recent texts
- Duplicate threshold: 1000ms
- If same text seen within threshold, marked as duplicate and ignored

Sound Recognition for Non-Speech Audio
----------------------------------------

When Vosk returns empty (no speech detected), the ``SpeechToTextService`` publishes a ``ProcessAudioChunkForSoundRecognitionEvent``. The ``SoundService`` handles non-speech audio like finger snaps, whistles, or clicks.

YAMNet-Based Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The service uses Google's pre-trained YAMNet model to convert audio into a 1024-dimensional embedding vector. During **training**, the user records multiple samples of a custom sound (e.g., finger snap), and the service collects these embeddings to train a KNN classifier. The classifier also includes pre-trained embeddings from the ESC-50 dataset (keyboard typing, mouse clicks, coughing, etc.) which serve as **negative labels** - they help the classifier distinguish what is NOT the target sound.

At **prediction time**, when audio arrives, YAMNet generates an embedding and the KNN classifier finds the closest match. If the match is a custom sound (not one of the ESC-50 negative examples), the service looks up its command mapping and publishes a ``CustomSoundRecognizedEvent`` with the corresponding command text. The parser then treats this like any other recognized text.

Training and Recognition Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This flowchart shows two operational modes: training (collecting samples to build the classifier) and recognition (identifying sounds in real-time):

.. mermaid::

   flowchart TD
       A[ProcessAudioChunkForSound Event] --> B{training_active?}
       B -->|Yes| C[Collect YAMNet embedding]
       B -->|No| D[recognizer.recognize_sound]

       C --> E[Append to training samples]
       E --> F{Enough samples?}
       F -->|Yes| G[Train KNN classifier]
       F -->|No| H[Wait for more]

       D --> I{Result found?}
       I -->|Yes| J{Is custom sound?}
       I -->|No| K[Ignore]

       J -->|Yes| L[Get command mapping]
       J -->|No esc50_*| K

       L --> M[Publish CustomSoundRecognized]

       G --> N[Save trained model]
       N --> O[Publish TrainingComplete]

Sound-to-Command Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~

After training, users can map sounds to command phrases in the UI. For example:

- ``finger_snap`` → ``"click"``
- ``whistle`` → ``"scroll down"``
- ``tongue_click`` → ``"press enter"``

When a sound is recognized, its mapped command text is published as a ``CustomSoundRecognizedEvent``. The ``CentralizedCommandParser`` treats this like STT text and parses it into a command.

Predictive Execution with Markov Chains
-----------------------------------------

The ``MarkovCommandService`` predicts and executes commands before STT completes, reducing perceived latency to near-zero for repetitive workflows.

Zero-Latency Command Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Problem**: Even Vosk's fast recognition takes 50-200ms + silence timeout of 180ms. For rapid command sequences, this latency is noticeable.

**The Solution**: When audio is *detected* (not yet recognized), predict the most likely command based on recent history and execute immediately.

How Prediction Works
~~~~~~~~~~~~~~~~~~~~~

The command recorder publishes ``AudioDetectedEvent`` as soon as speech energy exceeds the threshold (~5-10ms into speech). This triggers the predictor before any STT processing:

.. mermaid::

   sequenceDiagram
       participant User
       participant Recorder as Command Recorder
       participant Markov as MarkovCommandService
       participant STT as STT Service
       participant Parser as Command Parser

       User->>Recorder: Starts saying "scroll down"
       Note over Recorder: Energy > threshold (5-10ms)
       Recorder->>Markov: AudioDetectedEvent

       Note over Markov: Check command history<br/>Last 3 commands: [scroll down, scroll down, scroll down]
       Markov->>Markov: 4th-order prediction: "scroll down" (confidence: 0.95)
       Markov->>Parser: MarkovPredictionEvent(predicted="scroll down")
       Parser->>Parser: Execute immediately

       Note over STT: Still processing... (50ms later)
       Recorder->>STT: CommandAudioSegmentReady
       STT->>Parser: CommandTextRecognized("scroll down")
       Parser->>Parser: Matches prediction - skip duplicate
       Parser->>Markov: MarkovPredictionFeedbackEvent(correct=True)

The service trains multi-order Markov chains (2nd, 3rd, 4th order) on command history:

- **2nd order**: Predicts based on last 1 command
- **3rd order**: Predicts based on last 2 commands
- **4th order**: Predicts based on last 3 commands

**Backoff Strategy**: Try highest order first (more specific context), fall back to lower orders if no confident match.

Configuration parameters from ``MarkovPredictorConfig``:

.. code-block:: python

   confidence_threshold: 1.0  # Minimum probability to execute prediction
   training_window_commands: {2: 500, 3: 1000, 4: 1500}  # Commands per order
   training_window_days: {2: 7, 3: 21, 4: 60}  # Days of history per order
   min_command_frequency: {2: 15, 3: 10, 4: 10}  # Min occurrences to trust pattern

Error Handling and Safeguards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a prediction is wrong, a cooldown is applied:

.. code-block:: python

   # From MarkovCommandService
   if self._cooldown_remaining > 0:
       logger.debug(f"Skipping Markov (cooldown: {self._cooldown_remaining} commands remaining)")
       return

   # After wrong prediction
   self._cooldown_remaining = self._markov_config.incorrect_prediction_cooldown  # default: 2

This prevents repeated mispredictions from distributional shifts (eg. user changes pattern, or website layout changes, etc...)

The predictor also disables itself during dictation mode:

.. code-block:: python

   # From MarkovCommandService._handle_audio_detected_fast_track()
   if self._dictation_active:
       logger.debug("Skipping Markov prediction - dictation mode active")
       return

**Why?** Without this, the following would occur:

1. User says "dictate" → Markov adds "dictate" to history
2. User starts dictating → Audio detected
3. Markov sees "dictate" was last command → predicts "stop dictation"
4. Dictation ends immediately before user speaks!

The service subscribes to ``DictationModeDisableOthersEvent`` to track dictation state.


Command Parsing
================

After text is recognized (via STT, sound recognition, or Markov prediction), it must be converted into structured commands that execution services can act upon. The command parser also routes dictation text to the dictation system.

CentralizedCommandParser: Text to Structured Commands
-------------------------------------------------------

The ``CentralizedCommandParser`` receives text from three sources (STT, sound recognition, Markov predictions) and converts it into structured command objects.

The parser subscribes to three event types:

1. ``CommandTextRecognizedEvent``: Text from Vosk STT
2. ``DictationTextRecognizedEvent``: Text from Whisper STT (forwarded to dictation coordinator)
3. ``CustomSoundRecognizedEvent``: Mapped command text from sound recognition
4. ``MarkovPredictionEvent``: Predicted command text from Markov service

Deduplication Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~

The parser prevents double-execution through two layers:

**Text Deduplication**

.. code-block:: python

   # From _process_text_input_with_history()
   current_time = time.time()
   if text == self._last_text and current_time - self._last_text_time < self._duplicate_interval:
       logger.debug(f"Suppressing duplicate text: '{text}'")
       return

   self._last_text = text
   self._last_text_time = current_time

Prevents identical text within 1 second from being processed twice (default ``_duplicate_interval``).

**Markov-STT Deduplication**

When Markov predicts and executes a command, it's added to ``_recent_predictions`` dict. When the STT result arrives:

.. code-block:: python

   # From _handle_command_text_recognized()
   matched_prediction = self._check_prediction_match(text, timestamp)

   if matched_prediction:
       # Markov already executed this - send feedback but don't re-execute
       await self._send_markov_feedback(
           predicted_command=text,
           actual_command=text,
           was_correct=True,
           source="stt"
       )
       return

Hierarchical Parsing
~~~~~~~~~~~~~~~~~~~~

The parser tries command types in strict priority order. This ensures that more specific command types (like dictation triggers) are matched before falling through to general automation commands:

.. mermaid::

   flowchart TD
       A[Text Input] --> B[Normalize: lowercase, strip]
       B --> C{Empty?}
       C -->|Yes| D[NoMatchResult]
       C -->|No| E[Try: Dictation Commands]

       E --> F{Match?}
       F -->|Yes| G[Return DictationCommand]
       F -->|No| H[Try: Mark Commands]

       H --> I{Match?}
       I -->|Yes| J[Return MarkCommand]
       I -->|No| K[Try: Grid Commands]

       K --> L{Match?}
       L -->|Yes| M[Return GridCommand]
       L -->|No| N[Try: Automation Commands]

       N --> O{Match?}
       O -->|Yes| P[Return AutomationCommand]
       O -->|No| Q[Try: Mark Execute Fallback]

       Q --> R{Match?}
       R -->|Yes| S[Return MarkExecuteCommand]
       R -->|No| D


Command History Recording
~~~~~~~~~~~~~~~~~~~~~~~~~~

Only valid commands are recorded to history (for Markov training):

.. code-block:: python

   if isinstance(parse_result, BaseCommand):
       # Valid command - record to history and execute
       await self._history_manager.record_command(command=text, source=source)
       await self._publish_command_event(parse_result, source)
   elif isinstance(parse_result, NoMatchResult):
       # No match - don't record
       await self._event_bus.publish(CommandNoMatchEvent(...))


Dictation System
=================

Dictation is a core feature that converts continuous speech into typed text, with three distinct modes for different use cases. When the Command Parser identifies dictation commands ("dictate", "type", "smart dictate"), it routes them to the DictationCoordinator, which represents an alternative execution path parallel to automation commands.

DictationCoordinator: The Orchestrator
---------------------------------------

The ``DictationCoordinator`` is a sophisticated state machine that manages all dictation workflows, integrating STT, LLM processing, and text input.

Dictation Modes
~~~~~~~~~~~~~~~

The system supports three dictation modes, each with different behavior:

**1. STANDARD Mode** (trigger: "dictate"):

- Direct transcription without LLM processing
- Text is typed immediately as Whisper recognizes each segment
- Best for: Quick notes, continuous typing without editing

**2. TYPE Mode** (trigger: "type"):

- Single-phrase input with auto-stop
- Monitors silence - automatically stops after first silence is detected
- Best for: Short inputs like filenames, single sentences, google searches, etc...

**3. SMART Mode** (trigger: "smart dictate"):

- Accumulates all spoken text during session
- When stopped, sends accumulated text to LLM for enhancement
- LLM improves grammar, clarity, formatting per user-defined prompt
- Displays preview text in real-time, types final enhanced version
- Best for: Emails, documents, code - anything requiring polish

Dictation State Machine
~~~~~~~~~~~~~~~~~~~~~~~~

The coordinator implements a strict state machine with validated transitions. This diagram shows the four possible states and their allowed transitions:

.. mermaid::

   stateDiagram-v2
       [*] --> IDLE: Coordinator starts
       IDLE --> RECORDING: Start command
       RECORDING --> PROCESSING: Smart mode stop
       RECORDING --> IDLE: Standard/Type stop
       PROCESSING --> IDLE: LLM complete/failed
       IDLE --> SHUTDOWN: Shutdown
       RECORDING --> SHUTDOWN: Shutdown
       PROCESSING --> SHUTDOWN: Shutdown

       note right of IDLE
           No active session
           Ready to start
       end note

       note right of RECORDING
           Capturing speech
           Accumulating text
       end note

       note right of PROCESSING
           Smart mode only
           LLM enhancing text
       end note

**States**:

- ``IDLE``: No active session, ready to start
- ``RECORDING``: Actively capturing and processing speech
- ``PROCESSING_LLM``: Smart mode only - processing accumulated text through LLM
- ``SHUTTING_DOWN``: Coordinator shutting down

**Valid transitions** are enforced at runtime to prevent invalid state changes:

.. code-block:: python

   _VALID_TRANSITIONS = {
       DictationState.IDLE: {DictationState.RECORDING, DictationState.SHUTTING_DOWN},
       DictationState.RECORDING: {DictationState.PROCESSING_LLM, DictationState.IDLE, DictationState.SHUTTING_DOWN},
       DictationState.PROCESSING_LLM: {DictationState.IDLE, DictationState.SHUTTING_DOWN},
       DictationState.SHUTTING_DOWN: set(),
   }

Complete Event Flow
~~~~~~~~~~~~~~~~~~~

These sequence diagrams show the complete event flow for each dictation mode. Notice how events coordinate across multiple services.

**Standard/Type Mode Flow**:

This diagram shows the simpler flow where text is typed immediately as it's recognized:

.. mermaid::

   sequenceDiagram
       participant User
       participant Parser as Command Parser
       participant Coord as DictationCoordinator
       participant Audio as AudioService
       participant STT as STT Service
       participant TextInput as TextInputService

       User->>Parser: Says "dictate" or "type"
       Parser->>Coord: DictationStartCommand/TypeCommand
       Coord->>Coord: Create session (STANDARD/TYPE mode)
       Coord->>Coord: State: IDLE → RECORDING
       Coord->>Audio: AudioModeChangeRequest(mode="dictation")
       Coord->>STT: DictationModeDisableOthersEvent(active=True)

       Note over Audio,STT: Both recorders active<br/>STT in dictation mode

       User->>Audio: Speaks "Hello world..."
       Audio->>STT: DictationAudioSegmentReady
       STT->>Coord: DictationTextRecognized("Hello world")
       Coord->>Coord: Clean text, apply formatting
       Coord->>TextInput: input_text("Hello world")
       TextInput->>TextInput: pyautogui.write()

       User->>Parser: Says "stop dictation"
       Parser->>Coord: DictationStopCommand
       Coord->>Coord: State: RECORDING → IDLE
       Coord->>Audio: AudioModeChangeRequest(mode="command")
       Coord->>STT: DictationModeDisableOthersEvent(active=False)

**Smart Mode Flow** (with LLM processing):

This diagram shows the more complex flow where text is accumulated, then enhanced by LLM before typing:

.. mermaid::

   sequenceDiagram
       participant User
       participant Coord as DictationCoordinator
       participant Audio as AudioService
       participant STT as STT Service
       participant UI as Smart Dictation UI
       participant LLM as LLMService
       participant TextInput as TextInputService

       User->>Coord: Says "smart dictate"
       Coord->>Coord: Create session (SMART mode)
       Coord->>Coord: State: IDLE → RECORDING
       Coord->>Audio: AudioModeChangeRequest(mode="dictation")
       Coord->>UI: SmartDictationStartedEvent

       loop Multiple utterances
           User->>STT: Speaks segment
           STT->>Coord: DictationTextRecognized
           Coord->>Coord: Accumulate text
           Coord->>UI: SmartDictationTextDisplayEvent (preview)
       end

       User->>Coord: Says "stop dictation"
       Coord->>Coord: State: RECORDING → PROCESSING_LLM
       Coord->>Audio: AudioModeChangeRequest(mode="command")
       Coord->>UI: SmartDictationStoppedEvent
       Coord->>UI: LLMProcessingStartedEvent

       UI->>Coord: LLMProcessingReadyEvent (UI ready)
       Coord->>LLM: process_dictation_streaming()

       loop Token streaming
           LLM->>Coord: _stream_token(token)
           Coord->>UI: LLMTokenGeneratedEvent (real-time)
       end

       LLM->>Coord: LLMProcessingCompletedEvent
       Coord->>TextInput: input_text(enhanced_text)
       Coord->>Coord: State: PROCESSING_LLM → IDLE
       Coord->>STT: DictationModeDisableOthersEvent(active=False)


Command Execution
==================

The final stage of the pipeline executes the parsed commands by performing keyboard input, mouse clicks, or other automation actions.

AutomationService: Keyboard & Mouse Control
--------------------------------------------

The ``AutomationService`` executes automation commands using ``pyautogui`` for keyboard and mouse control.

Event Subscription
~~~~~~~~~~~~~~~~~~

The service subscribes to ``AutomationCommandParsedEvent``, which contains a structured command object with:

- ``command_key``: Unique identifier (e.g., "click", "scroll_down")
- ``action_type``: ActionType enum (hotkey, key, click, scroll, move, etc.)
- ``action_value``: Parameter for the action (e.g., "enter", "ctrl+c", "10")
- ``count``: Number of times to repeat (default 1)

Execution Flow
~~~~~~~~~~~~~~

This flowchart shows the complete execution pipeline from event reception to action execution. The service validates the command, checks cooldowns, creates the appropriate pyautogui action, and executes it in a thread pool:

.. mermaid::

   flowchart TD
       A[AutomationCommandParsed Event] --> B{count > 0?}
       B -->|No| C[Publish error status]
       B -->|Yes| D[Check cooldown]

       D --> E{On cooldown?}
       E -->|Yes| F[Publish cooldown status]
       E -->|No| G[Create action function]

       G --> H{action_type?}
       H -->|hotkey| I[lambda: pyautogui.hotkey]
       H -->|key| J[lambda: pyautogui.press]
       H -->|click| K[lambda: pyautogui.click]
       H -->|scroll| L[lambda: pyautogui.scroll]
       H -->|move| M[lambda: pyautogui.move]

       I --> N[Execute in thread pool]
       J --> N
       K --> N
       L --> N
       M --> N

       N --> O[Update cooldown timer]
       O --> P[Publish success status]

Action Function Creation
~~~~~~~~~~~~~~~~~~~~~~~~

The service maps action types to ``pyautogui`` calls:

.. code-block:: python

   # From AutomationService._create_action_function()
   if action_type == "hotkey":
       keys = [key.strip() for key in action_value.replace(" ", "+").split("+")]
       return lambda: pyautogui.hotkey(*keys)

   elif action_type == "key":
       return lambda: pyautogui.press(action_value)

   elif action_type == "click":
       return lambda: pyautogui.click()

   elif action_type == "scroll":
       amount = int(action_value)
       return lambda: pyautogui.scroll(amount)

Thread Pool Execution
~~~~~~~~~~~~~~~~~~~~~

PyAutoGUI is synchronous and blocks during execution. To avoid blocking the async event loop:

.. code-block:: python

   # From _execute_command()
   async with self._execution_lock:
   loop = asyncio.get_running_loop()
   return await loop.run_in_executor(
           self._thread_pool,  # ThreadPoolExecutor(max_workers=2)
       lambda: self._execute_action(action_function, count)
   )

The ``_execution_lock`` ensures only one automation command executes at a time.


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
