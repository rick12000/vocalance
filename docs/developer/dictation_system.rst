Dictation System
##################

This page explains how Vocalance handles dictation through five distinct modes—standard, visual, smart, type, and hidden—each optimized for different use cases, all orchestrated by the DictationCoordinator.

System Overview
================

The dictation system operates independently from command execution. Once activated, it accumulates speech input and processes it through mode-specific pipelines before outputting text. Unlike commands which execute immediately, dictation involves longer processing with explicit start and stop boundaries.

.. mermaid::

   flowchart TD
       A[DictationCommandParsedEvent] --> B{Command Type}
       B -->|start dictation| C[Standard Mode]
       B -->|dictate visual| D[Visual Mode]
       B -->|dictate smart| E[Smart Mode]
       B -->|dictate type| F[Type Mode]
       B -->|hidden| G[Hidden Mode]
       B -->|stop dictation| H[Stop & Finalize]

       C --> I[DictationTextRecognizedEvent]
       D --> I
       E --> I
       F --> I
       G --> I

       I --> J{Mode?}
       J -->|Standard| K[Direct Type]
       J -->|Visual| L[Popup Accumulate]
       J -->|Smart| M[LLM Queue]
       J -->|Type| N[Raw Type]
       J -->|Hidden| O[Silent Accumulate]

       M --> P[LLM Processing]
       P --> Q[Output]
       K --> Q
       L --> R[Accumulated Display]
       N --> Q
       O --> S[Paste on Stop]

       style C fill:#e8f5e9
       style D fill:#fff4e1
       style E fill:#e1f5ff
       style F fill:#fce4ec
       style G fill:#e8e8e8

Dictation flows from parse event → mode selection → text recognition → mode-specific processing → output. Each mode has distinct behavior: standard types immediately, visual accumulates for review, smart uses LLM for formatting, type provides raw unformatted insertion, and hidden silently accumulates for pasting.

The DictationCoordinator: Central Orchestration
================================================

The ``DictationCoordinator`` manages dictation through a strict state machine, mode-specific handlers, and lifecycle management.

State Machine
-------------

The coordinator uses three core states with validated transitions:

.. mermaid::

   stateDiagram-v2
       [*] --> IDLE
       IDLE --> RECORDING: Start dictation command
       RECORDING --> RECORDING: Accumulate text segments
       RECORDING --> PROCESSING_LLM: Stop smart dictation
       RECORDING --> IDLE: Stop standard/visual/type
       PROCESSING_LLM --> IDLE: LLM complete
       IDLE --> [*]

       note right of RECORDING
           Accumulating text from
           DictationTextRecognizedEvent
       end note

       note right of PROCESSING_LLM
           Only for smart mode
           LLM processing accumulated text
       end note

**State validation**: The coordinator enforces valid transitions. Invalid transitions are logged as errors and rejected:

.. code-block:: python

   _VALID_TRANSITIONS = {
       DictationState.IDLE: {DictationState.RECORDING, DictationState.SHUTTING_DOWN},
       DictationState.RECORDING: {DictationState.PROCESSING_LLM, DictationState.IDLE, DictationState.SHUTTING_DOWN},
       DictationState.PROCESSING_LLM: {DictationState.IDLE, DictationState.SHUTTING_DOWN},
       DictationState.SHUTTING_DOWN: set(),
   }

   async def _transition_to(self, new_state: DictationState):
       if new_state not in _VALID_TRANSITIONS[self._current_state]:
           logger.error(f"Invalid transition: {self._current_state} → {new_state}")
           return False

       self._current_state = new_state
       return True

This prevents race conditions where text arrives after stop is called, or multiple stop commands conflict.

Mode Selection and Activation
------------------------------

Dictation modes are triggered by voice commands parsed by the centralized parser:

- **"start dictation"** → Standard mode (immediate typing)
- **"dictate visual"** → Visual mode (popup accumulation)
- **"dictate smart"** → Smart mode (LLM formatting)
- **"dictate type"** → Type mode (raw insertion)
- **"stop dictation"** → Stop active mode and finalize

The coordinator subscribes to ``DictationCommandParsedEvent`` and routes commands:

.. code-block:: python

   async def _handle_dictation_command(self, event: DictationCommandParsedEvent):
       command = event.command

       if isinstance(command, DictationStartCommand):
           await self._start_session(DictationMode.STANDARD)
       elif isinstance(command, DictationVisualStartCommand):
           await self._start_session(DictationMode.VISUAL)
       elif isinstance(command, DictationSmartStartCommand):
           await self._start_session(DictationMode.SMART)
       elif isinstance(command, DictationTypeCommand):
           await self._start_session(DictationMode.TYPE)
       elif isinstance(command, DictationStopCommand):
           await self._stop_session()

System Awareness During Dictation
----------------------------------

When dictation starts, the coordinator broadcasts ``DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode=...)`` which:

- **Disables Markov predictions** (prevents false command predictions)
- **Disables sound recognition** (prevents sound-mapped commands)
- **Filters speech recognition** to only recognize stop trigger words

When dictation stops, ``DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive")`` re-enables all systems.

Standard Dictation Mode
========================

Standard mode provides immediate text output with minimal processing, designed for fast entry.

How It Works
------------

.. mermaid::

   sequenceDiagram
       participant U as User
       participant Coord as DictationCoordinator
       participant STT as SpeechToText
       participant Input as TextInputService

       U->>Coord: "start dictation"
       Coord->>Coord: Enter RECORDING state

       U->>STT: Speak: "Hello world"
       STT->>Coord: DictationTextRecognizedEvent("Hello world")
       Coord->>Input: Type "Hello world"

       U->>STT: Speak: "This is a test"
       STT->>Coord: DictationTextRecognizedEvent("This is a test")
       Coord->>Input: Type " This is a test"

       U->>Coord: "stop dictation"
       Coord->>Coord: Enter IDLE state

**Processing pipeline**:

1. **Receive text**: From ``DictationTextRecognizedEvent`` (speech-to-text engine)
2. **Clean text**: Remove artifacts like "..." and normalize spacing
3. **Handle spacing**: Add space before segment (except first)
4. **Apply formatting rules**: Lowercase first letter if mid-sentence, remove period if needed
5. **Type text**: Insert via clipboard or direct typing
6. **Continue**: Remain in RECORDING for next segment

**Latency**: Text appears ~1-2 seconds after you stop speaking (time for speech-to-text processing).

Text Cleaning and Concatenation
--------------------------------

Before output, text is cleaned and concatenated:

.. code-block:: python

   def clean_dictation_text(text: str, add_trailing_space: bool = True) -> str:
       # Remove "..." artifacts
       cleaned = re.sub(r"\.\.\.", " ", text)

       # Add trailing space for proper segment joining
       if add_trailing_space and cleaned and not cleaned[-1].isspace():
           cleaned = cleaned + " "

       return cleaned

**Segment joining rules**:

- **First segment**: No leading space (e.g., "Hello world")
- **Subsequent segments**: Leading space added (e.g., " this is a test")
- **Period removal**: If previous segment ends with "." and current starts lowercase, remove period
- **Capitalization**: If no sentence boundary (no period), lowercase first letter of current segment

Example:

```
Segment 1: "Hello world"
Segment 2: " This is a test" → " this is a test" (lowercased because no period before)
Result: "Hello world this is a test"
```

Visual Dictation Mode
======================

Visual mode accumulates text in a popup window before insertion, letting you review and edit before committing.

Popup Lifecycle
---------------

.. mermaid::

   flowchart TD
       A["dictate visual"] --> B[Create Popup Window]
       B --> C[Enter RECORDING State]
       C --> D[Accumulate Text Segments]
       D --> E{User Action}

       E -->|Stop dictation| F[Insert Accumulated Text]
       E -->|Close popup| G[Cancel - No Insert]
       E -->|More speech| D

       F --> H[Close Popup]
       G --> H
       H --> I[Return to IDLE]

       style B fill:#fff4e1
       style F fill:#e8f5e9
       style G fill:#ffebee

**Real-time updates**: Each ``DictationTextRecognizedEvent`` updates the popup, showing accumulated text as you speak.

**Review before insert**: Unlike standard mode, visual mode doesn't type immediately. Text is accumulated in the popup, and you can review before deciding to insert or cancel.

Accumulation and Session Management
------------------------------------

Visual mode maintains a session with accumulated text:

.. code-block:: python

   class DictationSession:
       session_id: str  # Unique UUID for debugging
       mode: DictationMode
       start_time: float
       accumulated_text: str = ""
       last_text_time: Optional[float] = None  # For TYPE mode silence monitoring
       is_first_segment: bool = True

   async def _handle_dictation_text_recognized(self, event: DictationTextRecognizedEvent):
       if self._current_mode == DictationMode.VISUAL:
           # Clean and prepare text
           cleaned = clean_dictation_text(event.text, add_trailing_space=True)

           # Add to accumulated text with proper spacing
           if self._current_session.is_first_segment:
               text_to_add = cleaned
               self._current_session.is_first_segment = False
           else:
               text_to_add = cleaned  # Already has leading space

           self._current_session.accumulated_text += text_to_add

           # Update popup display
           await self.event_bus.publish(
               PartialDictationTextEvent(text=self._current_session.accumulated_text)
           )

**Session IDs**: Each session gets a unique ID for tracking and debugging race conditions.

Smart Dictation Mode: LLM-Enhanced
====================================

Smart mode uses a locally-hosted LLM to format, punctuate, and edit dictated text, providing formatted output.

Architecture
------------

.. mermaid::

   flowchart TD
       A[Accumulated Raw Text] --> B[AgenticPromptService]
       B --> C[Generate Prompt]
       C --> D[LLMService]
       D --> E[llama.cpp Model]
       E --> F[Token Generation]
       F --> G{Token Type}

       G -->|Text Token| H[Display]
       G -->|Command Token| I[Execute]

       H --> F
       I --> J{Command}
       J -->|REMOVE:N| K[Backspace N chars]
       J -->|NEWLINE| L[Insert newline]
       J -->|END| M[Finalize]

       K --> F
       L --> F
       M --> N[Output Final Text]

       style D fill:#e1f5ff
       style I fill:#fff4e1

Streaming Transcription Architecture
-------------------------------------

Smart, Visual, and Hidden modes use streaming transcription to provide incremental text during speech. The core challenge: Whisper processes max 30 seconds of audio, but users speak longer. The solution: two offsets track which audio is finalized (text extracted) and which is trimmed (discarded), ensuring no text is lost during the trim cycle.

What Whisper Returns
~~~~~~~~~~~~~~~~~~~~

Each time the transcription loop calls Whisper, it returns a list of segments. Each segment contains:

- **text**: The recognized speech for this time interval
- **start/end**: Timestamp boundaries (seconds from audio chunk start)
- **no_speech_prob**: Confidence it's actual speech (0.0-1.0; <0.45 means speech detected)

Multiple segments (2+) means Whisper detected clear boundaries between phrases—it identified where one thought ends and another begins.

The Two-Offset System
~~~~~~~~~~~~~~~~~~~~~

Problem: Buffer grows to 45 seconds then trims 30 seconds (to stay under Whisper's 30-second limit). If trimmed audio contained unfinalized text (detected speech not yet transcribed), that text is lost.

Solution: Two offsets track different aspects of the audio lifecycle:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Offset
     - Meaning
     - Advances When
   * - ``frames_offset``
     - Audio trimmed away (discarded)
     - Buffer exceeds 45 seconds
   * - ``timestamp_offset``
     - Audio finalized (text extracted)
     - Segment is finalized

The rule: never trim audio before ``timestamp_offset``. This guarantees unfinalized audio is never lost.

How Segments Are Finalized
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Segments finalize under three conditions:

**1. Multi-segment output** (Whisper returns 2+ segments):
   - If 2+ segments: all but the last are high-confidence (Whisper found clear boundaries)
   - Finalize all except the last (which is still being refined as more audio arrives)
   - Example: Whisper returns [segment 0-8s, segment 8-15s, partial 15-22s] → finalize first two

**2. Same-output stability** (identical prediction 10+ times):
   - Whisper returns the exact same text repeatedly → indicates no new information
   - Text is stable; finalize it

**3. Buffer trim approaching** (forced finalization):
   - If buffer trim would discard unfinalized audio, force finalize it first
   - Check: will trim discard audio before ``timestamp_offset``? If yes, finalize incomplete text and advance ``timestamp_offset``
   - This prevents information loss at buffer boundaries

Example Timeline
~~~~~~~~~~~~~~~~

.. code-block:: text

   Initial state: Buffer [0s-------45s], no offsets set

   Loop iteration 1 (t=0ms):
   - Call Whisper on audio[0:45]
   - Returns 3 segments: [0-8s "hello", 8-15s "world", 15-22s partial "how"]
   - Finalize 0-8s and 8-15s (multi-segment detected)
   - Advance timestamp_offset to 15s
   - Publish partial: "how" (incomplete, for Visual mode)

   Loop iteration 2 (t=100ms):
   - New audio arrived; buffer now [0s-------50s]
   - Call Whisper on audio[15:50] (only unfinalized portion)
   - Returns 2 segments: [15-18s refined "how are", 18-25s partial "you"]
   - Finalize 15-18s (multi-segment)
   - Advance timestamp_offset to 18s
   - Publish partial: "how are you" (accumulated)

   Loop iteration 3 (t=200ms):
   - Buffer still growing [0s-------52s]
   - Trim triggered (buffer > 45s): remove [0s:30s], set frames_offset = 30s
   - Check: is timestamp_offset (18s) < frames_offset (30s)? YES—unfinalized audio will be lost!
   - Force finalize partial "you" at end of its segment (25s)
   - Advance timestamp_offset to 25s
   - Continue with audio[25:52] for next call

Natural Overlap Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~

Each loop iteration reprocesses unfinalized audio with fresh context:

- Finalized segments: never reprocessed (locked in place)
- Unfinalized segments: reprocessed each loop (refined with new audio context)

This creates natural overlap without manual duplication. Whisper automatically improves incomplete segments as more audio arrives.

Transcription Loop Timing
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Minimum audio threshold**: Wait for 1+ second of audio before calling Whisper (ensures adequate context)
- **Processing frequency**: Call Whisper every ~50-100ms (adaptive: faster when audio available, slower when silent)
- **Unfinalized audio maximum**: If unfinalized audio exceeds 25 seconds with no valid finalization, clip to last 5 seconds (prevents unbounded growth)

Two-Phase Workflow for Smart Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Phase 1 - Streaming Accumulation** (RECORDING state):
   Loop continuously calls Whisper → finalizes detected segments → accumulates text in `accumulated_text` → publishes partial events for Visual mode UI

**Phase 2 - LLM Post-Processing** (PROCESSING_LLM state, smart mode only):
   Send complete `accumulated_text` to LLM → stream tokens real-time → execute formatting commands (REMOVE:N for backspace, NEWLINE for line break, END to finish)

LLM Service
-----------

The LLM service wraps llama.cpp for local model inference:

.. code-block:: python

   class LLMService:
       def __init__(self, event_bus: EventBus, config: GlobalAppConfig):
           self.llm: Optional[Llama] = None
           self._model_loaded = False
           self._warmed_up = False

           # Auto-calculate threads: 75% of CPU cores, capped at 12
           cpu_count = multiprocessing.cpu_count()
           self.n_threads = config.llm.n_threads or max(4, min(int(cpu_count * 0.75), 12))

       async def initialize(self) -> bool:
           # Download model if missing
           if not self.model_downloader.model_exists(self.model_filename):
               await self.model_downloader.download_model(...)

           # Load with performance optimizations
           self.llm = await self._load_model_async()

           # Warm up with quick inference
           await self._warmup_model()
           return True

**Model loading options**:

- **GPU layers**: Configurable number of GPU-accelerated layers (0 = CPU only)
- **Flash attention**: Optional fast attention computation
- **Memory mapping**: Maps model file to memory for efficient loading
- **Thread configuration**: Configurable CPU threads for parallel token generation

**Model startup modes**:

- **Startup**: Load during application initialization (slow startup, fast first use)
- **Background**: Load in background after startup completes
- **On-demand**: Load when first smart dictation command is issued

Agentic Prompt System
---------------------

The ``AgenticPromptService`` generates prompts that instruct the LLM to format text AND issue editing commands:

.. code-block:: text

   You are a text formatting assistant. Format the following dictation:

   Raw text: "hello world this is a test period new line goodbye"

   Instructions:
   - Add proper punctuation and capitalization
   - When you need to edit previous text, use commands:
     - REMOVE:N - removes last N characters
     - NEWLINE - inserts line break
     - END - signals completion

   Output formatted text with embedded commands as needed.

The LLM output might be:

.. code-block:: text

   Hello world, this is a test.
   NEWLINE
   Goodbye!
   END

The coordinator parses this output and executes commands in real-time:

.. code-block:: python

   async def _handle_llm_token(self, event: LLMTokenGeneratedEvent):
       token = event.token

       if token.startswith("REMOVE:"):
           count = int(token[7:])  # Extract number after "REMOVE:"
           await self.text_input.backspace(count)
       elif token == "NEWLINE":
           await self.text_input.add_newline()
       elif token == "END":
           await self._finalize_smart_dictation()
       else:
           # Regular text token - display and type
           await self.text_input.input_text(token)

**Why commands?** The LLM can't directly see what's displayed. Commands let it correct mistakes, restructure sentences, and insert formatting after generating text.

Streaming Display
-----------------

Smart mode shows LLM output in real-time as tokens are generated:

.. mermaid::

   sequenceDiagram
       participant U as User
       participant C as DictationCoordinator
       participant LLM as LLMService
       participant Input as TextInputService

       U->>C: "stop dictation" (smart mode)
       C->>LLM: Process "hello world new line goodbye"

       LLM->>C: Token: "Hello"
       C->>Input: Type "Hello"

       LLM->>C: Token: " world"
       C->>Input: Type " world"

       LLM->>C: Token: "."
       C->>Input: Type "."

       LLM->>C: Command: NEWLINE
       C->>Input: Press Enter

       LLM->>C: Token: "Goodbye"
       C->>Input: Type "Goodbye"

       LLM->>C: Token: "!"
       C->>Input: Type "!"

       LLM->>C: Command: END
       C->>C: Finalize text

This streaming provides visual feedback during ~2-5 second LLM processing.

Type Mode: Raw Insertion
==========================

Type mode provides raw, unformatted text insertion with no cleaning or spacing adjustments. It automatically stops after silence timeout.

How It Works
------------

.. code-block:: python

   async def _start_session(self, mode: DictationMode) -> None:
       # For TYPE mode, start silence monitoring
       if mode == DictationMode.TYPE:
           self._type_silence_task = asyncio.create_task(self._monitor_type_silence())

   async def _monitor_type_silence(self) -> None:
       """Monitor silence timeout for TYPE dictation mode"""
       timeout = self.config.dictation.type_dictation_silence_timeout

       while True:
           await asyncio.sleep(0.1)

           with self._state_lock:
               session = self._current_session
               if not session or session.mode != DictationMode.TYPE:
                   return

               time_since_last_text = time.time() - session.last_text_time

               if time_since_last_text >= timeout:
                   logger.info(f"Type dictation silence timeout exceeded ({timeout}s)")
                   break

       await self._stop_session()

**Processing**: Type mode receives text from ``DictationTextRecognizedEvent`` but applies no formatting. The text is inserted directly as recognized.

**Auto-stop**: Type mode automatically stops after configurable silence (typically a few seconds). This prevents accidentally recording too much.

**Use cases**:

- Dictating variable names or code (no auto-formatting)
- Entering data that shouldn't be touched
- Quick insertion without processing or review

Hidden Dictation Mode
=======================

Hidden mode silently accumulates text without UI display or real-time feedback. When you stop, all text is pasted at once. Useful for seamless, uninterrupted dictation.

How It Works
------------

.. code-block:: python

   async def _start_session(self, mode: DictationMode) -> None:
       # For HIDDEN mode: start streaming without UI display
       if mode == DictationMode.HIDDEN:
           await self._start_streaming_mode(mode)
           # No popup window created—accumulation happens silently
           await self.event_bus.publish(HiddenDictationStartedEvent(...))

   async def _handle_dictation_text_recognized(self, event: DictationTextRecognizedEvent):
       if self._current_mode == DictationMode.HIDDEN:
           # Accumulate text exactly like visual mode
           cleaned = clean_dictation_text(event.text, add_trailing_space=True)
           self._current_session.accumulated_text += cleaned
           # No UI update—silent accumulation

**Processing**: Like visual mode, hidden mode uses streaming transcription. Text is accumulated but never displayed.

**Output**: When you say "stop dictation", all accumulated text is pasted at once via clipboard.

**Use cases**:

- Dictating without visible feedback (better focus)
- Dictation during presentations or streaming
- Raw data capture without distraction

Stop Detection During Dictation
================================

While dictating, the system monitors for stop triggers. The command listener continues running during dictation but filters output:

.. mermaid::

   flowchart TD
       A[Dictation Active] --> B[CommandAudioListener]
       B --> C[Running - Filtering]
       C --> D[CommandAudioSegmentReadyEvent]
       D --> E[SpeechToTextService]
       E --> F{Recognition}

       F -->|stop dictation| G[CommandTextRecognizedEvent]
       F -->|other text| H[Discard - Not stop trigger]

       G --> I[CentralizedCommandParser]
       I --> J[DictationStopCommand]
       J --> K[DictationCoordinator]
       K --> L[Stop Current Mode]

       style G fill:#e8f5e9
       style H fill:#ffebee
       style L fill:#e1f5ff

**Why keep listening?** The command listener doesn't disable—it filters. This allows you to say "stop dictation" at any time without the listener interfering with dictation text.

**Mode awareness**: When dictation starts, the coordinator publishes ``DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode=...)`` with the specific mode (standard, visual, smart, or type). Other systems use this to:

- Filter STT output to only recognize stop trigger words
- Disable sound recognition
- Disable Markov predictions

When dictation stops, ``DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive")`` re-enables all systems.

Text Input Service
==================

The ``TextInputService`` handles actual text insertion using two configurable methods.

Clipboard Method (Default)
---------------------------

.. code-block:: python

   async def input_text(self, text: str, add_trailing_space: bool = True) -> bool:
       cleaned_text = clean_dictation_text(text, add_trailing_space)

       if self.config.use_clipboard:
           # Save clipboard, copy text, paste, restore clipboard
           original = pyperclip.paste()
           pyperclip.copy(cleaned_text)
           time.sleep(self.config.clipboard_paste_delay_pre)

           # Use keyDown/keyUp instead of hotkey() to prevent repeat
           pyautogui.keyDown("ctrl")
           time.sleep(0.01)
           pyautogui.press("v")
           time.sleep(0.01)
           pyautogui.keyUp("ctrl")

           time.sleep(self.config.clipboard_paste_delay_post)
           pyperclip.copy(original)  # Restore

**Advantages**: Fast, reliable, handles special characters and Unicode.

**Disadvantages**: Temporarily overwrites clipboard (restored immediately).

**Key repeat prevention**: Uses explicit keyDown/keyUp with small delays instead of hotkey() to prevent Windows autorepeat issues.

Direct Typing Method (Fallback)
--------------------------------

.. code-block:: python

   async def input_text(self, text: str, add_trailing_space: bool = True) -> bool:
       cleaned_text = clean_dictation_text(text, add_trailing_space)

       if not self.config.use_clipboard:
           # Type character by character
           for char in cleaned_text:
               pyautogui.write(char, interval=self.config.typing_delay)

**Advantages**: Doesn't affect clipboard.

**Disadvantages**: Slower, less reliable with special characters.

Session Management and Cleanup
===============================

Each dictation session maintains state that must be properly cleaned up:

.. code-block:: python

   class DictationSession:
       session_id: str  # Unique UUID
       mode: DictationMode
       start_time: float
       accumulated_text: str = ""
       last_text_time: Optional[float] = None
       is_first_segment: bool = True

   async def _stop_session(self) -> None:
       """Stop dictation session with proper cleanup"""
       try:
           with self._state_lock:
               session = self._current_session
               if not session:
                   return

               # For TYPE mode: cancel silence monitoring
               if session.mode == DictationMode.TYPE:
                   self._cancel_type_silence_task()

               # For SMART/VISUAL: use streaming path
               if session.mode in (DictationMode.SMART, DictationMode.VISUAL):
                   await self._stop_streaming_mode(session)
                   return

               # For STANDARD: finalize immediately
               self._current_session = None
               self._set_state(DictationState.IDLE)

           # Notify other systems
           await self.event_bus.publish(
               DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive")
           )
       except Exception as e:
           logger.error(f"Session stop error: {e}", exc_info=True)

**Session IDs**: Each session gets a unique ID for tracking in logs and debugging.

Thread Safety
=============

The coordinator uses a ``threading.RLock`` to protect all state modifications:

.. code-block:: python

   with self._state_lock:
       # All state modifications within lock
       self._current_state = DictationState.RECORDING
       self._current_session = DictationSession(...)

This prevents race conditions when:

- Dictation text arrives while stopping dictation
- LLM tokens arrive after session was cancelled
- Multiple stop commands arrive in quick succession

What Happens Next
==================

After dictation text is output:

- **UI updates** reflect dictation status (active/inactive)
- **Command recognition** resumes full operation
- **Markov predictions** resume suggesting next commands
- **Sound recognition** resumes listening for sound-mapped commands
- **System returns** to idle state waiting for next voice input

The user interface that displays dictation status and controls is covered in :doc:`user_interface`.
