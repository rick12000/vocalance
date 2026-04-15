Dictation System
##################

This page explains how Vocalance handles dictation through six modes—standard, visual, smart, amend, type, and hidden—each optimized for different use cases, all orchestrated by the ``DictationCoordinator``. Voice phrases that start or stop dictation are configured in ``DictationConfig`` (see :doc:`command_parsing`).

System Overview
================

The dictation system operates independently from command execution. Once activated, it accumulates speech input and processes it through mode-specific pipelines before outputting text. Unlike commands which execute immediately, dictation involves longer processing with explicit start and stop boundaries.

.. mermaid::

   flowchart TD
       A[DictationCommandParsedEvent] --> B{Command Type}
       B -->|standard start| C[Standard Mode]
       B -->|visual start| D[Visual Mode]
       B -->|smart start| E[Smart Mode]
       B -->|amend start| AM[Amend Mode]
       B -->|type start| F[Type Mode]
       B -->|hidden start| G[Hidden Mode]
       B -->|stop| H[Stop & Finalize]

       C --> I[DictationTextRecognizedEvent]
       D --> I
       E --> I
       AM --> I
       F --> I
       G --> I

       I --> J{Mode?}
       J -->|Standard| K[Direct Type]
       J -->|Visual| L[Popup Accumulate]
       J -->|Smart| M[LLM Queue]
       J -->|Amend| M
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
       style AM fill:#d4e8fc
       style F fill:#fce4ec
       style G fill:#e8e8e8

Dictation flows from parse event → mode selection → text recognition → mode-specific processing → output. Standard types immediately; visual accumulates for review; **smart** and **amend** share streaming STT and a dual-pane LLM phase (smart formats dictated text; amend applies spoken instructions to a captured selection); type inserts raw text; hidden accumulates silently for paste on stop.

Post-processing and modifiers
-----------------------------

After Moonshine (or VAD) emits text, the coordinator runs a **base** pass on every segment: spoken cardinals and digit-by-digit sequences are replaced with decimal numerals via ``vocalance.app.utils.number_parser.replace_spoken_numbers_in_text``. Optional **modifiers** (which can be stacked if compatible) add transforms: title case (**upper**), ALL CAPS (**capitals**), UpperCamelCase (**camel**; punctuation stripped), snake_case (**snake**; punctuation stripped), kebab-case (**kebab**; punctuation stripped), lowercase (**diminish**), strip punctuation (**strip**), and a **spelling** mode that strips punctuation then maps spoken punctuation words to symbols and reapplies sentence casing. Modifier phrases are configured on ``DictationConfig`` (defaults: ``upper``, ``capitals``, ``camel``, ``snake``, ``spelling``, ``kebab``, ``diminish``, ``strip``). In **type** mode, ``strip`` and ``diminish`` are enabled by default.

While dictation is active, **Vosk** still runs on command audio segments. Besides the shared stop phrase (default ``amber``; see ``DictationConfig.stop_trigger``), it detects modifier phrases and publishes ``DictationModifierPhraseEvent`` (no ``CommandTextRecognizedEvent``). The coordinator toggles the same phrase off or switches to another modifier, updates session state, publishes ``DictationModifierStateChangedEvent`` (the **smart**, **amend**, and **visual** popup layouts show a small faded modifier chip; standard, type, and hidden wave-only UIs do not), and starts a short **Moonshine suppress window** (monotonic clock) so partials/finals from the same utterance are dropped. **Type** mode’s silence timeout only advances on segments that yield non-empty text after trigger stripping and post-processing; Vosk-only modifier phrases do not touch that timer.

.. _modifier-pipeline-reference:

Modifier pipeline (reference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following ties together the pieces added for voice modifiers; all phrase strings live on ``DictationConfig``.

1. **Detection** — ``SpeechToTextService`` runs Vosk on each command segment while dictation is active. If the transcript contains a configured modifier phrase (longest match first), it publishes ``DictationModifierPhraseEvent``. Stop phrase handling is unchanged.

2. **Session state** — ``DictationCoordinator._handle_dictation_modifier_phrase`` updates ``DictationSession.active_modifiers`` (a set of active modifiers). If the same phrase is spoken, it toggles off. If a new phrase is spoken, it is added to the set, and any mutually exclusive modifiers (e.g., casing vs casing, punctuation vs punctuation) are removed. It publishes ``DictationModifierStateChangedEvent`` and sets ``_moonshine_suppress_until`` to ``time.monotonic() + MOONSHINE_MODIFIER_SUPPRESS_SEC`` (~0.55s) so ``_moonshine_on_partial`` / ``_moonshine_on_final`` return early during that window.

3. **Segment text** — ``_clean_text`` removes each configured trigger and modifier phrase only as a **whole phrase** (case-insensitive word-boundary match via ``re.escape``). Phrases are applied **longest first** (multi-word before single-word) so a start trigger like ``green`` does not strip the second word of ``smart green`` before that phrase is removed. ``_prepare_dictation_segment_final`` / ``_prepare_dictation_segment_partial`` apply aliases, drop isolated punctuation fragments, then call ``dictation_postprocess.apply_dictation_postprocess`` or ``apply_dictation_postprocess_partial`` (partials skip the **spelling** transform).

4. **Typing** — For standard/type modes, ``_dictation_segment_input_options`` passes ``add_trailing_space=False`` and ``skip_prose_segment_join_rules=True`` when the active modifiers include **camel**, **snake**, **kebab**, or **spelling**, so ``TextInputService.input_text`` does not append a trailing space or apply mid-sentence lowercasing that would break identifiers.

5. **UI** — ``QtDictationPopupController`` forwards ``DictationModifierStateChangedEvent`` to ``QtDictationPopupView.set_modifier_banner``; reserved labels beside the dictation column titles (dual-pane and visual) show the faded chip. Wave-only popups ignore active modifier display.

6. **Types** — ``DictationModifierId`` is defined once in ``vocalance.app.events.dictation_events`` and reused by post-processing, the coordinator, and STT.

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
       RECORDING --> PROCESSING_LLM: Stop smart or amend
       RECORDING --> IDLE: Stop standard/visual/hidden/type
       PROCESSING_LLM --> IDLE: LLM complete
       IDLE --> [*]

       note right of RECORDING
           Accumulating text from
           DictationTextRecognizedEvent
       end note

       note right of PROCESSING_LLM
           Smart & amend dual-pane modes
           LLM on accumulated text
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

Dictation modes are triggered when recognized text matches the configured triggers (defaults: ``green``, ``visual green``, ``smart green``, ``amend``, ``type``, ``hidden green``, ``amber`` for stop—see ``DictationConfig``).

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
       elif isinstance(command, DictationHiddenStartCommand):
           await self._start_session(DictationMode.HIDDEN)
       elif isinstance(command, DictationAmendStartCommand):
           await self._start_session(DictationMode.AMEND)
       elif isinstance(command, DictationTypeCommand):
           await self._start_session(DictationMode.TYPE)
       elif isinstance(command, DictationStopCommand):
           await self._stop_session()

System Awareness During Dictation
----------------------------------

When dictation starts, the coordinator broadcasts ``DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode=...)`` which:

- **Disables sound recognition** (prevents sound-mapped commands)
- **Narrows command-path recognition** to the stop phrase and configured modifier phrases (see ``SpeechToTextService``)

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

       U->>Coord: Standard start (e.g. "green")
       Coord->>Coord: Enter RECORDING state

       U->>STT: Speak: "Hello world"
       STT->>Coord: DictationTextRecognizedEvent("Hello world")
       Coord->>Input: Type "Hello world"

       U->>STT: Speak: "This is a test"
       STT->>Coord: DictationTextRecognizedEvent("This is a test")
       Coord->>Input: Type " This is a test"

       U->>Coord: Stop phrase (e.g. "amber")
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

**Segment joining rules** (when ``skip_prose_segment_join_rules`` is false — default prose dictation):

- **First segment**: No leading space (e.g., "Hello world")
- **Subsequent segments**: Leading space added (e.g., " this is a test")
- **Period removal**: If previous segment ends with "." and current starts lowercase, remove period
- **Capitalization**: If no sentence boundary (no period), lowercase first letter of current segment

**Modifier modes** (camel, snake, kebab, spelling): the coordinator passes ``skip_prose_segment_join_rules=True`` and usually ``add_trailing_space=False`` so segments are not merged as running prose; see :ref:`modifier-pipeline-reference`.

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
       A["visual start phrase"] --> B[Create Popup Window]
       B --> C[Enter RECORDING State]
       C --> D[Accumulate Text Segments]
       D --> E{User Action}

       E -->|Stop phrase| F[Insert Accumulated Text]
       E -->|Close popup| G[Cancel - No Insert]
       E -->|More speech| D

       F --> H[Close Popup]
       G --> H
       H --> I[Return to IDLE]

       style B fill:#fff4e1
       style F fill:#e8f5e9
       style G fill:#ffebee

**Real-time updates**: The streaming pipeline publishes partial/final dictation events so the popup reflects text as you speak.

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

Streaming transcription (Moonshine)
------------------------------------

Smart, amend, visual, hidden, standard, and type dictation share the Moonshine streaming path: ``AudioService`` calls a dictation chunk callback from the recorder thread; ``DictationCoordinator`` enqueues PCM on an ingress thread and feeds ``MoonshineDictationStreamSession.add_audio_pcm16``. Native Moonshine callbacks deliver line text changes and completed lines; the coordinator publishes ``PartialDictationTextEvent``, ``FinalDictationTextEvent``, and (for standard/type) ``DictationTextRecognizedEvent`` with ``engine="moonshine"`` where appropriate.

**Partials vs finals**: Partials update live UI; completed lines append to the session accumulator and emit finals for streaming modes. Duplicate-line and hallucination checks run before publishing.

**Rotation**: When ``stt.moonshine_max_stream_line_duration_seconds`` is set, ``add_audio_pcm16`` signals the coordinator to close the current native stream and open a new one after enough audio on one line, bounding work on long sessions.

**Cadence**: ``stt.moonshine_streaming.stream_update_interval`` controls partial refresh frequency (trade-off: responsiveness vs CPU). Native decode gating uses ``stt.moonshine_streaming.transcription_interval`` and VAD via ``stt.moonshine_streaming.vad_threshold`` (see ``MoonshineStreamingConfig``).

**Batch API**: ``MoonshineSTT.recognize`` remains for short offline segments; continuous dictation does not use a rolling in-process buffer like the old streaming STT loop.

Two-phase workflow (smart / amend)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Phase 1 — Recording**: Moonshine partial/final events update the UI and ``accumulated_text`` (and the amend instructions column).

**Phase 2 — LLM** (``PROCESSING_LLM``): Smart sends accumulated dictation to ``process_dictation_streaming``; amend sends the selection snapshot plus spoken instructions to ``process_amend_streaming``. Both stream tokens and formatting commands (REMOVE:N, NEWLINE, END).

LLM Service
-----------

The LLM service uses llama.cpp with **CPU-only** inference. Weights come from the
built-in allow list (``LocalLLMAllowList`` in ``app_config``) of official `Qwen …-Instruct-GGUF` repositories (Q5_K_M). The active model
is selected in Settings (``llm.selected_model_id``). The setting updates when the
chosen model is already on disk, or after a successful download from Settings.
Additional models use a cancellable download dialog; the previous active model stays
in effect until a new one is fully installed.

On first launch, if the configured model bundle is missing, startup **blocks** with
the splash progress UI until that bundle is fetched.

For each request, the service loads the GGUF from disk (first shard path for split
files), runs chat completion, then unloads to avoid keeping large models resident
between sessions.

**Model loading options** (CPU):

- **Flash attention**: Optional faster attention where supported by the build
- **Memory mapping**: Maps model files for efficient loading
- **Thread configuration**: Configurable CPU threads for prompt batching and generation

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

       U->>C: Stop phrase (smart mode)
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

Amend mode (selection + instructions)
======================================

Amend mode is a second **dual-pane LLM** path alongside smart. After the amend start phrase, the coordinator captures the current selection via ``TextInputService.capture_selection_via_copy`` (Ctrl+C, read clipboard, restore prior clipboard) and stores a snapshot. Moonshine streaming fills the left column with your **spoken instructions**; the right column shows LLM output like smart mode. On stop, ``LLMService.process_amend_streaming`` builds messages from the snapshot plus instructions (and the current agentic preset). ``DictationSessionEvent`` uses ``mode="amend"``; the popup shows the same layout as smart with the left title **Prompt**. Streaming modes rely on partial/final dictation events rather than immediate typing from every ``DictationTextRecognizedEvent``.

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
           await self.event_bus.publish(DictationSessionEvent(mode="hidden", state="started"))

   async def _handle_dictation_text_recognized(self, event: DictationTextRecognizedEvent):
       if self._current_mode == DictationMode.HIDDEN:
           # Accumulate text exactly like visual mode
           cleaned = clean_dictation_text(event.text, add_trailing_space=True)
           self._current_session.accumulated_text += cleaned
           # No UI update—silent accumulation

**Processing**: Like visual mode, hidden mode uses streaming transcription. Text is accumulated but never displayed.

**Output**: When you say the configured stop phrase, all accumulated text is pasted at once via clipboard.

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

       F -->|stop phrase| G[CommandTextRecognizedEvent]
       F -->|other text| H[Discard - Not stop trigger]

       G --> I[CentralizedCommandParser]
       I --> J[DictationStopCommand]
       J --> K[DictationCoordinator]
       K --> L[Stop Current Mode]

       style G fill:#e8f5e9
       style H fill:#ffebee
       style L fill:#e1f5ff

**Why keep listening?** The command listener doesn't disable—it filters. This allows you to say the stop phrase at any time without the listener interfering with dictation text.

**Mode awareness**: When dictation starts, the coordinator publishes ``DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode=...)`` with the specific mode (including ``amend``). Other systems use this to:

- Filter STT output to only recognize stop trigger words
- Disable sound recognition

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

               # Streaming STT modes (smart, amend, visual, hidden)
               if session.mode in _STREAMING_STT_MODES:
                   await self._stop_streaming_mode(session)
                   return

               self._current_session = None
               self._set_state(DictationState.IDLE)

           if session:
               await self._finalize_session(session)
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
- **Sound recognition** resumes listening for sound-mapped commands
- **System returns** to idle state waiting for next voice input

The user interface that displays dictation status and controls is covered in :doc:`user_interface`.
