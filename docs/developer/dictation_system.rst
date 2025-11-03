Dictation System
=================

Dictation is a core feature that converts continuous speech into typed text, with four distinct modes for different use cases. When the Command Parser identifies dictation commands ("dictate", "type", "smart dictate", "visual dictate"), it routes them to the DictationCoordinator, which represents an alternative execution path parallel to automation commands.

DictationCoordinator: The Orchestrator
---------------------------------------

The ``DictationCoordinator`` is a sophisticated state machine that manages all dictation workflows, integrating STT, LLM processing, and text input.

Dictation Modes
~~~~~~~~~~~~~~~

The system supports four dictation modes, each with different behavior:

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

**4. VISUAL Mode** (trigger: "visual dictate"):

- Accumulates all spoken text during session with UI display
- Shows dictation in a popup window (single box, no LLM pane)
- When stopped, pastes accumulated text directly without LLM processing
- Provides visual feedback like smart mode but with standard mode's backend flow
- Best for: When you want to see what you're dictating without LLM overhead

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
