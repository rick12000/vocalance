Services Architecture
=====================

This document provides a comprehensive guide to the services architecture of Vocalance, detailing module dependencies, event flows, and optimization strategies.

Module Dependency Graph
-----------------------

.. mermaid::

   graph TB
    %% Core Infrastructure
    EventBus[EventBus]
    GlobalAppConfig[GlobalAppConfig]

    %% Audio Services
    SimpleAudioService[SimpleAudioService]
    Recorder[Recorder]
    STTService[SpeechToTextService]
    VoskSTT[VoskSTT]
    WhisperSTT[WhisperSTT]
    StreamlinedSoundService[StreamlinedSoundService]
    StreamlinedSoundRecognizer[StreamlinedSoundRecognizer]

    %% Command Processing
    CentralizedCommandParser[CentralizedCommandParser]
    AutomationService[AutomationService]
    DictationCoordinator[DictationCoordinator]
    TextInputService[TextInputService]
    LLMService[LLMService]
    AgenticPromptService[AgenticPromptService]

    %% Helper Services (Shared Components)
    ProtectedTermsValidator[ProtectedTermsValidator]
    CommandActionMapProvider[CommandActionMapProvider]
    CommandHistoryManager[CommandHistoryManager]

    %% UI Integration Services
    GridService[GridService]
    ClickTrackerService[ClickTrackerService]
    MarkService[MarkService]

    %% Storage & Persistence
    StorageService[StorageService]
    SettingsService[SettingsService]
    CommandManagementService[CommandManagementService]

    %% Optimization Services
    MarkovCommandService[MarkovCommandService]

    %% Dependencies
    EventBus --> SimpleAudioService
    EventBus --> STTService
    EventBus --> CentralizedCommandParser
    EventBus --> AutomationService
    EventBus --> DictationCoordinator
    EventBus --> GridService
    EventBus --> MarkService
    EventBus --> MarkovCommandService

    GlobalAppConfig --> SimpleAudioService
    GlobalAppConfig --> STTService
    GlobalAppConfig --> CentralizedCommandParser
    GlobalAppConfig --> AutomationService
    GlobalAppConfig --> DictationCoordinator
    GlobalAppConfig --> GridService
    GlobalAppConfig --> MarkService
    GlobalAppConfig --> MarkovCommandService

    %% Audio Flow
    SimpleAudioService --> Recorder
    SimpleAudioService --> EventBus
    Recorder --> EventBus

    STTService --> VoskSTT
    STTService --> WhisperSTT
    STTService --> StreamlinedSoundService

    StreamlinedSoundService --> StreamlinedSoundRecognizer
    StreamlinedSoundService --> StorageService

    %% Helper Services Dependencies
    ProtectedTermsValidator --> StorageService
    ProtectedTermsValidator --> GlobalAppConfig
    CommandActionMapProvider --> StorageService
    CommandHistoryManager --> StorageService

    %% Command Processing Flow
    CentralizedCommandParser --> CommandActionMapProvider
    CentralizedCommandParser --> CommandHistoryManager
    CentralizedCommandParser --> StorageService
    CentralizedCommandParser --> EventBus

    CommandManagementService --> ProtectedTermsValidator
    CommandManagementService --> CommandActionMapProvider
    MarkService --> ProtectedTermsValidator

    AutomationService --> EventBus

    DictationCoordinator --> TextInputService
    DictationCoordinator --> LLMService
    DictationCoordinator --> AgenticPromptService
    DictationCoordinator --> StorageService

    %% UI Integration Flow
    GridService --> EventBus
    GridService --> GlobalAppConfig

    ClickTrackerService --> StorageService
    ClickTrackerService --> EventBus

    MarkService --> StorageService
    MarkService --> EventBus

    %% Storage Dependencies
    StorageService --> EventBus

    SettingsService --> StorageService
    CommandManagementService --> StorageService

    %% Optimization Dependencies
    MarkovCommandService --> StorageService
    MarkovCommandService --> EventBus

High-Level Information Flow
---------------------------

Audio Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

**Recorder Layer**: The audio processing begins with dual independent recorders managed by ``SimpleAudioService``:

- **Command Recorder**: Optimized for speed and responsiveness, using shorter audio segments and aggressive silence detection
- **Dictation Recorder**: Optimized for accuracy, using longer audio segments and more tolerant silence thresholds

Both recorders run continuously, using ``sounddevice`` for audio capture and Voice Activity Detection (VAD) for speech identification. The command recorder publishes ``AudioDetectedEvent`` for ultra-low latency Markov prediction bypass.

**Recognition Layer**: Audio segments flow to ``SpeechToTextService`` with mode-aware processing:

- **Command Mode**: Uses Vosk STT for speed-optimized recognition, checking only for amber trigger words during dictation
- **Dictation Mode**: Uses Whisper STT for accuracy-optimized recognition, processing full speech content

**Parallel Processing**: If STT fails to recognize speech, audio segments are sent to ``StreamlinedSoundService`` for custom sound recognition.

Command Execution Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Parsing Layer**: Recognized text flows to ``CentralizedCommandParser`` using hierarchical parsing:

1. **Dictation Commands**: ``start dictation``, ``stop dictation``, ``smart dictation``, ``type dictation``
2. **Mark Commands**: ``mark <label>``, ``mark delete <label>``, ``visualize marks``
3. **Grid Commands**: ``show grid [number]``, ``select [number]``, ``cancel grid``
4. **Automation Commands**: Exact matches or parameterized (``command [number]``) from stored action map

The parser uses helper services for improved separation of concerns:

- **CommandActionMapProvider**: Centralized building of automation command maps from custom and default commands
- **CommandHistoryManager**: In-memory command history tracking with shutdown persistence
- **ProtectedTermsValidator**: Validates terms against all protected phrases (commands, marks, sounds, system triggers)

**Markov Prediction Integration**: The parser handles Markov prediction deduplication internally:

- Stores recent predictions in a 500ms time window
- Compares STT/sound results with predictions to avoid duplicate execution
- Sends accuracy feedback to ``MarkovCommandService`` for learning

**Execution Layer**: Parsed commands are published as specific events that services consume:

- ``DictationCommandParsedEvent`` → ``DictationCoordinator``
- ``AutomationCommandParsedEvent`` → ``AutomationService``
- ``MarkCommandParsedEvent`` → ``MarkService``
- ``GridCommandParsedEvent`` → ``GridService``

Storage Integration
~~~~~~~~~~~~~~~~~~~

**Centralized Storage**: All services use ``StorageService`` for:

- **Command Storage**: Automation commands and sound mappings with live updates
- **Settings Persistence**: User preferences and configurations
- **Mark Management**: Position and label storage with performance caching
- **Click Tracking**: Mouse position history for grid optimization

**Event-Driven Updates**: Storage changes trigger events that update dependent services in real-time.

Dictation Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dictation Modes**: ``DictationCoordinator`` handles multiple dictation modes:

- **Standard Dictation**: Immediate text input via ``TextInputService``
- **Type Dictation**: Similar to standard but different UI handling
- **Smart Dictation**: Accumulates text, sends to ``LLMService`` for processing, then inputs refined text

**LLM Integration**: Smart dictation uses ``AgenticPromptService`` for grammar correction and clarity improvements.

Event Structure and Dependencies
--------------------------------

Audio Event Flow
~~~~~~~~~~~~~~~~

The audio processing pipeline operates through a sequence of services that transform raw audio into actionable text or sound recognition results. Each service subscribes to specific events, processes them, and publishes new events for downstream consumers.

.. mermaid::

   graph TB
   Recorder[Recorder<br/>in SimpleAudioService] -->|publishes| CmdAudio[CommandAudioSegmentReadyEvent]
   Recorder -->|publishes| DictAudio[DictationAudioSegmentReadyEvent]
   Recorder -->|publishes| AudioDetected[AudioDetectedEvent]

   CmdAudio -->|consumed by| STTService[SpeechToTextService]
   DictAudio -->|consumed by| STTService
   AudioDetected -->|consumed by| MarkovPredictor[MarkovCommandService]

   STTService -->|publishes| CmdText[CommandTextRecognizedEvent]
   STTService -->|publishes| DictText[DictationTextRecognizedEvent]
   STTService -->|publishes on STT failure| SoundEvent[ProcessAudioChunkForSoundRecognitionEvent]

   SoundEvent -->|consumed by| SoundService[StreamlinedSoundService]
   SoundService -->|publishes| CustomSound[CustomSoundRecognizedEvent]

   MarkovPredictor -->|publishes on high confidence| MarkovPred[MarkovPredictionEvent]
   MarkovPred -->|consumed by| Parser[CentralizedCommandParser]

**Event Details and Data Structures**:

Audio capture events contain raw binary audio data along with metadata required for processing:

- **CommandAudioSegmentReadyEvent** (Priority: HIGH)

  - ``audio_bytes`` (bytes): Raw PCM audio data captured from microphone in int16 format
  - ``sample_rate`` (int): Sample rate in Hz (typically 16000)
  - Published by: ``Recorder`` when voice activity ends in command mode
  - Consumed by: ``SpeechToTextService._handle_command_audio_segment()``
  - Characteristics: Shorter segments (~0.5-2s) optimized for responsive command recognition

- **DictationAudioSegmentReadyEvent** (Priority: HIGH)

  - ``audio_bytes`` (bytes): Raw PCM audio data in int16 format, typically longer segments
  - ``sample_rate`` (int): Sample rate in Hz (typically 16000)
  - Published by: ``Recorder`` when voice activity ends in dictation mode
  - Consumed by: ``SpeechToTextService._handle_dictation_audio_segment()``
  - Characteristics: Longer segments (~1-5s) optimized for accuracy over speed

- **AudioDetectedEvent** (Priority: CRITICAL)

  - ``timestamp`` (float): Unix timestamp when audio above threshold was first detected
  - Published by: ``Recorder`` immediately upon detecting voice activity
  - Consumed by: ``MarkovCommandService._handle_audio_detected_fast_track()``
  - Purpose: Enables ultra-low latency prediction by triggering Markov chain analysis before STT completes
  - Timing: Published 50-200ms before audio segment is ready for STT processing

Speech recognition result events:

- **CommandTextRecognizedEvent** (Priority: HIGH, extends TextRecognizedEvent)

  - ``text`` (str): Recognized command text from Vosk STT engine
  - ``processing_time_ms`` (float): Time taken for STT processing in milliseconds
  - ``engine`` (str): STT engine identifier ("vosk")
  - ``mode`` (str): Processing mode ("command")
  - ``confidence`` (float): Recognition confidence (0.0-1.0, default 1.0)
  - Published by: ``SpeechToTextService._publish_recognition_result()``
  - Consumed by: ``CentralizedCommandParser._handle_command_text_recognized()``
  - When dictation is active: Only published if amber stop words are detected

- **DictationTextRecognizedEvent** (Priority: HIGH, extends TextRecognizedEvent)

  - ``text`` (str): Recognized dictation text from Whisper STT engine
  - ``processing_time_ms`` (float): Time taken for STT processing
  - ``engine`` (str): STT engine identifier ("whisper")
  - ``mode`` (str): Processing mode ("dictation")
  - ``confidence`` (float): Recognition confidence
  - Published by: ``SpeechToTextService._publish_recognition_result()``
  - Consumed by: ``DictationCoordinator._handle_dictation_text()``
  - Characteristics: Higher accuracy, slower processing than command recognition

Fallback processing for non-speech audio:

- **ProcessAudioChunkForSoundRecognitionEvent** (Priority: HIGH)

  - ``audio_chunk`` (bytes): Raw PCM audio data that failed STT recognition
  - ``sample_rate`` (int): Sample rate in Hz (default 16000)
  - Published by: ``SpeechToTextService`` when STT returns empty text
  - Consumed by: ``StreamlinedSoundService._handle_audio_chunk()``
  - Purpose: Enables custom sound recognition (claps, whistles, etc.) as command triggers

- **CustomSoundRecognizedEvent** (Priority: HIGH)

  - ``label`` (str): Recognized sound identifier (e.g., "whistle_1", "clap_2")
  - ``confidence`` (float): Recognition confidence score (0.0-1.0)
  - ``mapped_command`` (Optional[str]): Command phrase mapped to this sound, if any
  - Published by: ``StreamlinedSoundService._handle_audio_chunk()``
  - Consumed by: ``CentralizedCommandParser._handle_sound_recognized()``
  - Storage: Sound-to-command mappings persisted via ``SoundMappingsData``

Prediction bypass for ultra-low latency:

- **MarkovPredictionEvent** (Priority: CRITICAL)

  - ``predicted_command`` (str): Predicted command text based on historical patterns
  - ``confidence`` (float): Prediction confidence (0.0-1.0, typically >0.7 to publish)
  - ``audio_id`` (int): Identifier for the audio bytes that triggered prediction
  - Published by: ``MarkovCommandService._handle_audio_detected_fast_track()``
  - Consumed by: ``CentralizedCommandParser._handle_markov_prediction()``
  - Result: Directly executes predicted command and stores prediction for deduplication
  - Latency reduction: 200-800ms faster than standard STT processing

- **MarkovPredictionFeedbackEvent** (Priority: NORMAL)

  - ``predicted_command`` (str): The command that was predicted by Markov
  - ``actual_command`` (str): The command that was actually recognized by STT or Sound
  - ``was_correct`` (bool): True if prediction matched actual command
  - ``source`` (str): Source of actual command ("stt" or "sound")
  - Published by: ``CentralizedCommandParser._send_markov_feedback()``
  - Consumed by: ``MarkovCommandService._handle_prediction_feedback()``
  - Purpose: Allows Markov service to learn from correct/incorrect predictions and enter cooldown on errors

Command Processing Event Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command processing pipeline transforms recognized text into structured command objects through hierarchical parsing, then routes them to specialized execution services. This architecture decouples text recognition from command execution, enabling flexible command interpretation and consistent error handling.

.. mermaid::

   graph TB
   %% Markov Prediction Flow
   AudioDetect[AudioDetectedEvent] -->|consumed by| Markov[MarkovCommandService]
   Markov -->|publishes| MarkovPred[MarkovPredictionEvent]

   %% Command Recognition Flow
   CmdText[CommandTextRecognizedEvent] -->|consumed by| Parser[CentralizedCommandParser]
   CustomSound[CustomSoundRecognizedEvent] -->|consumed by| Parser

   %% Markov Prediction Flow
   MarkovPred -->|consumed by| Parser

   %% Feedback Loop
   Parser -->|publishes| Feedback[MarkovPredictionFeedbackEvent]
   Feedback -->|consumed by| Markov

   %% Command Execution Flow
   Parser -->|publishes| DictCmd[DictationCommandParsedEvent]
   Parser -->|publishes| AutoCmd[AutomationCommandParsedEvent]
   Parser -->|publishes| MarkCmd[MarkCommandParsedEvent]
   Parser -->|publishes| GridCmd[GridCommandParsedEvent]
   Parser -->|publishes| SoundCmd[SoundCommandParsedEvent]

   DictCmd -->|consumed by| DictCoord[DictationCoordinator]
   AutoCmd -->|consumed by| AutoSvc[AutomationService]
   MarkCmd -->|consumed by| MarkSvc[MarkService]
   GridCmd -->|consumed by| GridSvc[GridService]
   SoundCmd -->|consumed by| SoundSvc[StreamlinedSoundService]

   Parser -->|publishes on failure| NoMatch[CommandNoMatchEvent]
   Parser -->|publishes on error| ParseError[CommandParseErrorEvent]

**Command Event Details and Parsing Hierarchy**:

The ``CentralizedCommandParser`` implements a hierarchical parsing strategy, checking commands in order of priority: dictation controls first (highest priority for interruption), then marks, grids, automation, and finally sound commands. This ordering ensures that critical state-changing commands like "stop dictation" are never misinterpreted as automation commands.

**Command History Management**:

The parser delegates command history tracking to ``CommandHistoryManager`` for fast, zero-I/O command recording:

1. **Initialization**: ``CommandHistoryManager`` loads existing command history from storage
2. **Runtime**: Commands appended to in-memory list via ``record_command()`` (microsecond operation)
3. **Shutdown**: History manager writes complete history back to storage (single operation)
4. **Training**: Markov service accesses command history for training updates

This delegation separates history tracking concerns from parsing logic while maintaining performance.

**Markov Prediction Deduplication**:

The parser handles Markov prediction deduplication internally to avoid duplicate command execution:

1. **Prediction Storage**: Stores predictions in ``_recent_predictions`` dict with 500ms time window
2. **Comparison**: When STT/sound recognizes a command, checks for recent prediction match
3. **Feedback**: Sends ``MarkovPredictionFeedbackEvent`` with accuracy results
4. **Cleanup**: Automatically removes predictions outside the time window

This approach keeps deduplication logic close to the parsing workflow where it belongs.

Input events to the parser:

- **CommandTextRecognizedEvent** (described above)

  - Consumed by: ``CentralizedCommandParser._handle_text_recognized()``
  - Processing: Normalized to lowercase, stripped of whitespace, then matched against command patterns
  - Duplicate detection: Filtered using 1-second time window to prevent double-execution

- **CustomSoundRecognizedEvent** (described above)

  - Consumed by: ``CentralizedCommandParser._handle_sound_recognized()``
  - Processing: Sound label mapped to command phrase via ``_sound_to_command_mapping`` dict, then parsed as text
  - Mapping storage: Persisted in ``SoundMappingsData`` via ``StorageService``

Output command events (all extend ``BaseCommandEvent`` with source and context fields):

- **DictationCommandParsedEvent** (Priority: NORMAL)

  - ``command`` (DictationCommandType): Union of ``DictationStartCommand``, ``DictationStopCommand``, ``DictationTypeCommand``, or ``DictationSmartStartCommand``
  - ``source`` (Optional[str]): Origin of command (e.g., "speech", "sound:whistle_1")
  - ``context`` (Optional[Dict]): Additional metadata for command execution
  - Command types:

    - ``DictationStartCommand``: Activates continuous dictation mode
    - ``DictationStopCommand``: Deactivates any active dictation mode
    - ``DictationTypeCommand``: Activates type dictation (single phrase mode)
    - ``DictationSmartStartCommand``: Activates smart dictation with LLM processing

  - Published by: ``CentralizedCommandParser._publish_command_event()``
  - Consumed by: ``DictationCoordinator._handle_dictation_command()``
  - Pattern matching: Exact matches on "start dictation", "stop dictation", "type dictation", "smart dictation"

- **AutomationCommandParsedEvent** (Priority: NORMAL)

  - ``command`` (AutomationCommandType): Union of ``ExactMatchCommand`` or ``ParameterizedCommand``
  - Command structure:

    - ``command_key`` (str): Unique identifier for the command (e.g., "click", "scroll down")
    - ``action_type`` (str): Automation action ("hotkey", "key", "key_sequence", "click", "scroll")
    - ``action_value`` (str): Action-specific parameter (e.g., "ctrl+c" for hotkey)
    - ``count`` (int): Repeat count for parameterized commands (e.g., "scroll down 5")

  - Published by: ``CentralizedCommandParser._publish_command_event()``
  - Consumed by: ``AutomationService._handle_automation_command()``
  - Storage: Command definitions loaded from ``AutomationCommandRegistry`` and ``CommandsData``
  - Pattern matching: Exact phrase matching for ``ExactMatchCommand``, regex "command [number]" for ``ParameterizedCommand``

- **MarkCommandParsedEvent** (Priority: NORMAL)

  - ``command`` (MarkCommandType): Union of mark command variants
  - Command types:

    - ``MarkCreateCommand(label: str, x: int, y: int)``: Creates new positional mark at current cursor
    - ``MarkExecuteCommand(label: str)``: Navigates to saved mark position
    - ``MarkDeleteCommand(label: str)``: Removes mark from storage
    - ``MarkVisualizeCommand()``: Toggles mark visualization overlay
    - ``MarkResetCommand()``: Clears all marks
    - ``MarkVisualizeCancelCommand()``: Hides mark visualization

  - Published by: ``CentralizedCommandParser._publish_command_event()``
  - Consumed by: ``MarkService._handle_mark_command()``
  - Pattern matching: "mark <label>", "mark delete <label>", "visualize marks", etc.
  - Storage: Mark coordinates persisted via ``MarksData`` with caching for instant retrieval

- **GridCommandParsedEvent** (Priority: NORMAL)

  - ``command`` (GridCommandType): Union of grid command variants
  - Command types:

    - ``GridShowCommand(num_rects: Optional[int])``: Displays overlay grid for mouse navigation
    - ``GridSelectCommand(selected_number: int)``: Clicks specified grid cell
    - ``GridCancelCommand()``: Hides active grid overlay

  - Published by: ``CentralizedCommandParser._publish_command_event()``
  - Consumed by: ``GridService._handle_grid_command()``
  - Pattern matching: "show grid [number]", "select [number]", "cancel grid"
  - Integration: Grid uses ``ClickTrackerService`` click statistics for intelligent cell prioritization

- **SoundCommandParsedEvent** (Priority: NORMAL)

  - ``command`` (SoundCommandType): Union of sound management commands
  - Command types:

    - ``SoundTrainCommand(label: str, num_samples: int)``: Initiates sound training session
    - ``SoundDeleteCommand(label: str)``: Removes trained sound and mappings
    - ``SoundMapCommand(label: str, command_phrase: str)``: Maps sound to command phrase
    - ``SoundResetAllCommand()``: Clears all trained sounds and mappings
    - ``SoundListAllCommand()``: Requests list of all trained sounds

  - Published by: ``CentralizedCommandParser._publish_command_event()``
  - Consumed by: ``StreamlinedSoundService`` methods
  - Pattern matching: "train sound <label>", "delete sound <label>", "map sound <label> to <command>"

Error and no-match events:

- **CommandNoMatchEvent** (Priority: NORMAL, extends BaseCommandEvent)

  - ``attempted_parsers`` (List[str]): List of parser methods that failed to match
  - Published by: ``CentralizedCommandParser._handle_text_recognized()`` when no parser matches input
  - Purpose: Logging and debugging unrecognized commands for vocabulary expansion

- **CommandParseErrorEvent** (Priority: NORMAL, extends BaseCommandEvent)

  - ``error_message`` (str): Detailed error description
  - ``attempted_parser`` (Optional[str]): Parser method that encountered the error
  - Published by: ``CentralizedCommandParser`` exception handlers
  - Purpose: Graceful error handling and debugging of malformed commands

Dictation Event Flow
~~~~~~~~~~~~~~~~~~~~

Dictation mode enables continuous speech-to-text input with three operational variants: standard (immediate text input), type (single-phrase input), and smart (LLM-enhanced input). The ``DictationCoordinator`` orchestrates state transitions, manages session lifecycle, and coordinates between STT, LLM, and text input services.

.. mermaid::

   graph TB
   DictCmd[DictationCommandParsedEvent] -->|consumed by| DictCoord[DictationCoordinator]
   DictText[DictationTextRecognizedEvent] -->|consumed by| DictCoord

   DictCoord -->|publishes on smart mode start| SmartStart[SmartDictationStartedEvent]
   DictCoord -->|publishes raw text for display| TextDisplay[SmartDictationTextDisplayEvent]
   DictCoord -->|publishes on smart mode stop| SmartStop[SmartDictationStoppedEvent]

   SmartStop -->|triggers internal processing| DictCoord
   DictCoord -->|publishes| LLMStart[LLMProcessingStartedEvent]

   LLMStart -->|consumed by| LLMService[LLMService]
   LLMService -->|publishes streaming tokens| LLMToken[LLMTokenGeneratedEvent]
   LLMService -->|publishes on completion| LLMComplete[LLMProcessingCompletedEvent]
   LLMService -->|publishes on error| LLMFail[LLMProcessingFailedEvent]

   LLMComplete -->|consumed by| DictCoord
   LLMFail -->|consumed by| DictCoord

   DictCoord -->|uses for text input| TextService[TextInputService]
   TextService -->|executes| SysInput[System Text Input via pyautogui]

   DictCoord -->|publishes state changes| StatusEvent[DictationStatusChangedEvent]
   DictCoord -->|publishes to disable command processing| DisableOthers[DictationModeDisableOthersEvent]

**Dictation Event Details**:

- **DictationCommandParsedEvent** (described in previous section)

  - Consumed by: ``DictationCoordinator._handle_dictation_command()``
  - Processing: Transitions dictation state machine (INACTIVE → STANDARD/TYPE/SMART → INACTIVE)
  - State tracking: ``DictationCoordinator`` maintains ``active_mode`` (DictationMode enum) and ``_current_session``

- **DictationStatusChangedEvent** (Priority: LOW)

  - ``is_active`` (bool): Whether any dictation mode is currently active
  - ``mode`` (str): Current mode ("inactive", "continuous", "type", "smart")
  - ``show_ui`` (bool): Whether to display dictation UI indicator
  - Published by: ``DictationCoordinator._publish_status()``
  - Purpose: Synchronizes UI indicators and system state across application

- **DictationModeDisableOthersEvent** (Priority: CRITICAL)

  - ``dictation_mode_active`` (bool): Whether dictation mode is active
  - ``dictation_mode`` (str): Type of dictation mode active
  - Published by: ``DictationCoordinator`` on mode activation/deactivation
  - Consumed by: ``SpeechToTextService._handle_dictation_mode_change()``
  - Effect: When active, command audio processing only checks for amber stop words, suppressing normal command recognition

- **DictationTextRecognizedEvent** (described in Audio Event Flow section)

  - Consumed by: ``DictationCoordinator._handle_dictation_text()``
  - Processing varies by mode:

    - **Standard/Type mode**: Text immediately sent to ``TextInputService.type_text()``
    - **Smart mode**: Text accumulated in ``DictationSession.accumulated_text`` buffer

- **SmartDictationStartedEvent** (Priority: NORMAL)

  - ``mode`` (str): Always "smart"
  - Published by: ``DictationCoordinator._activate_smart_dictation()``
  - Purpose: Signals UI to prepare for smart dictation session

- **SmartDictationTextDisplayEvent** (Priority: HIGH)

  - ``text`` (str): Cleaned, formatted text for real-time display in UI
  - Published by: ``DictationCoordinator._handle_dictation_text()`` during smart dictation
  - Purpose: Provides real-time feedback to user before LLM processing

- **SmartDictationStoppedEvent** (Priority: NORMAL)

  - ``mode`` (str): Always "smart"
  - ``raw_text`` (str): Complete accumulated text before LLM processing
  - Published by: ``DictationCoordinator._deactivate_smart_dictation()``
  - Processing: Triggers LLM processing pipeline

LLM processing events:

- **LLMProcessingStartedEvent** (Priority: NORMAL)

  - ``raw_text`` (str): Original dictated text to be enhanced
  - ``agentic_prompt`` (str): LLM instruction prompt (e.g., "Fix grammar and improve clarity")
  - ``session_id`` (Optional[str]): Session identifier for correlation
  - Published by: ``DictationCoordinator._start_llm_processing()``

- **LLMTokenGeneratedEvent** (Priority: HIGH)

  - ``token`` (str): Individual token generated during LLM streaming
  - Published by: ``LLMService.process_dictation_streaming()`` via ``token_callback``
  - Frequency: Published for each token as generated (typically 10-50 tokens/second)

- **LLMProcessingCompletedEvent** (Priority: NORMAL)

  - ``processed_text`` (str): LLM-enhanced text with grammar corrections and clarity improvements
  - ``agentic_prompt`` (str): The agentic prompt that was used
  - Published by: ``LLMService._publish_completed()``
  - Consumed by: ``DictationCoordinator._handle_llm_completed()``
  - Processing: Completed text sent to ``TextInputService.type_text()`` for system input

- **LLMProcessingFailedEvent** (Priority: NORMAL)

  - ``error_message`` (str): Detailed error description
  - ``original_text`` (str): The text that failed processing
  - Published by: ``LLMService._publish_failed()``
  - Fallback: Original raw text is typed instead of processed text

UI Integration Event Flow
~~~~~~~~~~~~~~~~~~~~~~~~~

UI integration services manage visual overlays (grids, marks) and mouse interaction tracking. These services bridge voice commands with on-screen actions, providing visual feedback and intelligent click positioning.

**Grid Events:**

.. mermaid::

   graph TB
   GridCmd[GridCommandParsedEvent] -->|consumed by| GridSvc[GridService]
   GridSvc -->|publishes| ShowReq[ShowGridRequestEventData]
   GridSvc -->|publishes| ClickReq[ClickGridCellRequestEventData]
   GridSvc -->|publishes| Visibility[GridVisibilityChangedEventData]

   ShowReq -->|consumed by| UI[UI GridWindow]
   ClickReq -->|consumed by| UI

**Mark Events:**

.. mermaid::

   graph TB
   MarkCmd[MarkCommandParsedEvent] -->|consumed by| MarkSvc[MarkService]
   MarkSvc -->|publishes| Created[MarkCreatedEventData]
   MarkSvc -->|publishes| Deleted[MarkDeletedEventData]
   MarkSvc -->|publishes| VisState[MarkVisualizationStateChangedEventData]

   Created -->|consumed by| UI[UI MarkOverlay]
   Deleted -->|consumed by| UI
   VisState -->|consumed by| UI

**Click Tracking Events:**

.. mermaid::

   graph TB
   ClickReq[PerformMouseClickEventData] -->|consumed by| Tracker[ClickTrackerService]
   Tracker -->|publishes| Logged[ClickLoggedEventData]
   Tracker -->|provides data via| Stats[get_click_statistics]
   Stats -->|used by| GridSvc[GridService]
   GridSvc -->|publishes| Counts[ClickCountsForGridEventData]

**UI Integration Event Details**:

Grid navigation events:

- **GridCommandParsedEvent** → ``GridService._handle_grid_command()``

- **ShowGridRequestEventData** (Priority: NORMAL)

  - ``rows`` (int): Number of grid rows (default from config)
  - ``columns`` (int): Number of grid columns (default from config)
  - Published by: ``GridService._show_grid()``
  - Consumed by: ``GridWindow`` UI component

- **ClickGridCellRequestEventData** (Priority: CRITICAL)

  - ``cell_label`` (str): Grid cell identifier
  - Published by: ``GridService._select_cell()``
  - Consumed by: ``GridWindow`` UI component

- **GridVisibilityChangedEventData** (Priority: LOW)

  - ``visible`` (bool): Whether grid is currently visible
  - Published by: ``GridService`` on state changes

Mark navigation events:

- **MarkCreatedEventData** (Priority: NORMAL)

  - ``name`` (str): Unique mark identifier
  - ``x`` (int): Screen X coordinate
  - ``y`` (int): Screen Y coordinate
  - Published by: ``MarkService._execute_mark_command()``
  - Storage: Persisted via ``MarksData`` with 5-minute cache TTL

- **MarkDeletedEventData** (Priority: NORMAL)

  - ``name`` (str): Deleted mark identifier
  - Published by: ``MarkService._execute_mark_command()``

- **MarkVisualizationStateChangedEventData** (Priority: LOW)

  - ``visible`` (bool): Whether mark visualization overlay is visible
  - ``marks`` (List): Current marks to display
  - Published by: ``MarkService._execute_mark_command()``

Click tracking events:

- **PerformMouseClickEventData** (Priority: CRITICAL)

  - ``x`` (int): Screen X coordinate of click
  - ``y`` (int): Screen Y coordinate of click
  - ``source`` (str): Click origin ("grid", "mark", "automation", "unknown")
  - Published by: Services performing clicks
  - Consumed by: ``ClickTrackerService._handle_click()``

- **ClickLoggedEventData** (Priority: LOW)

  - ``x`` (int): Screen X coordinate
  - ``y`` (int): Screen Y coordinate
  - ``timestamp`` (float): Unix timestamp of click
  - Published by: ``ClickTrackerService`` after persisting

Helper Services Architecture
----------------------------

The Vocalance architecture includes lightweight helper services that provide shared functionality to multiple components, promoting code reuse and separation of concerns.

ProtectedTermsValidator
~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Centralized validation for protected/reserved terms across all command types.

**Features**:

- Single source of truth for what terms are protected (system commands, marks, sounds, triggers)
- Caching with 60-second TTL for performance
- Validates terms for commands, marks, and sounds consistently
- Aggregates protected terms from multiple sources: automation commands, system triggers, active marks, trained sounds

**Used by**:

- ``CommandManagementService``: Validates custom command phrases before creation/update
- ``MarkService``: Validates mark labels before creation

**Methods**:

- ``get_all_protected_terms()``: Returns cached set of all protected terms
- ``is_term_protected(term)``: Checks if a specific term is protected
- ``validate_term(term, exclude_term=None)``: Full validation with optional exclusion for updates
- ``invalidate_cache()``: Forces cache refresh on next access

CommandActionMapProvider
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Centralized provider for automation command action maps.

**Features**:

- Single source of truth for building complete command maps
- Merges custom commands with default commands
- Applies phrase overrides consistently
- Eliminates duplicate action map building logic

**Used by**:

- ``CommandManagementService``: Gets action map for validation and retrieval
- ``CentralizedCommandParser``: Gets action map for command parsing

**Methods**:

- ``get_action_map()``: Returns complete dictionary mapping normalized phrases to ``AutomationCommand`` objects

CommandHistoryManager
~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Manages command execution history with in-memory accumulation.

**Features**:

- Records commands to in-memory buffer (fast, zero I/O during session)
- Persists full history to storage at shutdown
- Thread-safe history access via locking
- Clean separation from parsing logic

**Used by**:

- ``CentralizedCommandParser``: Records executed commands for Markov training

**Methods**:

- ``initialize()``: Loads existing history from storage
- ``record_command(command, source)``: Adds command to in-memory history
- ``get_recent_history(count)``: Returns N most recent commands
- ``get_full_history()``: Returns complete command history
- ``shutdown()``: Persists accumulated history to storage

**Design Benefits**:

These helper services provide:

1. **Code Reuse**: Shared functionality eliminates duplication across services
2. **Single Responsibility**: Each helper has one clear job
3. **Testability**: Small, focused components are easy to test in isolation
4. **Maintainability**: Changes to shared logic happen in one place
5. **Performance**: Caching and efficient algorithms in centralized locations

Optimizations and Ancillary Services
------------------------------------

Markov Chain Prediction System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MarkovCommandService`` implements a sophisticated predictive command execution system that dramatically reduces interaction latency by anticipating user commands before speech recognition completes. By analyzing historical command sequences, the predictor achieves 200-800ms latency reduction for frequently-used command patterns.

**Core Concept**:

Traditional voice command systems follow: audio capture → STT processing → command parsing → execution. The Markov service breaks this constraint by predicting the next command based on previous commands, publishing predictions immediately when audio is detected, before STT begins processing.

**Multi-Order Markov Chain Architecture**:

The predictor implements a backoff strategy using 2nd through 4th order Markov chains:

- **2nd Order**: Predicts next command based on previous 1 command
- **3rd Order**: Uses previous 2 commands for context
- **4th Order**: Uses previous 3 commands for maximum context

Data structures:

- ``_transition_counts``: Nested dict structure ``{order: {context_tuple: Counter}}``
- ``_command_history``: ``deque`` with ``maxlen=max_order`` (typically 4)

**Prediction Algorithm**:

When ``AudioDetectedEvent`` fires:

1. Check if prediction cooldown has expired (50ms minimum between predictions)
2. Extract context tuple from ``_command_history``
3. Start with highest order (4th) and query ``_transition_counts``
4. Calculate confidence: ``count(predicted_command) / sum(all_counts_for_context)``
5. If confidence >= threshold (typically 0.7), publish ``MarkovPredictionEvent``
6. If confidence too low, backoff to lower order (4th → 3rd → 2nd)

**Performance Optimizations**:

- **In-Memory History**: Commands accumulated in memory during session, written once at shutdown
- **Context Deque**: Fixed-size deque for O(1) append
- **Hash-Based Lookups**: O(1) transition lookups
- **Confidence Thresholding**: Only publishes predictions above 70% confidence
- **Prediction Cooldown**: 50ms minimum between predictions
- **Deduplication Window**: 500ms window prevents duplicate execution

**Integration Points**:

Event subscriptions:

- ``AudioDetectedEvent`` → ``_handle_audio_detected_fast_track()``: Trigger prediction
- ``MarkovPredictionFeedbackEvent`` → ``_handle_prediction_feedback()``: Receive accuracy feedback

Event publications:

- ``MarkovPredictionEvent``: Published to ``CentralizedCommandParser``

Storage Service Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``StorageService`` serves as the foundational storage engine, providing low-level file operations with caching, atomic writes, and thread safety. All storage operations use type-safe Pydantic models for data validation.

**Core Components**:

- ``_cache`` (Dict[str, CacheEntry]): In-memory cache with 5-minute TTL

- ``_path_map`` (Dict[Type[StorageData], str]): Maps storage model types to filesystem paths

  - ``MarksData`` → ``{marks_dir}/marks.json``
  - ``SettingsData`` → ``{settings_dir}/user_settings.json``
  - ``CommandsData`` → ``{settings_dir}/custom_commands.json``
  - ``GridClicksData`` → ``{click_tracker_dir}/click_history.json``
  - ``SoundMappingsData`` → ``{sound_model_dir}/sound_mappings.json``
  - ``AgenticPromptsData`` → ``{user_data_root}/dictation/agentic_prompts.json``
  - ``CommandHistoryData`` → ``{command_history_dir}/command_history.json``

- ``_lock`` (threading.RLock): Reentrant lock for thread-safe cache access
- ``_executor`` (ThreadPoolExecutor): 2-worker thread pool for async I/O

**Read Operation Flow**:

The ``read(model_type)`` method implements async cache-through pattern:

1. Acquire ``_lock`` and check ``_cache`` for existing entry
2. If cache hit and not expired (< 5 minutes old):

   - Return cached data immediately (typical latency: <0.1ms)

3. If cache miss or expired:

   - Determine filepath from ``_path_map[model_type]``
   - Submit ``_read_json()`` to ``_executor`` for async execution
   - Validate with Pydantic model
   - Store in ``_cache`` for future reads
   - Return data (typical latency: 1-5ms for small files)

4. If file doesn't exist, return default model instance

**Write Operation Flow**:

The ``write(data)`` method implements atomic writes:

1. Determine filepath from ``_path_map[type(data)]``
2. Submit ``_write_json()`` to ``_executor`` for async execution
3. Perform atomic write:

   - Serialize data to JSON
   - Write to temporary file: ``{filepath}.tmp.{uuid}``
   - Atomically rename temp file to final filepath
   - Ensures file is never in partial/corrupt state

4. If write succeeds:

   - Update ``_cache`` with new data
   - Return ``True``

5. If write fails:

   - Log error
   - Return ``False``

**Storage Data Models**:

Type-safe Pydantic models define the schema:

- **MarksData**: Voice-navigable screen position bookmarks ``marks: Dict[str, Coordinate]``
- **SettingsData**: User configuration overrides ``user_overrides: Dict[str, Dict]``
- **CommandsData**: Custom automation commands ``custom_commands: Dict[str, AutomationCommand]``
- **GridClicksData**: Click pattern tracking ``clicks: List[GridClickEvent]``
- **AgenticPromptsData**: Smart dictation LLM prompts ``prompts: List[AgenticPrompt]``
- **SoundMappingsData**: Sound recognition mappings ``mappings: Dict[str, str]``
- **CommandHistoryData**: Markov training data ``entries: List[CommandHistoryEntry]``

**Performance Characteristics**:

- **Cached Read**: 0.05-0.1ms (dictionary lookup only)
- **Uncached Read**: 1-5ms (JSON parse + disk read)
- **Write**: 2-10ms (JSON serialize + atomic file write)
- **Cache Hit Rate**: 90-95% during normal operation
- **Thread Safety**: All operations protected by reentrant lock
- **Memory Footprint**: ~10-50KB for typical cache

**Error Handling**:

- **Read Failures**: Return default model instance, log error
- **Write Failures**: Return False to caller, preserve existing data
- **Corrupt JSON**: Catch validation exceptions, return default model
- **Missing Files**: Treat as missing, create directory on first write
- **Atomic Writes**: Temp file + rename ensures data integrity

**Event Integration**:

Storage changes trigger events for real-time synchronization:

- ``CommandMappingsUpdatedEvent`` → Triggers ``CentralizedCommandParser`` reload
- ``SoundToCommandMappingUpdatedEvent`` → Updates sound recognition mappings
- ``MarkCreatedEventData`` / ``MarkDeletedEventData`` → Updates UI overlays
- ``AgenticPromptUpdatedEvent`` → Synchronizes LLM prompt

Grid Optimization and Click Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ClickTrackerService`` implements an intelligent grid optimization system that learns from user behavior to prioritize frequently-clicked screen regions.

**Motivation**:

Voice-controlled grid navigation typically assigns cell numbers sequentially. Frequently-accessed areas might receive high cell numbers. The ``ClickTrackerService`` assigns lower cell numbers (1, 2, 3...) to frequently-clicked regions, enabling single-syllable access to common targets.

**Architecture**:

1. **Click Recording**: All services publish ``PerformMouseClickEventData`` when performing clicks
2. **Debounced Logging**: Batches clicks for 2 seconds before persisting
3. **Storage Persistence**: Clicks saved to ``click_history.json`` via ``GridClicksData``
4. **Statistics Generation**: ``get_click_statistics()`` analyzes history to produce heat map
5. **Grid Integration**: ``GridService`` requests statistics and reorders cells accordingly

**Click Recording and Debouncing**:

The ``_handle_click(event)`` method implements debounced logging:

- Append click to in-memory buffer
- If write timer not active, start 2-second countdown
- When timer expires:

  - Batch all pending clicks
  - Write to storage in single operation
  - Publish ``ClickLoggedEventData``
  - Clear buffer

Debouncing reduces disk writes from potentially hundreds per minute to ~30 per minute.

**Heat Map Generation Algorithm**:

The ``get_click_statistics(grid_rows, grid_columns)`` method:

1. Load click history from storage
2. Calculate screen dimensions via ``pyautogui.size()``
3. Compute cell dimensions
4. For each grid cell:

   - Count clicks within boundaries
   - Create rectangle dict with click_count

5. Sort rectangles by ``click_count`` descending
6. Return sorted list

**Grid Cell Reordering**:

``GridService`` consumes statistics and reorders cells:

1. Request statistics for current grid dimensions
2. Assign cell numbers 1-N based on sort order
3. Display grid overlay with reordered numbers
4. When user says "select three", click the 3rd-most-frequently-clicked region

**Performance Characteristics**:

- **Total clicks recorded**: Historical tracking
- **Clicks per grid cell**: Average, min, max statistics
- **Time range**: Recorded click history span
- **Storage file size**: Persisted data size

**Configuration**:

- ``click_history_retention_days``: How long to retain (default: 30 days)
- ``debounce_interval``: Click batching window (default: 2 seconds)

LLM Integration and Prompt Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LLM integration layer enables AI-powered text enhancement for smart dictation mode, transforming raw speech transcription into polished text through grammar correction and clarity improvements.

**LLMService Architecture**:

The ``LLMService`` wraps local LLM models (via ``llama-cpp-python``) to provide dictation enhancement while maintaining privacy. The service runs entirely locally using quantized models optimized for CPU inference.

**Model Management**:

1. ``LLMModelDownloader`` checks for model file at configured path
2. If missing, downloads from HuggingFace
3. ``LLMService.initialize()`` loads model via ``llama_cpp.Llama``
4. Model loaded into memory (~2-4GB for quantized 3-7B models)

**Streaming Inference Pipeline**:

The ``process_dictation_streaming(raw_text, agentic_prompt, token_callback)`` method:

1. Construct prompt: ``f"{agentic_prompt}\n\nOriginal: {raw_text}\n\nImproved:"``
2. Invoke ``llm.create_completion(prompt, stream=True, max_tokens=512)``
3. For each generated token:

   - Extract token text
   - Call ``token_callback(token)`` to publish ``LLMTokenGeneratedEvent``
   - Append to accumulated output buffer

4. After stream completes:

   - Validate output via ``_validate_output()``
   - Clean output via ``_clean_output()``
   - Publish ``LLMProcessingCompletedEvent``

5. On error:

   - Publish ``LLMProcessingFailedEvent``
   - ``DictationCoordinator`` falls back to typing raw text

**Output Validation**:

The ``_validate_output(output, original)`` method:

- **Minimum length**: Output must be ≥3 characters
- **Length ratio**: ``0.3 ≤ len(output)/len(original) ≤ 3.0``
- **Repetition detection**: Unique words must be ≥50% of total

**AgenticPromptService**:

Manages a library of user-defined prompts that guide LLM behavior.

**Prompt Storage**:

::

    {
      "prompts": [
        {
          "id": "uuid-1234",
          "name": "Grammar Fix",
          "text": "Fix grammar and punctuation. Keep meaning exactly the same."
        }
      ],
      "current_prompt_id": "uuid-1234"
    }

Key methods:

- ``get_current_prompt()`` → ``str``: Returns active prompt text
- ``add_prompt(name, text)`` → ``Dict``: Creates new prompt with UUID
- ``delete_prompt(prompt_id)`` → ``bool``: Removes prompt
- ``set_prompt(prompt_id)`` → ``bool``: Sets active prompt

**Performance Characteristics**:

- **Model load time**: 2-5 seconds (one-time at startup)
- **Inference latency**: 10-50 tokens/second (CPU), 50-200 tokens/second (GPU)
- **Memory usage**: 2-4GB for quantized 3-7B models
- **Context window**: 2048-4096 tokens

Command Management and Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CommandManagementService** orchestrates command lifecycle:

- Receives command management events from UI
- Delegates storage operations to ``CommandsData`` via ``StorageService``
- Validates command names against reserved terms
- Publishes ``CommandMappingsUpdatedEvent`` for live reloads

**AutomationCommandRegistry** provides built-in command vocabulary:

- Defines default automation commands
- Examples: ``{"click": AutomationCommand(..., action_type="click", action_value="left")}``
- Loaded during ``CentralizedCommandParser`` initialization
- Merged with custom commands to form complete action map

**Command Definition Structure**:

::

    {
      "command_phrase": AutomationCommand(
        command_key="command_phrase",
        action_type="hotkey",
        action_value="ctrl+c"
      )
    }

Action types supported:

- ``"hotkey"``: Sends keyboard combination via ``pyautogui.hotkey()``
- ``"key"``: Sends single key press
- ``"key_sequence"``: Sends key sequence
- ``"click"``: Performs mouse click
- ``"scroll"``: Scrolls by pixel amount

**Dynamic Command Updates**:

When user adds command:

1. UI publishes command management event
2. ``CommandManagementService`` receives event
3. Validates phrase against reserved terms
4. Delegates to ``StorageService.write(CommandsData)``
5. Storage writes to ``custom_commands.json``
6. Notifies ``CentralizedCommandParser``
7. New command immediately available for recognition

Performance Monitoring and Analytics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Service-Level Performance Metrics**:

Each service exposes performance data through dedicated methods:

- ``MarkService``: Coordinate cache hit rates, average lookup latency
- ``ClickTrackerService``: Total clicks, heat map coverage, storage file size
- ``MarkovCommandService``: Prediction accuracy, confidence distributions
- ``StorageService``: Cache entries, total accesses, hit rate percentage

**Event-Based Performance Tracking**:

Critical operations publish timing and status events:

- **CommandExecutedStatusEvent**: Published by ``AutomationService`` after command execution

  - Tracks success/failure rates
  - Measures end-to-end command latency
  - Identifies problematic commands

- **STT Processing Events**: Published by ``SpeechToTextService``

  - Measures STT processing time per engine
  - Tracks audio segment sizes
  - Enables comparison of command vs dictation mode performance

- **LLMProcessingCompletedEvent**: Includes processing time for smart dictation

**System-Wide Performance Characteristics** (typical values):

- **Audio Detection Latency**: 50-200ms
- **STT Command Recognition**: 150-400ms (Vosk)
- **STT Dictation Recognition**: 500-2000ms (Whisper)
- **Markov Prediction**: 1-5ms
- **Command Parsing**: 0.5-2ms
- **Command Execution**: 10-100ms
- **Storage Read (cached)**: 0.05-0.1ms
- **Storage Read (uncached)**: 1-5ms
- **End-to-End Command Latency**: 200-500ms (Markov), 400-800ms (standard STT)

**Optimization Strategies Employed**:

- **Adaptive VAD Thresholds**: ``Recorder`` adjusts energy thresholds based on ambient noise
- **Multi-Level Caching**: Storage layer implements memory cache with TTL-based expiration
- **Batch Processing**: Markov service writes in batches, ``ClickTrackerService`` batches clicks every 2 seconds
- **Predictive Loading**: ``MarkService`` pre-loads all marks into cache during initialization
- **Lazy Initialization**: LLM model only loaded when smart dictation first used
- **Duplicate Filtering**: Command parser prevents double-execution within 1-second window
