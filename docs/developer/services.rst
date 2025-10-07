Services Architecture
=====================

This document provides a comprehensive guide to the services architecture of Iris, detailing module dependencies, event flows, and optimization strategies.

Module Dependency Graph
-----------------------

.. mermaid::

   graph TB
    %% Core Infrastructure
    EventBus[EventBus]
    GlobalAppConfig[GlobalAppConfig]

    %% Audio Services
    SimpleAudioService[SimpleAudioService]
    AudioRecorder[AudioRecorder]
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

    %% UI Integration Services
    GridService[GridService]
    ClickTrackerService[ClickTrackerService]
    MarkService[MarkService]

    %% Storage & Persistence
    UnifiedStorageService[UnifiedStorageService]
    StorageAdapterFactory[StorageAdapterFactory]
    SettingsService[SettingsService]
    CommandManagementService[CommandManagementService]

    %% Optimization Services
    MarkovCommandPredictor[MarkovCommandPredictor]
    SmartTimeoutManager[SmartTimeoutManager]
    ProtectedTermsService[ProtectedTermsService]

    %% Dependencies
    EventBus --> SimpleAudioService
    EventBus --> STTService
    EventBus --> CentralizedCommandParser
    EventBus --> AutomationService
    EventBus --> DictationCoordinator
    EventBus --> GridService
    EventBus --> MarkService
    EventBus --> MarkovCommandPredictor

    GlobalAppConfig --> SimpleAudioService
    GlobalAppConfig --> STTService
    GlobalAppConfig --> CentralizedCommandParser
    GlobalAppConfig --> AutomationService
    GlobalAppConfig --> DictationCoordinator
    GlobalAppConfig --> GridService
    GlobalAppConfig --> MarkService
    GlobalAppConfig --> MarkovCommandPredictor

    %% Audio Flow
    SimpleAudioService --> AudioRecorder
    SimpleAudioService --> EventBus
    AudioRecorder --> EventBus

    STTService --> VoskSTT
    STTService --> WhisperSTT
    STTService --> StreamlinedSoundService

    StreamlinedSoundService --> StreamlinedSoundRecognizer
    StreamlinedSoundService --> StorageAdapterFactory

    %% Command Processing Flow
    CentralizedCommandParser --> StorageAdapterFactory
    CentralizedCommandParser --> EventBus

    AutomationService --> EventBus

    DictationCoordinator --> TextInputService
    DictationCoordinator --> LLMService
    DictationCoordinator --> AgenticPromptService
    DictationCoordinator --> StorageAdapterFactory

    %% UI Integration Flow
    GridService --> EventBus
    GridService --> GlobalAppConfig

    ClickTrackerService --> StorageAdapterFactory
    ClickTrackerService --> EventBus

    MarkService --> StorageAdapterFactory
    MarkService --> EventBus

    %% Storage Dependencies
    StorageAdapterFactory --> UnifiedStorageService
    UnifiedStorageService --> EventBus

    SettingsService --> StorageAdapterFactory
    CommandManagementService --> StorageAdapterFactory

    %% Optimization Dependencies
    MarkovCommandPredictor --> StorageAdapterFactory
    MarkovCommandPredictor --> EventBus

    SmartTimeoutManager --> GlobalAppConfig

    ProtectedTermsService --> StorageAdapterFactory
    ProtectedTermsService --> GlobalAppConfig

High-Level Information Flow
---------------------------

Audio Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

**Recorder Layer**: The audio processing begins with dual independent recorders managed by ``SimpleAudioService``:

- **Command Recorder**: Optimized for speed and responsiveness, using shorter audio segments and aggressive silence detection
- **Dictation Recorder**: Optimized for accuracy, using longer audio segments and more tolerant silence thresholds

Both recorders run continuously, using ``sounddevice`` for audio capture and Voice Activity Detection (VAD) for speech identification. The command recorder publishes ``AudioDetectedEvent`` for ultra-low latency Markov prediction bypass.

**Recognition Layer**: Audio segments flow to ``SpeechToTextService`` with mode-aware processing:

- **Command Mode**: Uses Vosk STT for speed-optimized recognition, checking only for "amber" trigger words during dictation
- **Dictation Mode**: Uses Whisper STT for accuracy-optimized recognition, processing full speech content

**Parallel Processing**: If STT fails to recognize speech, audio segments are sent to ``StreamlinedSoundService`` for custom sound recognition.

Command Execution Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Parsing Layer**: Recognized text flows to ``CentralizedCommandParser`` using hierarchical parsing:

1. **Dictation Commands**: ``start dictation``, ``stop dictation``, ``smart dictation``, ``type dictation``
2. **Mark Commands**: ``mark <label>``, ``mark delete <label>``, ``visualize marks``
3. **Grid Commands**: ``show grid [number]``, ``select [number]``, ``cancel grid``
4. **Automation Commands**: Exact matches or parameterized (``command [number]``) from stored action map

**Execution Layer**: Parsed commands are published as specific events that services consume:

- ``DictationCommandParsedEvent`` → ``DictationCoordinator``
- ``AutomationCommandParsedEvent`` → ``AutomationService``
- ``MarkCommandParsedEvent`` → ``MarkService``
- ``GridCommandParsedEvent`` → ``GridService``

Storage Integration
~~~~~~~~~~~~~~~~~~~

**Centralized Storage**: All services use ``StorageAdapterFactory`` and ``UnifiedStorageService`` for:

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
   AudioRecorder[AudioRecorder<br/>in SimpleAudioService] -->|publishes| CmdAudio[CommandAudioSegmentReadyEvent]
   AudioRecorder -->|publishes| DictAudio[DictationAudioSegmentReadyEvent]
   AudioRecorder -->|publishes| AudioDetected[AudioDetectedEvent]

   CmdAudio -->|consumed by| STTService[SpeechToTextService]
   DictAudio -->|consumed by| STTService
   AudioDetected -->|consumed by| MarkovPredictor[MarkovCommandPredictor]

   STTService -->|publishes| CmdText[CommandTextRecognizedEvent]
   STTService -->|publishes| DictText[DictationTextRecognizedEvent]
   STTService -->|publishes on STT failure| SoundEvent[ProcessAudioChunkForSoundRecognitionEvent]

   SoundEvent -->|consumed by| SoundService[StreamlinedSoundService]
   SoundService -->|publishes| CustomSound[CustomSoundRecognizedEvent]

   MarkovPredictor -->|publishes on high confidence| MarkovPred[MarkovPredictionEvent]
   MarkovPred -->|consumed by| STTService

**Event Details and Data Structures**:

Audio capture events contain raw binary audio data along with metadata required for processing:

- **CommandAudioSegmentReadyEvent** (Priority: HIGH)
  
  - ``audio_bytes`` (bytes): Raw PCM audio data captured from microphone in int16 format
  - ``sample_rate`` (int): Sample rate in Hz (typically 16000)
  - Published by: ``AudioRecorder`` when voice activity ends in command mode
  - Consumed by: ``SpeechToTextService._handle_command_audio_segment()``
  - Characteristics: Shorter segments (~0.5-2s) optimized for responsive command recognition

- **DictationAudioSegmentReadyEvent** (Priority: HIGH)
  
  - ``audio_bytes`` (bytes): Raw PCM audio data in int16 format, typically longer segments
  - ``sample_rate`` (int): Sample rate in Hz (typically 16000)
  - Published by: ``AudioRecorder`` when voice activity ends in dictation mode
  - Consumed by: ``SpeechToTextService._handle_dictation_audio_segment()``
  - Characteristics: Longer segments (~1-5s) optimized for accuracy over speed

- **AudioDetectedEvent** (Priority: CRITICAL)
  
  - ``timestamp`` (float): Unix timestamp when audio above threshold was first detected
  - Published by: ``AudioRecorder`` immediately upon detecting voice activity
  - Consumed by: ``MarkovCommandPredictor._handle_audio_detected()``
  - Purpose: Enables ultra-low latency prediction by triggering Markov chain analysis before STT completes
  - Timing: Published 50-200ms before audio segment is ready for STT processing

Speech recognition result events carry the transcribed text along with performance metrics:

- **CommandTextRecognizedEvent** (Priority: HIGH, extends TextRecognizedEvent)
  
  - ``text`` (str): Recognized command text from Vosk STT engine
  - ``processing_time_ms`` (float): Time taken for STT processing in milliseconds
  - ``engine`` (str): STT engine identifier ("vosk" or "markov")
  - ``mode`` (str): Processing mode ("command")
  - ``confidence`` (float): Recognition confidence (0.0-1.0, default 1.0)
  - Published by: ``SpeechToTextService._publish_recognition_result()``
  - Consumed by: ``CentralizedCommandParser``, ``DictationCoordinator`` (for trigger detection), ``MarkovCommandPredictor`` (for feedback)
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
  - Published by: ``SpeechToTextService._publish_sound_recognition_event()`` when STT returns empty text
  - Consumed by: ``StreamlinedSoundService._handle_audio_chunk()``
  - Purpose: Enables custom sound recognition (claps, whistles, etc.) as command triggers

- **CustomSoundRecognizedEvent** (Priority: HIGH)
  
  - ``label`` (str): Recognized sound identifier (e.g., "whistle_1", "clap_2")
  - ``confidence`` (float): Recognition confidence score (0.0-1.0)
  - ``mapped_command`` (Optional[str]): Command phrase mapped to this sound, if any
  - Published by: ``StreamlinedSoundService._handle_audio_chunk()``
  - Consumed by: ``CentralizedCommandParser._handle_sound_recognized()``, ``DictationCoordinator._handle_sound_trigger()``
  - Storage: Sound-to-command mappings persisted via ``SoundStorageAdapter``

Prediction bypass for ultra-low latency:

- **MarkovPredictionEvent** (Priority: CRITICAL)
  
  - ``predicted_command`` (str): Predicted command text based on historical patterns
  - ``confidence`` (float): Prediction confidence (0.0-1.0, typically >0.7 to publish)
  - ``audio_id`` (int): Identifier for the audio bytes that triggered prediction
  - Published by: ``MarkovCommandPredictor._handle_audio_detected()``
  - Consumed by: ``SpeechToTextService._handle_markov_prediction()``
  - Result: Bypasses STT processing entirely, publishing CommandTextRecognizedEvent with engine="markov"
  - Latency reduction: 200-800ms faster than standard STT processing

Command Processing Event Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command processing pipeline transforms recognized text into structured command objects through hierarchical parsing, then routes them to specialized execution services. This architecture decouples text recognition from command execution, enabling flexible command interpretation and consistent error handling.

.. mermaid::

   graph TB
   CmdText[CommandTextRecognizedEvent] -->|consumed by| Parser[CentralizedCommandParser]
   CustomSound[CustomSoundRecognizedEvent] -->|consumed by| Parser

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

Input events to the parser:

- **CommandTextRecognizedEvent** (described above)
  
  - Consumed by: ``CentralizedCommandParser._handle_text_recognized()``
  - Processing: Normalized to lowercase, stripped of whitespace, then matched against command patterns
  - Duplicate detection: Filtered using 1-second time window to prevent double-execution

- **CustomSoundRecognizedEvent** (described above)
  
  - Consumed by: ``CentralizedCommandParser._handle_sound_recognized()``
  - Processing: Sound label mapped to command phrase via ``_sound_to_command_mapping`` dict, then parsed as text
  - Mapping storage: Persisted in ``sound_mappings.json`` via ``SoundStorageAdapter``

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
    - ``action_type`` (str): Automation action ("hotkey", "click", "mouse_move", "scroll", "text")
    - ``action_value`` (str): Action-specific parameter (e.g., "ctrl+c" for hotkey, "100" for scroll)
    - ``count`` (int): Repeat count for parameterized commands (e.g., "scroll down 5")
  
  - Published by: ``CentralizedCommandParser._publish_command_event()``
  - Consumed by: ``AutomationService._handle_automation_command()``
  - Storage: Command definitions loaded from ``AutomationCommandRegistry`` and ``custom_commands.json``
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
  - Storage: Mark coordinates persisted via ``MarkStorageAdapter`` with caching for instant retrieval

- **GridCommandParsedEvent** (Priority: NORMAL)
  
  - ``command`` (GridCommandType): Union of grid command variants
  - Command types:
    
    - ``GridShowCommand(grid_number: Optional[int])``: Displays overlay grid for mouse navigation
    - ``GridSelectCommand(cell_number: int)``: Clicks specified grid cell
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
  - Consumed by: ``StreamlinedSoundService`` methods (``_handle_training_request``, ``_handle_delete_sound``, etc.)
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

**Dictation Event Details and Processing Modes**:

Control and state management events:

- **DictationCommandParsedEvent** (described in previous section)
  
  - Consumed by: ``DictationCoordinator._handle_dictation_command()``
  - Processing: Transitions dictation state machine (INACTIVE → STANDARD/TYPE/SMART → INACTIVE)
  - State tracking: ``DictationCoordinator`` maintains ``active_mode`` (DictationMode enum) and ``_sessions`` dict

- **DictationStatusChangedEvent** (Priority: LOW)
  
  - ``is_active`` (bool): Whether any dictation mode is currently active
  - ``mode`` (str): Current mode ("inactive", "continuous", "type", "smart")
  - ``show_ui`` (bool): Whether to display dictation UI indicator
  - ``stop_command`` (Optional[str]): Voice command to stop current dictation mode
  - Published by: ``DictationCoordinator._publish_status()``
  - Purpose: Synchronizes UI indicators and system state across application

- **DictationModeDisableOthersEvent** (Priority: CRITICAL)
  
  - ``dictation_mode_active`` (bool): Whether dictation mode is active
  - ``dictation_mode`` (str): Type of dictation mode active
  - Published by: ``DictationCoordinator`` on mode activation/deactivation
  - Consumed by: ``SpeechToTextService._handle_dictation_mode_change()``
  - Effect: When active, command audio processing only checks for amber stop words ("amber", "stop", "end"), suppressing normal command recognition to prevent interference

Text recognition and processing events:

- **DictationTextRecognizedEvent** (described in Audio Event Flow section)
  
  - Consumed by: ``DictationCoordinator._handle_dictation_text()``
  - Processing varies by mode:
    
    - **Standard/Type mode**: Text immediately sent to ``TextInputService.type_text()``
    - **Smart mode**: Text accumulated in ``DictationSession.accumulated_text`` buffer
  
  - Session management: Each dictation session tracked with unique ID, start time, and accumulated text

Smart dictation-specific events (LLM-enhanced text processing):

- **SmartDictationStartedEvent** (Priority: NORMAL)
  
  - ``mode`` (str): Always "smart"
  - Published by: ``DictationCoordinator._activate_smart_dictation()``
  - Purpose: Signals UI to prepare for smart dictation session

- **SmartDictationTextDisplayEvent** (Priority: HIGH)
  
  - ``text`` (str): Cleaned, formatted text for real-time display in UI
  - Published by: ``DictationCoordinator._handle_dictation_text()`` during smart dictation
  - Processing: Raw STT output cleaned (normalized spacing, capitalization) before display
  - Purpose: Provides real-time feedback to user before LLM processing

- **SmartDictationStoppedEvent** (Priority: NORMAL)
  
  - ``mode`` (str): Always "smart"
  - ``raw_text`` (str): Complete accumulated text before LLM processing
  - Published by: ``DictationCoordinator._deactivate_smart_dictation()``
  - Processing: Triggers LLM processing pipeline via ``_process_smart_dictation()``

- **LLMProcessingReadyEvent** (Priority: HIGH)
  
  - ``session_id`` (str): Unique identifier matching LLM processing request
  - Published by: UI components when ready to receive streaming tokens
  - Consumed by: ``DictationCoordinator._handle_llm_processing_ready()``
  - Purpose: Synchronization mechanism ensuring UI is ready before LLM streaming begins

LLM processing events:

- **LLMProcessingStartedEvent** (Priority: NORMAL)
  
  - ``raw_text`` (str): Original dictated text to be enhanced
  - ``agentic_prompt`` (str): LLM instruction prompt (e.g., "Fix grammar and improve clarity")
  - ``session_id`` (Optional[str]): Session identifier for correlation
  - Published by: ``DictationCoordinator._start_llm_processing()``
  - Purpose: Informational event for logging and UI state

- **LLMTokenGeneratedEvent** (Priority: HIGH)
  
  - ``token`` (str): Individual token generated during LLM streaming
  - Published by: ``LLMService.process_dictation_streaming()`` via ``token_callback``
  - Consumed by: UI components for real-time streaming display
  - Frequency: Published for each token as generated (typically 10-50 tokens/second)
  - Implementation: Uses ``ThreadSafeEventPublisher`` to safely publish from LLM's synchronous callback

- **LLMProcessingCompletedEvent** (Priority: NORMAL)
  
  - ``processed_text`` (str): LLM-enhanced text with grammar corrections and clarity improvements
  - ``agentic_prompt`` (str): The agentic prompt that was used
  - Published by: ``LLMService._publish_completed()``
  - Consumed by: ``DictationCoordinator._handle_llm_completed()``
  - Processing: Completed text sent to ``TextInputService.type_text()`` for system input
  - Validation: Output validated for length ratio (0.3-3.0x original) and repetition checks

- **LLMProcessingFailedEvent** (Priority: NORMAL)
  
  - ``error_message`` (str): Detailed error description
  - ``original_text`` (str): The text that failed processing
  - Published by: ``LLMService._publish_failed()``
  - Consumed by: ``DictationCoordinator._handle_llm_failed()``
  - Fallback: Original raw text is typed instead of processed text
  - Error recovery: Session cleaned up, UI notified of failure

Agentic prompt management events:

- **AgenticPromptUpdatedEvent** (Priority: LOW)
  
  - ``prompt`` (str): New active agentic prompt text
  - ``prompt_id`` (str): Unique identifier for the prompt
  - Published by: ``AgenticPromptService.set_prompt()``
  - Purpose: Notifies UI and services of prompt changes

- **AgenticPromptActionRequest** (Priority: NORMAL)
  
  - ``action`` (str): Action to perform ("add_prompt", "delete_prompt", "set_prompt", "get_prompts")
  - ``name`` (Optional[str]): Prompt name for add_prompt
  - ``text`` (Optional[str]): Prompt text for add_prompt
  - ``prompt_id`` (Optional[str]): Identifier for delete/set operations
  - Published by: UI components or voice commands
  - Consumed by: ``AgenticPromptService._handle_prompt_action()``
  - Storage: Prompts persisted via ``AgenticPromptStorageAdapter``

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

Grid navigation events enable voice-controlled mouse positioning through numbered overlay cells:

- **GridCommandParsedEvent** (described in Command Processing section)
  
  - Consumed by: ``GridService._handle_grid_command()``
  - Processing: Manages grid state, calculates cell positions, triggers UI updates

- **ShowGridRequestEventData** (Priority: NORMAL)
  
  - ``rows`` (int): Number of grid rows (default from config)
  - ``columns`` (int): Number of grid columns (default from config)
  - Published by: ``GridService._show_grid()``
  - Consumed by: ``GridWindow`` UI component
  - Grid calculation: Cell dimensions computed from screen size divided by row/column counts
  - Click optimization: Cell ordering uses ``ClickTrackerService`` statistics to prioritize frequently-clicked regions

- **ClickGridCellRequestEventData** (Priority: CRITICAL)
  
  - ``cell_label`` (str): Grid cell identifier (e.g., "5", "12")
  - Published by: ``GridService._select_cell()``
  - Consumed by: ``GridWindow`` UI component
  - Processing: Cell label mapped to screen coordinates, click performed via ``pyautogui.click()``
  - State management: Grid automatically hidden after cell selection

- **GridVisibilityChangedEventData** (Priority: LOW)
  
  - ``visible`` (bool): Whether grid is currently visible
  - ``rows`` (int): Current grid row count
  - ``columns`` (int): Current grid column count
  - Published by: ``GridService`` on state changes
  - Purpose: Synchronizes grid state across UI components

Mark navigation events enable voice-controlled return to saved screen positions:

- **MarkCommandParsedEvent** (described in Command Processing section)
  
  - Consumed by: ``MarkService._handle_mark_command()``
  - Processing: Executes mark operations with cached coordinate lookups

- **MarkCreatedEventData** (Priority: NORMAL)
  
  - ``name`` (str): Unique mark identifier
  - ``x`` (int): Screen X coordinate
  - ``y`` (int): Screen Y coordinate
  - Published by: ``MarkService._execute_mark_command()`` after successful mark creation
  - Consumed by: ``MarkOverlay`` UI component
  - Storage: Persisted via ``MarkStorageAdapter`` with 5-minute cache TTL
  - Validation: Mark names checked against ``ProtectedTermsService`` to prevent conflicts

- **MarkDeletedEventData** (Priority: NORMAL)
  
  - ``name`` (str): Deleted mark identifier
  - Published by: ``MarkService._execute_mark_command()`` after mark deletion
  - Consumed by: ``MarkOverlay`` UI component
  - Cache invalidation: Removes mark from ``MarkStorageAdapter`` cache immediately

- **MarkVisualizationStateChangedEventData** (Priority: LOW)
  
  - ``visible`` (bool): Whether mark visualization overlay is visible
  - ``marks`` (List): Current marks to display (name, x, y coordinates)
  - Published by: ``MarkService._execute_mark_command()`` on visualization toggle
  - Consumed by: ``MarkOverlay`` UI component
  - Purpose: Displays numbered overlays at each mark position for visual reference

Click tracking events collect mouse interaction patterns for grid optimization:

- **PerformMouseClickEventData** (Priority: CRITICAL)
  
  - ``x`` (int): Screen X coordinate of click
  - ``y`` (int): Screen Y coordinate of click
  - ``source`` (str): Click origin ("grid", "mark", "automation", "unknown")
  - Published by: ``GridService``, ``MarkService``, ``AutomationService`` when performing clicks
  - Consumed by: ``ClickTrackerService._handle_click()``
  - Debouncing: Clicks batched with 2-second window to reduce storage writes

- **ClickLoggedEventData** (Priority: LOW)
  
  - ``x`` (int): Screen X coordinate
  - ``y`` (int): Screen Y coordinate
  - ``timestamp`` (float): Unix timestamp of click
  - Published by: ``ClickTrackerService`` after persisting click to storage
  - Purpose: Confirmation event for logging and debugging

- **ClickCountsForGridEventData** (Priority: LOW)
  
  - ``rectangles`` (List): List of grid cells with click statistics
  - Each rectangle contains: ``x``, ``y``, ``width``, ``height``, ``click_count``
  - Published by: ``ClickTrackerService.get_click_statistics()``
  - Used by: ``GridService`` to prioritize cell numbering based on usage patterns
  - Algorithm: Grid cells sorted by click frequency, most-clicked areas get lower numbers for faster access

Optimizations and Ancillary Services
------------------------------------

Disambiguation and Smart Timeout Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``SmartTimeoutManager`` addresses a fundamental challenge in voice-controlled systems: distinguishing between complete commands and partial prefixes of longer commands. When a user says "right", the system must determine whether they meant the standalone "right" command or if they're about to continue with "right click". This service analyzes the command vocabulary to identify ambiguous patterns and provides guidance to other services.

**Architecture and Initialization**:

The ``SmartTimeoutManager`` is instantiated by ``SpeechToTextService`` during initialization, receiving the ``GlobalAppConfig`` and an optional command action map. Upon initialization, the service performs a comprehensive analysis of all registered commands to build an ambiguity detection structure.

The analysis process examines every command pair to identify prefix relationships. A command is marked as "ambiguous" if it forms a complete prefix of another command when followed by a space. For example, "right" is ambiguous if "right click" exists, but "click" is unambiguous even if "clicking" exists (no space boundary). This word-boundary-aware approach prevents false positives while catching true ambiguities.

**Core Data Structures**:

- ``_all_commands`` (Set[str]): Complete set of normalized command phrases (lowercased, stripped)
- ``_ambiguous_commands`` (Set[str]): Subset of commands that are prefixes of longer commands
- ``_command_action_map`` (Dict[str, tuple]): Live command mappings from ``CentralizedCommandParser``

**Ambiguity Detection Algorithm**:

The ``_build_ambiguity_analysis()`` method implements the prefix detection logic:

1. Convert all command phrases to a list for comparison
2. For each command, check if any other command starts with ``command + " "``
3. If prefix relationship found, add command to ``_ambiguous_commands`` set
4. Log statistics: total commands analyzed and ambiguous commands identified

Example ambiguous patterns detected:

- "right" → "right click", "right double click"
- "scroll" → "scroll down", "scroll up", "scroll left"
- "page" → "page down", "page up"

**Integration with Speech Recognition**:

The ``is_ambiguous(text: str)`` method provides a simple boolean check that other services use to determine whether additional processing time should be allocated. The ``SpeechToTextService`` uses this during command mode processing, although the primary consumer is intended to be the Markov prediction system to avoid premature command execution.

**Dynamic Updates**:

The service subscribes to ``CommandMappingsUpdatedEvent`` via the ``SpeechToTextService`` proxy. When custom commands are added, deleted, or modified, the service receives updated mappings and reanalyzes the complete command vocabulary. This ensures ambiguity detection remains accurate as users customize their command set.

The ``update_command_action_map()`` method triggers reanalysis:

1. Replace ``_command_action_map`` with new mappings
2. Clear existing ``_all_commands`` and ``_ambiguous_commands`` sets
3. Re-extract all command phrases from updated map
4. Rebuild ambiguity analysis with new command set
5. Log updated statistics for debugging

**Performance Characteristics**:

- Initialization time: O(n²) where n = number of commands (typically <1ms for 50-100 commands)
- Query time: O(1) hash set lookup for ``is_ambiguous()``
- Memory footprint: Two sets storing string references, negligible (<10KB)
- Update frequency: Only on command mapping changes (rare during normal operation)

**Configuration Integration**:

The service accesses configuration through ``GlobalAppConfig.stt``, which provides default timeout values and STT engine settings. While the service currently focuses on ambiguity detection, the configuration structure supports future enhancements like adaptive timeout windows based on command history.

Markov Chain Prediction System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MarkovCommandPredictor`` implements a sophisticated predictive command execution system that dramatically reduces interaction latency by anticipating user commands before speech recognition completes. By analyzing historical command sequences, the predictor achieves 200-800ms latency reduction for frequently-used command patterns.

**Core Concept and Motivation**:

Traditional voice command systems follow a sequential pipeline: audio capture → STT processing → command parsing → execution. This pipeline introduces unavoidable latency (typically 300-1000ms for STT alone). The Markov predictor breaks this constraint by predicting the next command based on previous commands, publishing predictions immediately when audio is detected, before STT begins processing.

The predictor leverages a fundamental insight: user command sequences often follow predictable patterns. For example, users frequently execute "select all" followed by "copy", or "mark label" followed by navigation commands. By learning these patterns, the system can predict with high confidence what command the user will speak, allowing execution to begin immediately.

**Multi-Order Markov Chain Architecture**:

The predictor implements a backoff strategy using 2nd through 4th order Markov chains:

- **2nd Order**: Predicts next command based on previous 1 command (e.g., "copy" after "select all")
- **3rd Order**: Uses previous 2 commands for context (e.g., "paste" after ["select all", "copy"])
- **4th Order**: Uses previous 3 commands for maximum context (e.g., specific editing workflows)

Data structures:

- ``_transition_counts``: Nested dict structure ``{order: {context_tuple: Counter}}``
  
  - Example: ``{2: {('select all',): Counter({'copy': 15, 'cut': 3})}}``
  - Stores raw transition counts for probability calculation
  - Context tuples are immutable sequences of previous commands

- ``_command_history``: ``deque`` with ``maxlen=max_order`` (typically 4)
  
  - Maintains sliding window of recent commands
  - Automatically discards oldest commands when new ones arrive
  - Used to build context tuples for prediction queries

**Prediction Algorithm (Backoff Strategy)**:

When ``AudioDetectedEvent`` fires, the predictor executes:

1. Check if prediction cooldown has expired (50ms minimum between predictions)
2. Extract context tuple from ``_command_history`` (previous 3, 2, or 1 commands)
3. Start with highest order (4th) and query ``_transition_counts``
4. Calculate confidence: ``count(predicted_command) / sum(all_counts_for_context)``
5. If confidence >= threshold (typically 0.7), publish ``MarkovPredictionEvent``
6. If confidence too low, backoff to lower order (4th → 3rd → 2nd)
7. If all orders fail to reach threshold, skip prediction (STT will handle normally)

**Training Process and Data Sources**:

The predictor trains on historical command sequences stored by ``CommandHistoryStorageAdapter``. Training occurs in two phases:

**Initial Training** (during ``initialize()``):

1. Load command history from ``command_history.json`` (persistent storage)
2. Filter commands by time window:
   
   - 4th order: Recent 7 days
   - 3rd order: Recent 14 days
   - 2nd order: Recent 30 days
   
3. Process sequences chronologically to build transition counts
4. Store completed model in ``_transition_counts`` dictionaries
5. Set ``_model_trained`` flag to enable predictions

**Continuous Training** (during operation):

1. Subscribe to ``CommandTextRecognizedEvent`` for real-time feedback
2. When command executes, append to ``_command_history`` deque
3. Extract contexts and update transition counts for all orders
4. Add command to ``_pending_commands`` buffer for batch persistence
5. Every 5 seconds, batch-write pending commands to storage via ``_batch_write_loop()``

**Prediction Verification and Feedback Loop**:

The predictor implements a verification mechanism to track prediction accuracy:

1. When ``MarkovPredictionEvent`` published, store prediction in ``_pending_prediction``
2. Subscribe to ``CommandTextRecognizedEvent`` to receive actual STT result
3. Compare predicted command with actual command:
   
   - **Match**: Log success, maintain/increase confidence in that transition
   - **Mismatch**: Log failure, potentially decrease transition weight (future enhancement)

4. Update ``_cooldown_remaining`` to prevent prediction spam on verification failures

**Fast-Track Bypass Mechanism**:

The prediction bypass flow operates as follows:

1. ``AudioRecorder`` detects voice activity, publishes ``AudioDetectedEvent`` immediately
2. ``MarkovCommandPredictor`` receives event, performs prediction query (1-5ms)
3. If high confidence, publishes ``MarkovPredictionEvent`` with predicted command text
4. ``SpeechToTextService`` receives ``MarkovPredictionEvent``:
   
   - Marks audio with ``audio_id`` as handled by Markov (in ``_markov_handled_audio`` set)
   - Publishes ``CommandTextRecognizedEvent`` with ``engine="markov"``
   - When original ``CommandAudioSegmentReadyEvent`` arrives, skips STT processing

5. ``CentralizedCommandParser`` parses command and publishes execution event
6. Command executes 200-800ms faster than standard STT path

**Performance Optimizations**:

- **Batch Writing**: Commands buffered in memory, written to disk every 5 seconds to reduce I/O
- **Context Deque**: Fixed-size deque for O(1) append and automatic oldest-item eviction
- **Hash-Based Lookups**: Context tuples used as dict keys for O(1) transition lookups
- **Confidence Thresholding**: Only publishes predictions above 70% confidence to minimize false positives
- **Prediction Cooldown**: 50ms minimum between predictions prevents rapid-fire mistakes

**Configuration Parameters** (``GlobalAppConfig.markov_predictor``):

- ``min_order``: Minimum Markov order (typically 2)
- ``max_order``: Maximum Markov order (typically 4)
- ``confidence_threshold``: Minimum confidence to publish prediction (0.0-1.0)
- ``training_window_days``: Time window for historical data per order
- ``enable_fast_track``: Master toggle for prediction system

**Integration Points**:

Event subscriptions:

- ``AudioDetectedEvent`` → ``_handle_audio_detected()``: Trigger prediction
- ``CommandTextRecognizedEvent`` → ``_handle_command_recognized()``: Feedback and training
- ``AutomationCommandParsedEvent`` → ``_handle_command_executed()``: Record successful execution
- ``MarkCommandParsedEvent`` → ``_handle_command_executed()``: Record mark commands
- ``GridCommandParsedEvent`` → ``_handle_command_executed()``: Record grid commands
- ``DictationCommandParsedEvent`` → ``_handle_command_executed()``: Record dictation transitions

Event publications:

- ``MarkovPredictionEvent``: Published to ``EventBus`` for STT bypass
- ``CommandExecutedEvent``: Published after confirmed command execution (for internal tracking)

Storage integration:

- Uses ``CommandHistoryStorageAdapter`` for persistent command sequence storage
- Reads from ``command_history.json`` during initialization
- Writes batched updates via ``_batch_write_loop()`` async task

Storage Service Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The storage layer implements a unified, high-performance persistence system that manages all application data through a centralized service with specialized adapters. This architecture eliminates code duplication, ensures consistent error handling, and provides sophisticated caching to minimize disk I/O while maintaining data consistency.

**UnifiedStorageService Core Architecture**:

The ``UnifiedStorageService`` serves as the foundational storage engine, providing low-level file operations with caching, thread safety, and atomic writes. All higher-level storage adapters delegate to this service for actual disk operations.

**Core Components and Data Structures**:

- ``_cache`` (Dict[str, CacheEntry]): In-memory cache mapping storage keys to cached data
  
  - Key format: ``"{storage_type}:{key}"`` (e.g., ``"marks:all_marks"``)
  - ``CacheEntry`` dataclass contains: ``data``, ``timestamp``, ``access_count``
  - TTL: 5 minutes (300 seconds) for automatic expiration

- ``_paths`` (Dict[StorageType, str]): Maps storage types to filesystem paths
  
  - ``StorageType.SETTINGS`` → ``{settings_dir}/user_settings.json``
  - ``StorageType.COMMANDS`` → ``{settings_dir}/custom_commands.json``
  - ``StorageType.MARKS`` → ``{marks_dir}/marks.json``
  - ``StorageType.GRID_CLICKS`` → ``{click_tracker_dir}/click_history.json``
  - ``StorageType.SOUND_MAPPINGS`` → ``{sound_model_dir}/sound_mappings.json``
  - ``StorageType.AGENTIC_PROMPTS`` → ``{user_data_root}/dictation/agentic_prompts.json``
  - ``StorageType.COMMAND_HISTORY`` → ``{command_history_dir}/command_history.json``

- ``_lock`` (threading.RLock): Reentrant lock for thread-safe cache access
- ``_executor`` (ThreadPoolExecutor): 2-worker thread pool for async I/O operations

**Read Operation Flow**:

The ``read(storage_key, default)`` method implements a cache-through pattern:

1. Construct cache key from storage type and key: ``"{storage_type.value}:{key}"``
2. Acquire ``_lock`` and check ``_cache`` for existing entry
3. If cache hit and not expired (< 5 minutes old):
   
   - Increment ``access_count`` for cache statistics
   - Return cached data immediately (typical latency: <0.1ms)

4. If cache miss or expired:
   
   - Determine filepath from ``_paths[storage_key.storage_type]``
   - Submit ``_sync_read()`` to ``_executor`` thread pool for async execution
   - ``_sync_read()`` reads JSON file, parses content, returns data
   - Create new ``CacheEntry`` with current timestamp and data
   - Store in ``_cache`` for future reads
   - Return data to caller (typical latency: 1-5ms for small files)

5. If file doesn't exist or JSON parse fails, return ``default`` value

**Write Operation Flow**:

The ``write(storage_key, data, immediate)`` method implements atomic writes:

1. Determine filepath from ``_paths[storage_key.storage_type]``
2. Submit ``_sync_write()`` to ``_executor`` for async execution
3. ``_sync_write()`` performs atomic write operation:
   
   - Serialize data to JSON string with indentation for readability
   - Write to temporary file: ``{filepath}.tmp.{uuid}``
   - Flush and fsync to ensure data reaches disk
   - Atomically rename temp file to final filepath (OS-level atomic operation)
   - This ensures file is never in partial/corrupt state, even during crash

4. If write succeeds:
   
   - Update ``_cache`` with new data and current timestamp
   - Invalidate old cache entry by replacing with fresh data
   - Return ``True`` to caller

5. If write fails:
   
   - Log error with full exception details
   - Return ``False`` to caller
   - Cache remains unchanged (stale data better than corrupt data)

**Cache Management and Performance**:

The caching layer provides significant performance benefits for frequently-accessed data:

- **Hit Rate Tracking**: ``CacheEntry.access_count`` tracks cache effectiveness
- **TTL Expiration**: ``is_expired(ttl)`` method checks if entry older than 5 minutes
- **Manual Invalidation**: ``clear_cache(storage_type)`` allows targeted cache clearing
- **Statistics API**: ``get_cache_stats()`` returns entries count, total accesses, hit rate

Example cache statistics after 1 hour of operation:

- Entries: 12 (marks, commands, settings, prompts, etc.)
- Total accesses: 1,847
- Hit rate: 95.2% (only 89 disk reads vs 1,847 requests)

**Storage Adapters and Specialization**:

The adapter pattern provides domain-specific storage interfaces while delegating to ``UnifiedStorageService`` for actual I/O. Each adapter focuses on business logic, data validation, and event publication.

**MarkStorageAdapter** (Position Bookmark Management):

Manages voice-navigable screen position bookmarks with instant cached lookups.

Key methods:

- ``get_all_marks()`` → ``List[Dict]``: Returns all marks ``[{"name": str, "x": int, "y": int}, ...]``
  
  - Cached aggressively for instant voice command navigation
  - Cache invalidated on mark creation/deletion

- ``add_mark(name, x, y)`` → ``bool``: Creates new mark with validation
  
  - Checks for duplicate names before adding
  - Validates coordinates are positive integers
  - Publishes ``MarkCreatedEventData`` on success

- ``get_mark(name)`` → ``Optional[Dict]``: Retrieves specific mark by name
  
  - O(n) linear search through marks list
  - Could be optimized with name-to-mark dict cache (future enhancement)

- ``delete_mark(name)`` → ``bool``: Removes mark and invalidates cache
- ``update_mark(name, x, y)`` → ``bool``: Updates mark coordinates

**CommandStorageAdapter** (Custom Command Management):

Manages user-defined automation commands with live reload support.

Key methods:

- ``get_action_map()`` → ``Dict[str, Tuple]``: Returns ``{command_phrase: (action_type, action_value)}``
  
  - Primary interface for command lookup during parsing
  - Merges built-in commands from ``AutomationCommandRegistry`` with custom commands
  - Cached for fast command recognition

- ``add_command(phrase, action_type, action_value)`` → ``Tuple[bool, str]``: Adds custom command
  
  - Validates phrase not protected/reserved
  - Checks for duplicates
  - Publishes ``CommandMappingsUpdatedEvent`` to trigger parser refresh

- ``update_command_phrase(old_phrase, new_phrase)`` → ``Tuple[bool, str]``: Renames command
  
  - Validates new phrase availability
  - Preserves action_type and action_value
  - Publishes update event for live reload

- ``delete_command(phrase)`` → ``Tuple[bool, str]``: Removes custom command
  
  - Only affects custom commands, built-ins cannot be deleted
  - Publishes update event

**SoundStorageAdapter** (Sound Recognition Mappings):

Manages custom sound-to-command mappings for non-speech audio triggers.

Key methods:

- ``get_sound_mappings()`` → ``Dict[str, str]``: Returns ``{sound_label: command_phrase}``
  
  - Example: ``{"whistle_1": "click", "clap_2": "scroll down"}``
  - Used by ``CentralizedCommandParser`` to convert sounds to commands

- ``add_sound_mapping(label, command)`` → ``bool``: Creates new mapping
  
  - Validates sound label exists (trained via ``StreamlinedSoundRecognizer``)
  - Publishes ``SoundToCommandMappingUpdatedEvent``

- ``delete_sound_mapping(label)`` → ``bool``: Removes mapping

**AgenticPromptStorageAdapter** (LLM Prompt Management):

Manages user-defined prompts for smart dictation LLM processing.

Key methods:

- ``get_all_prompts()`` → ``List[Dict]``: Returns all prompts with IDs and text
- ``add_prompt(name, text)`` → ``Dict``: Creates new prompt, returns dict with UUID
- ``delete_prompt(prompt_id)`` → ``bool``: Removes prompt by UUID
- ``set_current_prompt(prompt_id)`` → ``bool``: Sets active prompt for dictation

**GridClickStorageAdapter** (Click Pattern Tracking):

Persists mouse click history for grid cell optimization.

Key methods:

- ``log_click(x, y, timestamp)`` → ``bool``: Records click event
  
  - Debounced writes: batches clicks to reduce I/O
  - Stores as list: ``[{"x": int, "y": int, "timestamp": float}, ...]``

- ``get_click_history()`` → ``List[Dict]``: Returns all recorded clicks
  
  - Used by ``ClickTrackerService`` to calculate heat maps
  - Enables grid optimization based on usage patterns

**CommandHistoryStorageAdapter** (Markov Training Data):

Persists command execution history for Markov chain training.

Key methods:

- ``append_command(command_text, timestamp)`` → ``bool``: Records executed command
- ``get_command_history(start_time, end_time)`` → ``List[Dict]``: Time-windowed retrieval
  
  - Used by ``MarkovCommandPredictor`` during initialization training
  - Filters by date range for order-specific training windows

**Performance Characteristics and Benchmarks**:

- **Cached Read**: 0.05-0.1ms (dictionary lookup only)
- **Uncached Read**: 1-5ms (JSON parse + disk read for typical 1-10KB files)
- **Write**: 2-10ms (JSON serialize + atomic file write)
- **Cache Hit Rate**: 90-95% for marks, commands, settings during normal operation
- **Thread Safety**: All operations protected by reentrant lock, safe for concurrent access
- **Memory Footprint**: ~10-50KB for typical cache (12 entries × 1-4KB each)

**Error Handling and Recovery**:

- **Read Failures**: Return default value, log error, continue operation
- **Write Failures**: Return False to caller, preserve existing data, log detailed error
- **Corrupt JSON**: Catch parse exceptions, return default, log corruption warning
- **Missing Files**: Treat as empty, create on first write, normal initialization pattern
- **Atomic Writes**: Temp file + rename ensures no partial writes, data integrity guaranteed

**Event Integration for Live Updates**:

Storage adapters publish events when data changes, enabling real-time synchronization:

- ``CommandMappingsUpdatedEvent`` → Triggers ``CentralizedCommandParser`` reload
- ``SoundToCommandMappingUpdatedEvent`` → Updates sound recognition mappings
- ``MarkCreatedEventData`` / ``MarkDeletedEventData`` → Updates UI overlays
- ``AgenticPromptUpdatedEvent`` → Synchronizes LLM prompt across dictation service

This event-driven approach eliminates polling, reduces latency, and ensures all services operate on consistent data.

Protected Terms and Command Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ProtectedTermsService`` implements a comprehensive command naming validation system that prevents users from creating custom commands that conflict with system commands, existing marks, or trained sounds. This service maintains dynamic awareness of all protected terms across the application and provides centralized validation logic.

**Core Responsibility and Design Philosophy**:

Voice command systems require unique, unambiguous command phrases. If a user creates a custom command "click" that conflicts with the built-in "click" command, or names a mark "scroll down" that matches an automation command, the system cannot reliably determine user intent. The ``ProtectedTermsService`` enforces naming uniqueness as a preventive measure rather than handling conflicts reactively.

**Architecture and Data Sources**:

The service aggregates protected terms from multiple sources:

1. **System Automation Commands**: Built-in commands from ``AutomationCommandRegistry``
   
   - Examples: "click", "scroll down", "copy", "paste", "enter"
   - Source: ``registry.get_all_commands()`` loaded during initialization

2. **System Dictation Commands**: Hard-coded dictation control phrases
   
   - Examples: "start dictation", "stop dictation", "smart dictation", "type dictation"
   - Source: Defined in ``SYSTEM_DICTATION_COMMANDS`` constant

3. **System Mark Commands**: Reserved mark operation phrases
   
   - Examples: "visualize marks", "mark delete", "mark reset"
   - Source: Defined in ``SYSTEM_MARK_COMMANDS`` constant

4. **System Grid Commands**: Reserved grid operation phrases
   
   - Examples: "show grid", "cancel grid", "select [number]"
   - Source: Defined in ``SYSTEM_GRID_COMMANDS`` constant

5. **System Sound Commands**: Reserved sound management phrases
   
   - Examples: "train sound", "delete sound", "map sound"
   - Source: Defined in ``SYSTEM_SOUND_COMMANDS`` constant

6. **Live Mark Names**: Currently defined position bookmarks
   
   - Source: ``MarkStorageAdapter.get_all_marks()`` called dynamically
   - Examples: User-created marks like "editor", "browser", "terminal"

7. **Live Sound Labels**: Trained sound recognition labels
   
   - Source: ``SoundStorageAdapter.get_trained_sounds()`` called dynamically
   - Examples: User-trained sounds like "whistle_1", "clap_2"

**Validation Algorithm**:

The ``validate_command_name(name, exclude_name)`` method implements the validation logic:

1. Normalize input: Convert to lowercase, strip whitespace
2. Check if ``name == exclude_name`` (allowing rename of existing command to itself)
3. Query each protection category:
   
   - Check against system automation commands
   - Check against dictation, mark, grid, sound system commands
   - Check against live marks (fetch current list from storage)
   - Check against live sound labels (fetch current list from storage)

4. If conflict found, return error message: ``"Command name '{name}' conflicts with {category}: '{conflicting_term}'"``
5. If no conflicts, return ``None`` (validation passed)

**Live Data Integration and Freshness**:

Unlike static validation against pre-loaded data, this service fetches mark and sound data on every validation call. This ensures validation reflects current state:

- User creates mark "editor" → Immediately protected, cannot create command "editor"
- User deletes sound "whistle_1" → Immediately available for command naming
- No caching, no staleness risk, always accurate

**Integration Points**:

The service is used by:

- ``CommandManagementService._validate_command_phrase()``: Validates custom command additions
- ``MarkService`` (indirectly via storage adapter): Validates mark names before creation
- ``SoundService`` (indirectly via storage adapter): Validates sound labels before training

**Performance Considerations**:

- Validation involves 2-4 storage reads (marks, sounds) per call
- With caching in ``UnifiedStorageService``, typical validation latency: 0.2-1ms
- Validation only occurs during user configuration (rare), not during command execution
- Performance impact negligible compared to usability benefit

**Context-Aware Validation (Exclude Pattern)**:

The ``exclude_name`` parameter enables command renaming scenarios:

- User wants to rename command "old_click" to "click"
- Without exclusion, "click" would fail validation (conflicts with built-in)
- With exclusion: ``validate_command_name("click", exclude_name="old_click")``
- Validation allows "click" because it's replacing "old_click", not creating new conflict

**Debug and Introspection**:

The service provides ``get_all_protected_terms()`` method returning categorized lists:

::

    {
      "automation": ["click", "scroll down", ...],
      "dictation": ["start dictation", "stop dictation", ...],
      "marks": ["editor", "browser", ...],
      "sounds": ["whistle_1", "clap_2", ...],
      "grid": ["show grid", "cancel grid", ...],
      "sound_commands": ["train sound", "delete sound", ...]
    }

This enables UI display of protected terms and debugging of unexpected validation failures.

Grid Optimization and Click Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ClickTrackerService`` implements an intelligent grid optimization system that learns from user behavior to prioritize frequently-clicked screen regions. By analyzing historical click patterns, the service enables voice navigation to reach common targets with fewer commands.

**Motivation and Use Case**:

Voice-controlled grid navigation typically assigns cell numbers sequentially (left-to-right, top-to-bottom). This forces users to remember arbitrary cell numbers that change with different grid sizes. Worse, frequently-accessed areas might receive high cell numbers, requiring multi-syllable commands like "select twenty-seven".

The ``ClickTrackerService`` solves this by assigning lower cell numbers (1, 2, 3...) to frequently-clicked regions, enabling single-syllable access to common targets. If a user frequently clicks in the top-right corner (e.g., browser close button), that region automatically receives cell number "1" regardless of standard sequential numbering.

**Architecture and Data Flow**:

1. **Click Recording**: All services that perform mouse clicks publish ``PerformMouseClickEventData``
2. **Debounced Logging**: ``ClickTrackerService`` batches clicks for 2 seconds before persisting
3. **Storage Persistence**: Clicks saved to ``click_history.json`` via ``GridClickStorageAdapter``
4. **Statistics Generation**: ``get_click_statistics()`` analyzes history to produce heat map
5. **Grid Integration**: ``GridService`` requests statistics and reorders cells accordingly

**Click Recording and Debouncing**:

The ``_handle_click(event)`` method implements debounced logging:

::

    # Pseudocode flow
    def _handle_click(self, event: PerformMouseClickEventData):
        1. Append click to in-memory buffer: _pending_clicks.append(event)
        2. If write timer not active, start 2-second countdown
        3. When timer expires:
           - Batch all pending clicks
           - Write to storage in single operation
           - Publish ClickLoggedEventData for each click
           - Clear buffer

Debouncing reduces disk writes from potentially hundreds per minute to ~30 per minute, while maintaining data accuracy.

**Heat Map Generation Algorithm**:

The ``get_click_statistics(grid_rows, grid_columns)`` method transforms click history into grid-aligned statistics:

1. Load click history from storage (list of {x, y, timestamp} dicts)
2. Calculate screen dimensions via ``pyautogui.size()``
3. Compute cell dimensions: ``cell_width = screen_width / grid_columns``, ``cell_height = screen_height / grid_rows``
4. For each grid cell (row, col):
   
   - Calculate cell boundaries: ``x1 = col * cell_width``, ``y1 = row * cell_height``
   - Count clicks within boundaries: ``sum(1 for click if x1 <= click.x < x2 and y1 <= click.y < y2)``
   - Create rectangle dict: ``{x: x1, y: y1, width: cell_width, height: cell_height, click_count: count}``

5. Sort rectangles by ``click_count`` descending (most-clicked first)
6. Return sorted list

**Grid Cell Reordering**:

``GridService`` consumes statistics and reorders cells:

1. Request statistics for current grid dimensions (e.g., 4x4 = 16 cells)
2. Receive sorted rectangles with click counts
3. Assign cell numbers 1-16 based on sort order:
   
   - Rectangle with highest click_count → cell 1
   - Second highest → cell 2
   - Etc.

4. Display grid overlay with reordered numbers
5. When user says "select three", click the 3rd-most-frequently-clicked region

**Example Optimization**:

Standard sequential numbering (4x4 grid):

```
1   2   3   4
5   6   7   8
9   10  11  12
13  14  15  16
```

Click history shows user frequently clicks cells 4, 8, 12, 16 (right column, browser controls).

Optimized numbering after analysis:

```
13  9   5   1   (rightmost column gets 1-4, most-clicked)
14  10  6   2
15  11  7   3
16  12  8   4
```

User says "select one" to click their most common target (previously "select four").

**Performance Monitoring**:

The service tracks internal statistics via ``get_click_statistics_summary()``:

- Total clicks recorded
- Clicks per grid cell (average, min, max)
- Time range of recorded clicks
- Storage file size

This enables performance debugging and capacity planning (e.g., pruning old clicks if file grows too large).

**Configuration and Tuning**:

- ``click_history_retention_days``: How long to retain click history (default: 30 days)
- ``debounce_interval``: Click batching window (default: 2 seconds)
- ``min_clicks_for_optimization``: Minimum clicks before optimization activates (default: 20)

**Integration Points**:

Event subscriptions:

- ``PerformMouseClickEventData`` → ``_handle_click()``: Records all application clicks

Data consumers:

- ``GridService.show_grid()`` → ``get_click_statistics()``: Fetches optimized cell ordering

Storage dependencies:

- ``GridClickStorageAdapter.log_click()``: Persists click history
- ``GridClickStorageAdapter.get_click_history()``: Loads history for analysis

LLM Integration and Prompt Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LLM integration layer enables AI-powered text enhancement for smart dictation mode, transforming raw speech transcription into polished text through grammar correction, clarity improvements, and style refinement.

**LLMService Architecture**:

The ``LLMService`` wraps local LLM models (via ``llama-cpp-python``) to provide dictation enhancement while maintaining privacy and eliminating API costs. The service runs entirely locally using quantized models optimized for CPU inference.

**Model Management and Initialization**:

Initialization sequence:

1. ``LLMModelDownloader`` checks for model file at configured path
2. If missing, downloads from HuggingFace (e.g., ``Phi-3-mini-4k-instruct-q4.gguf``)
3. ``LLMService.initialize()`` loads model via ``llama_cpp.Llama``:
   
   - ``n_ctx``: Context length (typically 2048-4096 tokens)
   - ``n_threads``: CPU threads for inference (typically 4-8)
   - ``n_gpu_layers``: GPU offloading layers (0 for CPU-only)

4. Model loaded into memory (~2-4GB for quantized 3-7B models)
5. Service sets ``_model_loaded`` flag, ready for processing

**Streaming Inference Pipeline**:

The ``process_dictation_streaming(raw_text, agentic_prompt, token_callback)`` method implements token-by-token generation:

1. Construct prompt: ``f"{agentic_prompt}\n\nOriginal: {raw_text}\n\nImproved:"``
2. Invoke ``llm.create_completion(prompt, stream=True, max_tokens=512)``
3. For each generated token:
   
   - Extract token text from completion chunk
   - Call ``token_callback(token)`` to publish ``LLMTokenGeneratedEvent``
   - Append token to accumulated output buffer

4. After stream completes:
   
   - Validate output via ``_validate_output()`` (length ratio, repetition checks)
   - Clean output via ``_clean_output()`` (remove markdown artifacts, fix spacing)
   - Publish ``LLMProcessingCompletedEvent`` with final text

5. On error:
   
   - Catch exceptions (OOM, model errors, timeout)
   - Publish ``LLMProcessingFailedEvent`` with error details
   - ``DictationCoordinator`` falls back to typing raw text unmodified

**Output Validation and Quality Control**:

The ``_validate_output(output, original)`` method implements safety checks:

- **Minimum length**: Output must be ≥3 characters (prevents empty/garbage)
- **Length ratio**: ``0.3 ≤ len(output)/len(original) ≤ 3.0`` (prevents truncation or hallucination)
- **Repetition detection**: Unique words must be ≥50% of total (prevents model loops)

Failed validation triggers fallback to original text, ensuring user never receives broken output.

**AgenticPromptService Architecture**:

The ``AgenticPromptService`` manages a library of user-defined prompts that guide LLM behavior. Users can create custom prompts for different use cases: formal writing, casual messaging, code comments, etc.

**Prompt Storage and Management**:

Data structure (stored in ``agentic_prompts.json``):

::

    {
      "prompts": [
        {
          "id": "uuid-1234",
          "name": "Grammar Fix",
          "text": "Fix grammar and punctuation. Keep the meaning exactly the same."
        },
        {
          "id": "uuid-5678",
          "name": "Formal Writing",
          "text": "Rewrite in formal business tone. Fix grammar and improve clarity."
        }
      ],
      "current_prompt_id": "uuid-1234"
    }

Key methods:

- ``get_current_prompt()`` → ``str``: Returns active prompt text for LLM processing
- ``add_prompt(name, text)`` → ``Dict``: Creates new prompt with UUID, stores to JSON
- ``delete_prompt(prompt_id)`` → ``bool``: Removes prompt, publishes ``AgenticPromptListUpdatedEvent``
- ``set_prompt(prompt_id)`` → ``bool``: Sets active prompt, publishes ``AgenticPromptUpdatedEvent``

**Prompt Action Handling**:

The service subscribes to ``AgenticPromptActionRequest`` events from UI:

- ``action="get_prompts"``: Publishes ``AgenticPromptListUpdatedEvent`` with all prompts
- ``action="add_prompt"``: Validates name/text, adds to storage, publishes update
- ``action="delete_prompt"``: Removes from storage, publishes update
- ``action="set_prompt"``: Updates current_prompt_id, publishes ``AgenticPromptUpdatedEvent``

**Integration with Smart Dictation Flow**:

1. User says "smart dictation" → ``DictationCoordinator`` activates smart mode
2. User speaks text → Accumulated in session buffer
3. User says "stop dictation" → ``SmartDictationStoppedEvent`` published
4. ``DictationCoordinator`` queries: ``agentic_service.get_current_prompt()``
5. Current prompt + raw text sent to ``LLMService.process_dictation_streaming()``
6. LLM streams tokens → Published via ``LLMTokenGeneratedEvent`` → UI displays live
7. LLM completes → ``LLMProcessingCompletedEvent`` → Final text typed via ``TextInputService``

**Performance Characteristics**:

- **Model load time**: 2-5 seconds (one-time at startup)
- **Inference latency**: 10-50 tokens/second (CPU), 50-200 tokens/second (GPU)
- **Memory usage**: 2-4GB for quantized 3-7B models
- **Context window**: 2048-4096 tokens (sufficient for typical dictation)

**Configuration Parameters** (``GlobalAppConfig.llm``):

- ``model_name``: Model identifier (e.g., "phi-3-mini-4k-instruct-q4")
- ``context_length``: Maximum token context (default: 2048)
- ``n_threads``: CPU inference threads (default: 4)
- ``temperature``: Generation randomness (0.0-1.0, default: 0.3 for deterministic output)
- ``max_tokens``: Maximum generation length (default: 512)

Command Management and Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command management layer provides centralized administration of automation commands, enabling users to customize voice command behavior while maintaining system integrity through validation and conflict prevention.

**CommandManagementService** acts as the orchestration layer for command lifecycle:

- Receives command management events from UI (add, update, delete, query)
- Delegates storage operations to ``CommandStorageAdapter``
- Validates command names via ``ProtectedTermsService``
- Publishes ``CommandMappingsUpdatedEvent`` to trigger live reloads across dependent services

**AutomationCommandRegistry** provides the built-in command vocabulary:

- Defines default automation commands with action types and values
- Examples: ``{"click": ("click", "left")}``, ``{"scroll down": ("scroll", "100")}``
- Loaded during ``CentralizedCommandParser`` initialization
- Merged with custom commands from storage to form complete action map

**Command Definition Structure**:

Each command maps phrase to action tuple:

::

    {
      "command_phrase": ("action_type", "action_value"),
      "copy": ("hotkey", "ctrl+c"),
      "click": ("click", "left"),
      "scroll down": ("scroll", "100"),
      "type hello": ("text", "hello")
    }

Action types supported by ``AutomationService``:

- ``"hotkey"``: Sends keyboard combination via ``pyautogui.hotkey()``
- ``"click"``: Performs mouse click (left, right, middle)
- ``"mouse_move"``: Moves cursor to coordinates
- ``"scroll"``: Scrolls by pixel amount
- ``"text"``: Types literal text string

**Dynamic Command Updates**:

When user adds command "navigate browser" → "ctrl+l":

1. UI publishes ``AddCustomCommandEvent(phrase="navigate browser", action_type="hotkey", action_value="ctrl+l")``
2. ``CommandManagementService`` receives event
3. Validates phrase via ``ProtectedTermsService.validate_command_name()``
4. If valid, delegates to ``CommandStorageAdapter.add_command()``
5. Storage writes to ``custom_commands.json``, updates cache
6. Storage publishes ``CommandMappingsUpdatedEvent``
7. ``CentralizedCommandParser`` receives event, reloads action map
8. New command immediately available for recognition

Performance Monitoring and Analytics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The services architecture includes comprehensive performance monitoring to track system health, identify bottlenecks, and optimize resource usage.

**Service-Level Performance Metrics**:

Each service exposes performance data through dedicated methods:

- **MarkService.get_cache_stats()**: Returns mark coordinate cache hit rates, average lookup latency
- **ClickTrackerService.get_click_statistics_summary()**: Provides total clicks, heat map coverage, storage file size
- **MarkovCommandPredictor.get_prediction_stats()**: Reports prediction accuracy, confidence distributions, model training time
- **UnifiedStorageService.get_cache_stats()**: Shows cache entries, total accesses, hit rate percentage

**Event-Based Performance Tracking**:

Critical operations publish timing and status events:

- **CommandExecutedStatusEvent**: Published by ``AutomationService`` after each command execution
  
  - Tracks success/failure rates
  - Measures end-to-end command latency
  - Identifies problematic commands requiring optimization

- **STTProcessingStartedEvent / STTProcessingCompletedEvent**: Published by ``SpeechToTextService``
  
  - Measures STT processing time per engine (Vosk vs Whisper)
  - Tracks audio segment sizes correlated with processing time
  - Enables comparison of command vs dictation mode performance

- **LLMProcessingCompletedEvent**: Includes processing time for smart dictation analysis

**System-Wide Performance Characteristics** (typical values):

- **Audio Detection Latency**: 50-200ms (from speech start to ``AudioDetectedEvent``)
- **STT Command Recognition**: 150-400ms (Vosk, command mode)
- **STT Dictation Recognition**: 500-2000ms (Whisper, dictation mode)
- **Markov Prediction**: 1-5ms (when high-confidence match exists)
- **Command Parsing**: 0.5-2ms (pattern matching + validation)
- **Command Execution**: 10-100ms (varies by action type)
- **Storage Read (cached)**: 0.05-0.1ms (dictionary lookup)
- **Storage Read (uncached)**: 1-5ms (JSON parse + disk I/O)
- **End-to-End Command Latency**: 200-500ms (Markov fast-path), 400-800ms (standard STT path)

**Optimization Strategies Employed**:

- **Adaptive VAD Thresholds**: ``AudioRecorder`` adjusts energy thresholds based on ambient noise levels, reducing false positive audio detection
- **Multi-Level Caching**: Storage layer implements memory cache (L1), disk cache (L2), with TTL-based expiration
- **Batch Processing**: ``MarkovCommandPredictor`` writes command history in 5-second batches, ``ClickTrackerService`` batches clicks every 2 seconds
- **Predictive Loading**: ``MarkService`` pre-loads all marks into cache during initialization for instant voice navigation
- **Lazy Initialization**: LLM model only loaded when smart dictation first used, saving 2-4GB memory for users who don't use feature
- **Duplicate Filtering**: ``DuplicateTextFilter`` in ``SpeechToTextService`` prevents double-execution of same command within 1-second window
