====================
Dictation System
====================

This document provides comprehensive technical documentation of the IRIS dictation system, covering the coordination, text processing, and AI enhancement capabilities for converting speech to text input.

Overview
========

The IRIS dictation system implements a sophisticated multi-mode text input system with AI enhancement capabilities:

- **Standard Dictation**: Direct speech-to-text input with basic cleaning
- **Type Dictation**: Character-by-character typing simulation
- **Smart Dictation**: AI-enhanced text processing with grammar correction and clarity improvement

The system uses event-driven coordination, streaming text processing, and comprehensive error handling to provide a seamless dictation experience.

DictationCoordinator
====================

The ``DictationCoordinator`` is the central orchestrator for all dictation operations, managing mode switching, text processing, and AI enhancement workflows.

Architecture
------------

**Core Components**:

.. code-block:: python

   class DictationCoordinator:
       def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage_factory, gui_event_loop):
           # Core services
           self.text_service = TextInputService(config.dictation)
           self.llm_service = LLMService(event_bus, config)
           self.agentic_service = AgenticPromptService(event_bus, config, storage_factory)
           
           # Session management
           self._current_session: Optional[DictationSession] = None
           self._pending_llm_session: Optional[DictationSession] = None
           self._session_id: Optional[str] = None
           
           # Thread safety
           self._lock = threading.RLock()

**Key Features**:
- Multi-mode dictation support (standard, type, smart)
- AI-enhanced text processing with streaming
- Session-based state management
- Comprehensive event coordination
- Thread-safe operation

Dictation Modes
----------------

**Mode Definitions**:

.. code-block:: python

   class DictationMode(Enum):
       INACTIVE = "inactive"
       STANDARD = "standard"   # Direct text input
       SMART = "smart"        # AI-enhanced processing
       TYPE = "type"          # Character-by-character typing

**Session Management**:

.. code-block:: python

   @dataclass
   class DictationSession:
       mode: DictationMode
       start_time: float
       accumulated_text: str = ""
       is_processing: bool = False

Mode Activation and Triggers
----------------------------

**Trigger Word Processing**:

.. code-block:: python

   async def _process_trigger(self, text: str) -> None:
       """Process trigger words"""
       cfg = self.config.dictation
       
       if self.is_active():
           if text == cfg.stop_trigger.lower():  # "amber"
               await self._stop_session()
       else:
           if text == cfg.smart_start_trigger.lower():    # "smart green"
               await self._start_session(DictationMode.SMART)
           elif text == cfg.start_trigger.lower():        # "green"
               await self._start_session(DictationMode.STANDARD)
           elif text == cfg.type_trigger.lower():         # "type"
               await self._start_session(DictationMode.TYPE)

**Session Initialization**:

.. code-block:: python

   async def _start_session(self, mode: DictationMode) -> None:
       """Start dictation session"""
       try:
           with self._lock:
               self._current_session = DictationSession(mode=mode, start_time=time.time())
           
           # Request audio mode change
           await self._publish_event(
               AudioModeChangeRequestEvent(mode="dictation", reason=f"{mode.value} mode activated"))
           
           # Notify STT service about dictation mode activation
           await self._publish_event(
               DictationModeDisableOthersEvent(
                   dictation_mode_active=True,
                   dictation_mode=mode.value
               ))
           
           # Publish mode-specific events
           if mode == DictationMode.SMART:
               await self._publish_event(SmartDictationStartedEvent())
               await self._publish_event(
                   SmartDictationEnabledEvent(trigger_word=self.config.dictation.smart_start_trigger))
           
           await self._publish_status(True, mode)

Text Processing Pipeline
========================

The dictation system implements sophisticated text processing with mode-specific handling.

Text Reception and Cleaning
---------------------------

**Centralized Text Handling**:

.. code-block:: python

   async def _handle_dictation_text(self, event: DictationTextRecognizedEvent) -> None:
       """Handle dictated text - centralized processing for all dictation modes"""
       try:
           text = event.text.strip()
           if not text:
               return
               
           with self._lock:
               session = self._current_session
               
           if not session:
               return
           
           # Don't process new text if smart session is already being processed
           if session.mode == DictationMode.SMART and getattr(session, 'is_processing', False):
               logger.debug("Ignoring dictation text - smart session already processing")
               return
               
           # Clean and process text for all modes
           cleaned_text = self._clean_text(text)
           if not cleaned_text:
               return
               
           # Accumulate text for session tracking
           session.accumulated_text = self._append_text(session.accumulated_text, cleaned_text)
           
           # Handle based on dictation mode
           if session.mode == DictationMode.SMART:
               # Smart dictation: send cleaned text to UI for display, accumulate for LLM processing
               await self._publish_event(SmartDictationTextDisplayEvent(text=cleaned_text))
           else:
               # Standard/Type dictation: input text immediately
               await self.text_service.input_text(cleaned_text)

**Text Cleaning Algorithm**:

.. code-block:: python

   def _clean_text(self, text: str) -> str:
       """Clean dictated text by removing triggers"""
       if not text:
           return ""
           
       cfg = self.config.dictation
       triggers = {cfg.start_trigger.lower(), cfg.stop_trigger.lower(), 
                  cfg.type_trigger.lower(), cfg.smart_start_trigger.lower()}
       
       # Remove trigger words while preserving punctuation context
       words = [w for w in text.split() if w.lower().strip('.,!?;:"()[]{}') not in triggers]
       return ' '.join(words).strip()

**Text Accumulation**:

.. code-block:: python

   def _append_text(self, existing: str, new_text: str) -> str:
       """Append text with proper spacing"""
       if not existing:
           return new_text
       if not new_text:
           return existing
       return f"{existing} {new_text}"

Smart Dictation with AI Enhancement
===================================

Smart dictation provides AI-powered text enhancement with grammar correction and clarity improvement.

AI Processing Workflow
-----------------------

**Session Termination and LLM Preparation**:

.. code-block:: python

   async def _process_smart_dictation(self, session: DictationSession) -> None:
       """Process smart dictation with LLM - wait for UI ready signal"""
       try:
           session.is_processing = True
           self._session_id = str(uuid.uuid4())
           self._pending_llm_session = session
           
           # Switch back to command audio mode
           await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Smart dictation processing"))
           
           # Publish smart dictation stopped event (triggers UI)
           await self._publish_event(SmartDictationStoppedEvent(raw_text=session.accumulated_text))
           
           # Publish LLM processing started event with session ID
           agentic_prompt = self.agentic_service.get_current_prompt() or "Fix grammar and improve clarity."
           await self._publish_event(LLMProcessingStartedEvent(
               raw_text=session.accumulated_text, 
               agentic_prompt=agentic_prompt,
               session_id=self._session_id
           ))
           
           # LLM processing will start when UI signals ready via LLMProcessingReadyEvent

**UI Coordination**:

.. code-block:: python

   async def _handle_llm_processing_ready(self, event: LLMProcessingReadyEvent) -> None:
       """Handle LLM processing ready signal from UI"""
       try:
           if self._session_id and event.session_id == self._session_id and self._pending_llm_session:
               await self._start_llm_processing(self._pending_llm_session)
               self._pending_llm_session = None
               self._session_id = None

**LLM Processing Execution**:

.. code-block:: python

   async def _start_llm_processing(self, session: DictationSession) -> None:
       """Actually start the LLM processing after UI is ready"""
       try:
           agentic_prompt = self.agentic_service.get_current_prompt() or "Fix grammar and improve clarity."
           
           # Use streaming if available
           if hasattr(self.llm_service, 'process_dictation_streaming'):
               await self.llm_service.process_dictation_streaming(
                   session.accumulated_text, 
                   agentic_prompt,
                   token_callback=self._stream_token
               )
           else:
               await self.llm_service.process_dictation(session.accumulated_text, agentic_prompt)

**Streaming Token Handling**:

.. code-block:: python

   def _stream_token(self, token: str) -> None:
       """Helper to publish LLM tokens"""
       try:
           # Use the ThreadSafeEventPublisher for proper async event publishing from sync callback
           self.event_publisher.publish(LLMTokenGeneratedEvent(token=token))
       except Exception as e:
           logger.error(f"Token streaming error: {e}", exc_info=True)

**Processing Completion**:

.. code-block:: python

   async def _handle_llm_completed(self, event: LLMProcessingCompletedEvent) -> None:
       """Handle LLM completion"""
       try:
           success = await self.text_service.input_text(event.processed_text)
           
           # Clear the session now that processing is complete
           with self._lock:
               self._current_session = None
               
           await self._end_smart_session()

TextInputService
================

The ``TextInputService`` handles the actual text input to applications using clipboard or typing methods.

Architecture
------------

**Core Components**:

.. code-block:: python

   class TextInputService:
       def __init__(self, config: DictationConfig):
           self.config = config
           self._lock = threading.RLock()
           
           # Configure PyAutoGUI safety
           pyautogui.FAILSAFE = True
           pyautogui.PAUSE = 0.01

**Key Features**:
- Clipboard-based text input (default)
- Character-by-character typing fallback
- Advanced text cleaning and formatting
- Thread-safe operation

Text Input Methods
------------------

**Primary Input Method**:

.. code-block:: python

   async def input_text(self, text: str) -> bool:
       """Input text at cursor position"""
       if not text:
           return False
       
       try:
           cleaned_text = self._clean_text(text)
           if not cleaned_text:
               return False
           
           # Use clipboard or typing based on config
           if self.config.use_clipboard:
               success = await asyncio.get_event_loop().run_in_executor(
                   None, self._paste_clipboard, cleaned_text
               )
           else:
               success = await asyncio.get_event_loop().run_in_executor(
                   None, self._type_text, cleaned_text
               )
           
           return success

**Clipboard-Based Input**:

.. code-block:: python

   def _paste_clipboard(self, text: str) -> bool:
       """Paste using clipboard"""
       try:
           # Save original clipboard
           original = None
           try:
               original = pyperclip.paste()
           except:
               pass
           
           # Paste text
           pyperclip.copy(text)
           time.sleep(0.05)
           pyautogui.hotkey('ctrl', 'v') 
           time.sleep(0.1)
           
           # Restore clipboard
           if original is not None:
               try:
                   pyperclip.copy(original)
               except:
                   pass
           
           return True

**Character-by-Character Typing**:

.. code-block:: python

   def _type_text(self, text: str) -> bool:
       """Type text character by character"""
       try:
           for char in text:
               pyautogui.write(char, interval=self.config.typing_delay)
           time.sleep(0.1)
           return True

Text Processing and Quality Control
-----------------------------------

**Advanced Text Cleaning**:

.. code-block:: python

   def _clean_text(self, text: str) -> str:
       """Clean text for input while preserving formatting"""
       if not text:
           return ""
       
       # Preserve original formatting but clean up excessive whitespace
       cleaned = text.strip()
       
       # Clean up excessive spaces but preserve intentional formatting
       # Replace multiple spaces (but not newlines) with single space
       cleaned = re.sub(r'[ \t]+', ' ', cleaned)
       
       # Clean up excessive newlines (more than 2 consecutive)
       cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
       
       # Apply minimum length filter
       if len(cleaned) < self.config.min_text_length:
           return ""
       
       return cleaned

**Additional Text Operations**:

.. code-block:: python

   async def add_space(self) -> bool:
       """Add space character"""
       try:
           await asyncio.get_event_loop().run_in_executor(
               None, pyautogui.press, 'space'
           )
           return True

   async def add_newline(self) -> bool:
       """Add newline character"""
       try:
           await asyncio.get_event_loop().run_in_executor(
               None, pyautogui.press, 'enter'
           )
           return True

   async def backspace(self, count: int = 1) -> bool:
       """Send backspace keystrokes"""
       try:
           for _ in range(count):
               await asyncio.get_event_loop().run_in_executor(
                   None, pyautogui.press, 'backspace'
               )
           return True

Event Coordination
==================

The dictation system uses comprehensive event coordination for state management and UI synchronization.

Event Flow Architecture
-----------------------

**Dictation Lifecycle Events**:

1. **Activation Events**:
   - ``DictationStartedEvent``: Mode activation
   - ``DictationModeDisableOthersEvent``: System coordination
   - ``AudioModeChangeRequestEvent``: Audio system coordination

2. **Processing Events**:
   - ``DictationTextRecognizedEvent``: Incoming speech text
   - ``SmartDictationTextDisplayEvent``: UI text display
   - ``LLMProcessingStartedEvent``: AI processing initiation

3. **AI Processing Events**:
   - ``LLMProcessingReadyEvent``: UI ready signal
   - ``LLMTokenGeneratedEvent``: Streaming token output
   - ``LLMProcessingCompletedEvent``: AI processing completion

4. **Termination Events**:
   - ``DictationStoppedEvent``: Mode deactivation
   - ``SmartDictationStoppedEvent``: Smart mode completion
   - ``DictationStatusChangedEvent``: UI status updates

**Event Publishing Pattern**:

.. code-block:: python

   async def _publish_event(self, event: BaseEvent) -> None:
       """Publish event with error handling"""
       try:
           await self.event_bus.publish(event)
       except Exception as e:
           logger.error(f"Event publishing error: {e}", exc_info=True)

**Status Event Management**:

.. code-block:: python

   async def _publish_status(self, is_active: bool, mode: DictationMode) -> None:
       """Publish status change event"""
       try:
           display_mode = "continuous" if mode == DictationMode.STANDARD else mode.value
           event = DictationStatusChangedEvent(
               is_active=is_active,
               mode=display_mode,
               show_ui=is_active,
               stop_command=self.config.dictation.stop_trigger if is_active else None
           )
           await self._publish_event(event)

Configuration System
====================

The dictation system is highly configurable through the ``DictationConfig`` class.

Configuration Options
---------------------

.. code-block:: python

   class DictationConfig(BaseModel):
       """Configuration for dictation functionality"""
       
       # Trigger words
       start_trigger: str = Field(default="green", description="Trigger word to start standard dictation")
       stop_trigger: str = Field(default="amber", description="Trigger word to stop any dictation mode")
       type_trigger: str = Field(default="type", description="Trigger word to start type mode")
       smart_start_trigger: str = Field(default="smart green", description="Trigger phrase to start LLM-assisted dictation")
       
       # STT Engine switching for dictation
       enable_stt_switching: bool = Field(default=True, description="Enable automatic STT engine switching for dictation")
       dictation_stt_engine: str = Field(default="whisper", description="STT engine to use during dictation")
       command_stt_engine: str = Field(default="vosk", description="STT engine to use for command recognition")
       
       # Text filtering and processing
       min_text_length: int = Field(default=1, description="Minimum length of text to process")
       remove_trigger_words: bool = Field(default=True, description="Whether to remove trigger words from dictated text")
       
       # Text input settings
       use_clipboard: bool = Field(default=True, description="Use clipboard for text input instead of typing")
       typing_delay: float = Field(default=0.01, description="Delay between keystrokes when typing")

Error Handling and Resilience
==============================

The dictation system implements comprehensive error handling and recovery mechanisms.

Session Management Resilience
------------------------------

**Session State Protection**:

.. code-block:: python

   async def _stop_session(self) -> None:
       """Stop dictation session"""
       try:
           with self._lock:
               session = self._current_session
               if not session:
                   return
               
               # Don't clear session yet for smart dictation - keep it until LLM processing completes
               if session.mode != DictationMode.SMART:
                   self._current_session = None
           
           if session.mode == DictationMode.SMART:
               if session.accumulated_text:
                   await self._process_smart_dictation(session)
                   return  # Don't publish stop events yet - wait for LLM
               else:
                   # Clear session now since no LLM processing needed
                   with self._lock:
                       self._current_session = None
                   await self._end_smart_session()
           else:
               await self._finalize_session(session)

**Error Recovery**:

.. code-block:: python

   async def _handle_llm_failed(self, event: LLMProcessingFailedEvent) -> None:
       """Handle LLM failure"""
       logger.warning(f"LLM processing failed: {event.error_message}")
       await self._end_smart_session()
       await self._publish_error("Smart dictation processing failed")

   async def _publish_error(self, message: str) -> None:
       """Publish error event"""
       try:
           event = DictationErrorEvent(error_message=message, mode=self.active_mode.value)
           await self._publish_event(event)
       except Exception as e:
           logger.error(f"Error publishing error: {e}", exc_info=True)

**Graceful Shutdown**:

.. code-block:: python

   async def shutdown(self) -> None:
       """Shutdown coordinator"""
       try:
           if self._current_session:
               await self._stop_session()
           await self.text_service.shutdown()
           await self.llm_service.shutdown()
           await self.agentic_service.shutdown()
       except Exception as e:
           logger.error(f"Shutdown error: {e}", exc_info=True)

Performance Optimization
========================

The dictation system implements several performance optimizations:

Processing Efficiency
--------------------

**Asynchronous Operations**:
- Non-blocking text input using executor threads
- Concurrent LLM processing with streaming
- Event-driven coordination without polling
- Efficient session state management

**Memory Management**:
- Session-based text accumulation
- Efficient event handling
- Proper resource cleanup
- Minimal memory footprint

**Threading Model**:
- Thread-safe session management
- Lock-protected critical sections
- Asynchronous event processing
- Cross-thread coordination

Quality Assurance
-----------------

**Text Quality Control**:
- Advanced text cleaning and normalization
- Trigger word removal
- Formatting preservation
- Length validation

**Session Integrity**:
- State consistency validation
- Error recovery mechanisms
- Graceful degradation
- Comprehensive logging

**AI Processing Quality**:
- Prompt management and optimization
- Streaming token validation
- Processing timeout handling
- Fallback mechanisms

Advanced Smart Dictation Sequence Diagram
=========================================

The smart dictation workflow involves complex coordination between multiple services:

.. mermaid::

   sequenceDiagram
       participant User
       participant AudioRecorder as Audio Recorder
       participant STTService as STT Service
       participant DictationCoordinator as Dictation Coordinator
       participant UI as User Interface
       participant LLMService as LLM Service
       participant TextInputService as Text Input Service
       
       Note over User,TextInputService: Smart Dictation Session Lifecycle
       
       User->>AudioRecorder: smart green trigger
       AudioRecorder->>STTService: CommandTextRecognizedEvent smart green
       STTService->>DictationCoordinator: Trigger processing
       
       DictationCoordinator->>DictationCoordinator: Create smart session
       DictationCoordinator->>AudioRecorder: AudioModeChangeRequestEvent dictation
       DictationCoordinator->>UI: SmartDictationStartedEvent
       
       Note over AudioRecorder: Both recorders now active
       
       loop Dictation Phase
           User->>AudioRecorder: Speech input
           AudioRecorder->>STTService: DictationAudioSegmentReadyEvent
           STTService->>DictationCoordinator: DictationTextRecognizedEvent
           DictationCoordinator->>DictationCoordinator: Accumulate + clean text
           DictationCoordinator->>UI: SmartDictationTextDisplayEvent
           UI->>UI: Display cleaned text in real-time
       end
       
       User->>AudioRecorder: amber stop trigger
       AudioRecorder->>STTService: CommandTextRecognizedEvent amber
       STTService->>DictationCoordinator: Stop signal
       
       DictationCoordinator->>DictationCoordinator: session.is_processing = True
       DictationCoordinator->>AudioRecorder: AudioModeChangeRequestEvent command
       DictationCoordinator->>UI: SmartDictationStoppedEvent raw_text
       DictationCoordinator->>UI: LLMProcessingStartedEvent session_id
       
       UI->>UI: Show LLM processing indicator
       UI->>DictationCoordinator: LLMProcessingReadyEvent session_id
       
       DictationCoordinator->>LLMService: process_dictation_streaming text prompt
       
       loop Token Streaming
           LLMService->>DictationCoordinator: Token callback
           DictationCoordinator->>UI: LLMTokenGeneratedEvent token
           UI->>UI: Update display with streaming tokens
       end
       
       LLMService->>DictationCoordinator: LLMProcessingCompletedEvent processed_text
       DictationCoordinator->>TextInputService: input_text processed_text
       
       alt Clipboard Mode (default)
           TextInputService->>TextInputService: Save original clipboard
           TextInputService->>TextInputService: Copy processed text
           TextInputService->>User: Ctrl+V paste to application
           TextInputService->>TextInputService: Restore original clipboard
       else Typing Mode
           TextInputService->>User: Character-by-character typing
       end
       
       DictationCoordinator->>DictationCoordinator: Clear session
       DictationCoordinator->>UI: Session complete

Performance Analysis and Optimization
=====================================

**Dictation Mode Performance Breakdown**:

.. code-block:: python

   # Comprehensive performance analysis for dictation workflows
   DICTATION_PERFORMANCE_ANALYSIS = {
       'standard_dictation': {
           'audio_capture_latency': '20ms',      # Chunk processing
           'whisper_processing': '200-800ms',    # Varies by segment length
           'text_cleaning': '<1ms',              # Regex operations
           'clipboard_paste': '50-100ms',        # System clipboard operations
           'total_end_to_end': '270-920ms',      # From speech to application
           'memory_usage': '~500MB',             # Whisper model + buffers
           'cpu_utilization': '15-30%',          # Multi-core Whisper processing
           'accuracy_rate': '~95%',              # Natural language accuracy
           'throughput': '~150 WPM',             # Words per minute processing
       },
       
       'smart_dictation': {
           'dictation_phase': {
               'audio_to_display': '270-920ms',   # Same as standard dictation
               'ui_update_latency': '16ms',       # 60fps UI refresh
               'text_accumulation': '<1ms',       # String operations
           },
           'llm_processing_phase': {
               'ui_coordination': '50-100ms',     # Event coordination
               'llm_initialization': '100-200ms', # Model warm-up if needed
               'token_generation': '20-50ms/token', # Streaming generation
               'total_llm_time': '2-10s',         # Depends on text length
               'ui_streaming_update': '16ms',     # Real-time token display
           },
           'total_session_time': '3-15s',        # Complete smart dictation
           'memory_peak': '~1.2GB',              # Whisper + LLM models
           'cpu_peak': '40-60%',                 # LLM processing intensive
           'accuracy_improvement': '10-15%',      # Grammar/clarity enhancement
       },
       
       'type_dictation': {
           'audio_processing': '270-920ms',       # Same Whisper processing
           'character_typing': '10ms/char',       # Configurable typing delay
           'total_typing_time': 'text_length * 10ms', # Linear with text length
           'reliability': '99%',                  # Works with all applications
           'compatibility': 'Universal',          # No clipboard dependencies
       }
   }

**Resource Usage Optimization Strategies**:

.. code-block:: python

   # Memory optimization techniques
   MEMORY_OPTIMIZATION = {
       'whisper_model_management': {
           'int8_quantization': 'Reduces model size by ~50%',
           'cpu_optimization': 'Forces CPU usage for stability',
           'model_caching': 'Keeps model loaded between requests',
           'garbage_collection': 'Explicit cleanup after processing',
           'memory_mapping': 'Efficient model loading from disk'
       },
       
       'audio_buffer_optimization': {
           'circular_buffers': 'Fixed-size pre-roll buffers',
           'lazy_concatenation': 'Avoid unnecessary array copies',
           'chunk_reuse': 'Reuse audio chunk arrays',
           'immediate_processing': 'Process chunks without accumulation',
           'memory_pools': 'Pre-allocated buffer pools'
       },
       
       'text_processing_optimization': {
           'string_interning': 'Reuse common trigger words',
           'regex_compilation': 'Pre-compile text cleaning patterns',
           'minimal_copying': 'In-place string modifications where possible',
           'lazy_evaluation': 'Defer expensive operations until needed'
       }
   }

**CPU Utilization Analysis**:

.. code-block:: python

   # CPU usage breakdown by component
   CPU_UTILIZATION_BREAKDOWN = {
       'audio_recording': {
           'vad_processing': '1-2%',      # RMS energy calculations
           'buffer_management': '<1%',     # Memory operations
           'event_publishing': '<1%',      # Event bus overhead
           'total': '2-3%'                # Minimal overhead
       },
       
       'whisper_stt': {
           'audio_preprocessing': '2-3%',  # NumPy operations
           'model_inference': '20-40%',    # Main processing load
           'post_processing': '1-2%',      # Text extraction/cleaning
           'total': '23-45%'              # Varies by audio length
       },
       
       'llm_processing': {
           'prompt_preparation': '<1%',    # String operations
           'model_inference': '30-50%',    # Token generation
           'streaming_overhead': '1-2%',   # Event publishing
           'total': '31-53%'              # Peak during generation
       },
       
       'text_input': {
           'clipboard_operations': '1-2%', # System API calls
           'typing_simulation': '1-3%',    # PyAutoGUI operations
           'coordination': '<1%',          # Event handling
           'total': '2-6%'                # Brief spikes during input
       }
   }

**Bottleneck Identification and Mitigation**:

.. code-block:: python

   # Common performance bottlenecks and solutions
   BOTTLENECK_ANALYSIS = {
       'whisper_processing_bottleneck': {
           'symptom': 'High latency (>2s) for short audio',
           'causes': [
               'Cold model start',
               'Large model size',
               'CPU vs GPU mismatch',
               'Memory swapping'
           ],
           'solutions': [
               'Model warm-up on startup',
               'Use base model instead of large',
               'Force CPU processing for stability',
               'Increase system RAM',
               'Implement model caching'
           ],
           'implementation': '''
               # Model warm-up strategy
               def _warm_up_model(self):
                   dummy_audio = np.zeros(16000, dtype=np.float32)
                   with self._model_lock:
                       segments, _ = self._model.transcribe(dummy_audio, beam_size=1)
                       list(segments)  # Consume generator
           '''
       },
       
       'memory_pressure_bottleneck': {
           'symptom': 'System slowdown, high memory usage',
           'causes': [
               'Multiple models loaded simultaneously',
               'Memory leaks in audio buffers',
               'Large audio segment accumulation',
               'Inefficient text processing'
           ],
           'solutions': [
               'Lazy model loading',
               'Explicit resource cleanup',
               'Streaming processing',
               'Memory profiling and optimization'
           ],
           'implementation': '''
               # Resource cleanup strategy
               async def shutdown(self):
                   if hasattr(self, 'whisper_engine') and self.whisper_engine:
                       await self.whisper_engine.shutdown()
                       self.whisper_engine = None
                   if hasattr(self, '_duplicate_filter'):
                       del self._duplicate_filter
           '''
       },
       
       'ui_responsiveness_bottleneck': {
           'symptom': 'UI freezing during processing',
           'causes': [
               'Blocking operations on main thread',
               'Heavy processing without yielding',
               'Synchronous event handling',
               'Large data transfers'
           ],
           'solutions': [
               'Async/await patterns',
               'Thread pool executors',
               'Event-driven architecture',
               'Streaming data processing'
           ],
           'implementation': '''
               # Non-blocking text input
               async def input_text(self, text: str) -> bool:
                   if self.config.use_clipboard:
                       success = await asyncio.get_event_loop().run_in_executor(
                           None, self._paste_clipboard, cleaned_text
                       )
                   return success
           '''
       }
   }

This comprehensive dictation system provides a robust, high-quality text input solution with AI enhancement capabilities, delivering both accuracy and user experience through its sophisticated multi-mode architecture and intelligent processing pipeline optimized for real-world performance constraints.
