=====================================
Audio System Architecture Overview
=====================================

This document provides a comprehensive overview of the IRIS audio system architecture, explaining how audio flows from microphone input to text output across both command and dictation modes.

System Overview
===============

The IRIS audio system is a sophisticated, event-driven architecture designed for high-performance voice recognition with dual-mode operation:

- **Command Mode**: Ultra-low latency recognition for voice commands (optimized for speed)
- **Dictation Mode**: High-accuracy recognition for text input (optimized for accuracy)

The system employs independent audio recorders, specialized STT engines, sound recognition, and a comprehensive event bus for coordination.

High-Level Architecture
=======================

Core Components
---------------

1. **Audio Recording Layer**
   - ``SimpleAudioService``: Manages dual independent recorders
   - ``AudioRecorder``: Mode-specific audio capture and VAD (Voice Activity Detection)

2. **Speech-to-Text Layer**
   - ``StreamlinedSpeechToTextService``: Coordinates STT processing
   - ``EnhancedVoskSTT``: Fast command recognition engine
   - ``WhisperSpeechToText``: High-accuracy dictation engine

3. **Sound Recognition Layer**
   - ``StreamlinedSoundService``: Manages custom sound recognition
   - ``StreamlinedSoundRecognizer``: YAMNet-based sound classification

4. **Dictation Processing Layer**
   - ``DictationCoordinator``: Orchestrates dictation workflows
   - ``TextInputService``: Handles text input to applications
   - ``LLMService``: AI-powered text enhancement

5. **Event System**
   - ``EventBus``: Asynchronous event coordination
   - Comprehensive event types for all audio operations

Audio Flow Architecture
=======================

Complete System Flow Diagram
-----------------------------

.. mermaid::

   flowchart TD
       MIC["Microphone Input<br/>16kHz, Mono, int16"] --> DUAL["Dual Recorder System"]
       
       DUAL --> CMD_REC["Command Recorder<br/>60ms chunks - 960 samples<br/>Ultra-sensitive VAD"]
       DUAL --> DICT_REC["Dictation Recorder<br/>20ms chunks - 320 samples<br/>Balanced VAD"]
       
       CMD_REC --> CMD_VAD{"Command VAD<br/>Energy > 0.002?"}
       DICT_REC --> DICT_VAD{"Dictation VAD<br/>Energy > 0.0035?"}
       
       CMD_VAD -->|Yes| CMD_PREROLL["Pre-roll Buffer<br/>4 chunks - 240ms"]
       DICT_VAD -->|Yes| DICT_PREROLL["Pre-roll Buffer<br/>2 chunks - 40ms"]
       
       CMD_PREROLL --> CMD_COLLECT["Collect Audio Segment<br/>Adaptive timeout 0.25-0.6s"]
       DICT_PREROLL --> DICT_COLLECT["Collect Audio Segment<br/>Fixed timeout 0.5s"]
       
       CMD_COLLECT --> CMD_STREAM{"Streaming<br/>Recognition?"}
       CMD_STREAM -->|Every 2 chunks| VOSK_STREAM["Vosk Streaming<br/>Partial Results"]
       VOSK_STREAM --> INSTANT{"Instant<br/>Command?"}
       INSTANT -->|Yes| CMD_EXECUTE["Execute Command<br/>Immediately"]
       
       CMD_COLLECT --> CMD_SEGMENT["CommandAudioSegmentReadyEvent"]
       DICT_COLLECT --> DICT_SEGMENT["DictationAudioSegmentReadyEvent"]
       
       CMD_SEGMENT --> STT_SERVICE["StreamlinedSpeechToTextService"]
       DICT_SEGMENT --> STT_SERVICE
       
       STT_SERVICE --> MODE_CHECK{"System Mode?"}
       
       MODE_CHECK -->|Command Mode| CMD_PROCESSING["Command Processing Path"]
       MODE_CHECK -->|Dictation Mode| DICT_PROCESSING["Dictation Processing Path"]
       
       CMD_PROCESSING --> VOSK_STT["Vosk STT Engine<br/>Under 50ms processing"]
       DICT_PROCESSING --> WHISPER_STT["Whisper STT Engine<br/>200-1000ms processing"]
       
       VOSK_STT --> CMD_RESULT{"Text<br/>Recognized?"}
       WHISPER_STT --> DICT_RESULT{"Text<br/>Recognized?"}
       
       CMD_RESULT -->|Yes| CMD_DUPLICATE{"Duplicate<br/>Filter?"}
       CMD_RESULT -->|No| SOUND_REC["Sound Recognition<br/>YAMNet + k-NN"]
       
       DICT_RESULT -->|Yes| DICT_DUPLICATE{"Duplicate<br/>Filter?"}
       
       CMD_DUPLICATE -->|Pass| CMD_TEXT_EVENT["CommandTextRecognizedEvent"]
       DICT_DUPLICATE -->|Pass| DICT_TEXT_EVENT["DictationTextRecognizedEvent"]
       
       CMD_TEXT_EVENT --> DICT_ACTIVE{"Dictation<br/>Active?"}
       DICT_ACTIVE -->|Yes| AMBER_CHECK{"Amber<br/>Trigger?"}
       DICT_ACTIVE -->|No| CMD_PARSE["Command Parser"]
       
       AMBER_CHECK -->|Yes| STOP_DICTATION["Stop Dictation"]
       AMBER_CHECK -->|No| IGNORE["Ignore Command"]
       
       DICT_TEXT_EVENT --> DICTATION_COORDINATOR["DictationCoordinator"]
       
       DICTATION_COORDINATOR --> DICT_MODE_CHECK{"Dictation<br/>Mode?"}
       DICT_MODE_CHECK -->|Standard/Type| TEXT_INPUT["TextInputService<br/>Direct Input"]
       DICT_MODE_CHECK -->|Smart| SMART_PROCESSING["Smart Dictation<br/>LLM Processing"]
       
       SMART_PROCESSING --> LLM_SERVICE["LLMService<br/>Grammar and Clarity"]
       LLM_SERVICE --> LLM_RESULT["Enhanced Text"]
       LLM_RESULT --> TEXT_INPUT
       
       TEXT_INPUT --> CLIPBOARD{"Use<br/>Clipboard?"}
       CLIPBOARD -->|Yes| PASTE["Ctrl+V Paste"]
       CLIPBOARD -->|No| TYPE["Character Typing"]
       
       SOUND_REC --> YAMNET["YAMNet Embeddings<br/>1024-dimensional"]
       YAMNET --> KNN["k-NN Classification<br/>Cosine Similarity"]
       KNN --> SOUND_RESULT{"Custom Sound<br/>Detected?"}
       SOUND_RESULT -->|Yes| SOUND_COMMAND["CustomSoundRecognizedEvent"]
       
       PASTE --> APP["Target Application"]
       TYPE --> APP
       SOUND_COMMAND --> CMD_PARSE
       CMD_PARSE --> APP

Command Mode Detailed Flow
--------------------------

The command mode is optimized for ultra-low latency response with sophisticated streaming recognition:

.. code-block:: python

   # Command Mode Audio Processing Pipeline
   class AudioRecorder:
       def _recording_thread(self):
           while self._is_recording:
               # 1. Continuous audio capture
               data, _ = self._stream.read(self.chunk_size)  # 960 samples = 60ms
               energy = self._calculate_energy(data)
               
               # 2. Pre-roll buffer management
               pre_roll_buffer.append(data)
               if len(pre_roll_buffer) > self.pre_roll_chunks:  # Keep last 4 chunks
                   pre_roll_buffer.pop(0)
               
               # 3. Speech detection with ultra-sensitive threshold
               if energy > self.energy_threshold:  # 0.002 - catches soft consonants
                   speech_detected = True
                   audio_buffer.extend(pre_roll_buffer)  # Include 240ms pre-roll
                   
                   # 4. Segment collection with streaming recognition
                   while collecting_speech:
                       data, _ = self._stream.read(self.chunk_size)
                       audio_buffer.append(data)
                       chunks_collected += 1
                       
                       # 5. Streaming recognition every 2nd chunk after initial 3
                       if (self.enable_streaming and chunks_collected >= 3 and 
                           chunks_collected % 2 == 0):
                           current_audio = np.concatenate(audio_buffer)
                           recognized_command = self.on_streaming_chunk(
                               current_audio.tobytes(), False
                           )
                           
                           # 6. Instant command execution
                           if recognized_command and self._is_instant_command(recognized_command):
                               self.logger.info(f"Instant command '{recognized_command}' detected")
                               break  # Stop recording immediately
                       
                       # 7. Adaptive silence timeout
                       current_timeout = self._get_timeout(chunks_collected)
                       if energy < self.silence_threshold:
                           if silence_start is None:
                               silence_start = time.time()
                           elif time.time() - silence_start > current_timeout:
                               break  # End of speech
                       else:
                           silence_start = None

**Adaptive Timeout Algorithm:**

.. code-block:: python

   def _get_timeout(self, chunks_collected: int) -> float:
       """Progressive timeout based on speech duration"""
       if chunks_collected <= 4:      # < 240ms - likely single word
           return 0.25                # Quick timeout for "click", "enter"
       elif chunks_collected <= 8:    # < 480ms - short phrase
           return self.silence_timeout # Standard timeout (0.35s)
       else:                          # > 480ms - longer command
           return 0.6                 # Extended timeout for "open file browser"

**Smart Timeout Integration:**

.. code-block:: python

   def _is_instant_command(self, recognized_text: str) -> bool:
       """Context-aware instant command detection"""
       if not self._smart_timeout_manager:
           return False
           
       # Get timeout based on command knowledge
       timeout = self._smart_timeout_manager.get_timeout_for_text(recognized_text)
       
       # Commands with ≤50ms timeout execute instantly
       return timeout <= 0.05

**Performance Characteristics:**
- **Audio Latency**: 60ms (chunk size)
- **VAD Latency**: <1ms (RMS calculation)
- **STT Latency**: <50ms (Vosk processing)
- **Total End-to-End**: <100ms
- **Streaming Updates**: Every 120ms during recording
- **Memory Usage**: ~100MB (Vosk model)

Dictation Mode Detailed Flow
----------------------------

Dictation mode prioritizes accuracy and natural speech handling with extended processing:

.. mermaid::

   flowchart TD
       TRIGGER["Dictation Trigger<br/>green, smart green, type"] --> MODE_SWITCH["Audio Mode Switch"]
       MODE_SWITCH --> BOTH_ACTIVE["Both Recorders Active<br/>Command: Amber detection<br/>Dictation: Text capture"]
       
       BOTH_ACTIVE --> DICT_AUDIO["Dictation Audio Capture<br/>20ms chunks, 0.0035 threshold"]
       BOTH_ACTIVE --> CMD_AMBER["Command Amber Detection<br/>60ms chunks, 0.002 threshold"]
       
       DICT_AUDIO --> WHISPER["Whisper STT Processing<br/>Advanced preprocessing"]
       CMD_AMBER --> AMBER_CHECK{"Amber<br/>Detected?"}
       
       WHISPER --> TEXT_CLEAN["Text Cleaning<br/>Remove triggers, normalize"]
       AMBER_CHECK -->|Yes| STOP_SIGNAL["Stop Dictation Signal"]
       
       TEXT_CLEAN --> MODE_BRANCH{"Dictation<br/>Mode?"}
       
       MODE_BRANCH -->|Standard| DIRECT_INPUT["Direct Text Input<br/>TextInputService"]
       MODE_BRANCH -->|Type| DIRECT_INPUT
       MODE_BRANCH -->|Smart| ACCUMULATE["Accumulate Text<br/>Session tracking"]
       
       ACCUMULATE --> DISPLAY["SmartDictationTextDisplayEvent<br/>Show in UI"]
       
       STOP_SIGNAL --> SMART_CHECK{"Smart Mode<br/>Active?"}
       SMART_CHECK -->|Yes| LLM_PREP["Prepare LLM Processing<br/>Generate session ID"]
       SMART_CHECK -->|No| FINALIZE["Finalize Session"]
       
       LLM_PREP --> UI_SIGNAL["LLMProcessingStartedEvent<br/>Wait for UI ready"]
       UI_SIGNAL --> UI_READY["LLMProcessingReadyEvent<br/>UI coordination"]
       UI_READY --> LLM_PROCESS["LLM Processing<br/>Grammar and clarity"]
       
       LLM_PROCESS --> STREAM_TOKENS["Stream LLM Tokens<br/>Real-time display"]
       STREAM_TOKENS --> ENHANCED_TEXT["Enhanced Text Result"]
       
       ENHANCED_TEXT --> DIRECT_INPUT
       DIRECT_INPUT --> CLIPBOARD_PASTE["Clipboard Paste<br/>or Character Typing"]

**Dictation Audio Processing Pipeline:**

.. code-block:: python

   # Dictation Mode Processing in AudioRecorder
   def _recording_thread(self):
       # Dictation mode configuration
       self.chunk_size = 320           # 20ms chunks for responsiveness
       self.energy_threshold = 0.0035  # Balanced sensitivity
       self.silence_timeout = 0.5      # Natural speech pauses
       self.pre_roll_chunks = 2        # 40ms pre-roll for quick response
       
       while self._is_recording and self._is_active:
           # Continuous capture with natural speech handling
           data, _ = self._stream.read(self.chunk_size)
           energy = self._calculate_energy(data)
           
           # Progressive silence detection for natural speech patterns
           if self._is_speech_continuation(energy, previous_energies):
               continue_recording = True
           elif self._detect_sentence_boundary(energy_pattern):
               # Natural pause detected - prepare to segment
               prepare_segmentation = True

**Text Processing and Cleaning:**

.. code-block:: python

   async def _handle_dictation_text(self, event: DictationTextRecognizedEvent):
       """Centralized dictation text processing with mode-specific handling"""
       text = event.text.strip()
       if not text:
           return
           
       with self._lock:
           session = self._current_session
           
       if not session:
           return
       
       # Prevent processing during smart mode LLM processing
       if session.mode == DictationMode.SMART and getattr(session, 'is_processing', False):
           logger.debug("Ignoring dictation text - smart session already processing")
           return
           
       # Advanced text cleaning with context preservation
       cleaned_text = self._clean_text(text)
       if not cleaned_text:
           return
           
       # Accumulate text for session tracking
       session.accumulated_text = self._append_text(session.accumulated_text, cleaned_text)
       
       # Mode-specific processing
       if session.mode == DictationMode.SMART:
           # Smart dictation: display cleaned text, accumulate for LLM
           await self._publish_event(SmartDictationTextDisplayEvent(text=cleaned_text))
       else:
           # Standard/Type dictation: input text immediately
           await self.text_service.input_text(cleaned_text)

**Performance Characteristics:**
- **Audio Latency**: 20ms (chunk size)
- **VAD Latency**: <1ms (RMS calculation) 
- **STT Latency**: 200-1000ms (Whisper processing)
- **Total End-to-End**: 300-1200ms
- **Memory Usage**: ~500MB (Whisper base model)
- **Accuracy**: ~95% (natural language)

Sound Recognition Processing Flow
---------------------------------

Sound recognition uses machine learning for custom sound detection when no speech is recognized:

.. mermaid::

   flowchart TD
       AUDIO_CHUNK["Audio Chunk<br/>No speech detected"] --> PREPROCESS["Audio Preprocessing<br/>Silence trim, normalize"]
       PREPROCESS --> YAMNET["YAMNet Feature Extraction<br/>1024-dimensional embeddings"]
       
       YAMNET --> TRAINING_CHECK{"Training<br/>Mode?"}
       TRAINING_CHECK -->|Yes| COLLECT["Collect Training Sample<br/>Store with label"]
       TRAINING_CHECK -->|No| CLASSIFY["Classification Pipeline"]
       
       COLLECT --> SAMPLE_COUNT{"Enough<br/>Samples?"}
       SAMPLE_COUNT -->|Yes| TRAIN_MODEL["Train k-NN Model<br/>Add ESC-50 negatives"]
       SAMPLE_COUNT -->|No| WAIT_MORE["Wait for more samples"]
       
       CLASSIFY --> SCALE["StandardScaler<br/>Normalize features"]
       SCALE --> KNN["k-NN Classification<br/>k=5, cosine similarity"]
       
       KNN --> CONFIDENCE{"Confidence greater than<br/>0.7 threshold?"}
       CONFIDENCE -->|No| REJECT["Reject classification"]
       CONFIDENCE -->|Yes| VOTING["Majority Voting<br/>Among top-k neighbors"]
       
       VOTING --> CUSTOM_CHECK{"Custom Sound<br/>vs ESC-50?"}
       CUSTOM_CHECK -->|ESC-50| REJECT
       CUSTOM_CHECK -->|Custom| VOTE_THRESHOLD{"Vote ratio greater than<br/>0.6 threshold?"}
       
       VOTE_THRESHOLD -->|Yes| MAPPING["Check Command Mapping"]
       VOTE_THRESHOLD -->|No| REJECT
       
       MAPPING --> SOUND_EVENT["CustomSoundRecognizedEvent<br/>Execute mapped command"]

**YAMNet Feature Extraction:**

.. code-block:: python

   def _extract_embedding(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
       """Extract YAMNet embedding with comprehensive preprocessing"""
       try:
           # 1. Audio preprocessing pipeline
           processed_audio = self.preprocessor.preprocess_audio(audio, sr)
           
           # 2. Convert to TensorFlow tensor
           audio_tensor = tf.convert_to_tensor(processed_audio, dtype=tf.float32)
           
           # 3. YAMNet inference - returns (scores, embeddings, spectrogram)
           _, embeddings, _ = self.yamnet_model(audio_tensor)
           
           # 4. Temporal aggregation - average embeddings across time
           # Shape: (time_frames, 1024) -> (1024,)
           embedding = tf.reduce_mean(embeddings, axis=0).numpy()
           
           return embedding
           
       except Exception as e:
           logger.error(f"Failed to extract embedding: {e}")
           return None

**k-NN Classification with Noise Rejection:**

.. code-block:: python

   def recognize_sound(self, audio: np.ndarray, sr: int) -> Optional[Tuple[str, float]]:
       """Advanced sound recognition with noise rejection"""
       if len(self.embeddings) == 0:
           return None
       
       # 1. Extract and scale embedding
       embedding = self._extract_embedding(audio, sr)
       scaled_embedding = self.scaler.transform(embedding.reshape(1, -1))[0]
       
       # 2. Calculate cosine similarities to all training samples
       similarities = cosine_similarity(scaled_embedding.reshape(1, -1), self.embeddings)[0]
       
       # 3. Get top-k neighbors
       top_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
       top_similarities = similarities[top_indices]
       top_labels = [self.labels[i] for i in top_indices]
       
       # 4. Confidence and voting with ESC-50 rejection
       best_similarity = top_similarities[0]
       if best_similarity < self.confidence_threshold:  # 0.7
           return None
       
       # 5. Prioritize custom sounds over ESC-50 background sounds
       custom_labels = [label for label in top_labels if not label.startswith('esc50_')]
       if not custom_labels:
           return None
       
       # 6. Majority voting among custom sounds
       votes = Counter(custom_labels)
       majority_label, vote_count = votes.most_common(1)[0]
       vote_ratio = vote_count / len(custom_labels)
       
       if vote_ratio >= self.vote_threshold:  # 0.6
           majority_indices = [i for i, label in enumerate(top_labels) 
                             if label == majority_label]
           confidence = np.mean([top_similarities[i] for i in majority_indices])
           return majority_label, confidence
       
       return None

Mode Switching and System Coordination
=======================================

The system implements sophisticated mode switching with comprehensive state management:

.. mermaid::

   stateDiagram-v2
       [*] --> CommandMode: System Startup
       
       CommandMode --> DictationStandard: green trigger
       CommandMode --> DictationSmart: smart green trigger  
       CommandMode --> DictationType: type trigger
       
       DictationStandard --> CommandMode: amber trigger
       DictationSmart --> LLMProcessing: amber trigger
       DictationType --> CommandMode: amber trigger
       
       LLMProcessing --> CommandMode: Processing complete
       
       state CommandMode {
           [*] --> CommandRecorderActive
           CommandRecorderActive --> VoskProcessing: Audio segment
           VoskProcessing --> CommandExecution: Text recognized
           VoskProcessing --> SoundRecognition: No text
           CommandExecution --> [*]
           SoundRecognition --> CommandExecution: Custom sound
           SoundRecognition --> [*]: No sound
       }
       
       state DictationStandard {
           [*] --> BothRecordersActive
           BothRecordersActive --> WhisperProcessing: Dictation audio
           BothRecordersActive --> AmberDetection: Command audio
           WhisperProcessing --> DirectTextInput: Text recognized
           DirectTextInput --> [*]
           AmberDetection --> [*]: Stop signal
       }
       
       state DictationSmart {
           [*] --> BothRecordersActive
           BothRecordersActive --> WhisperProcessing: Dictation audio
           BothRecordersActive --> AmberDetection: Command audio
           WhisperProcessing --> TextAccumulation: Text recognized
           TextAccumulation --> UIDisplay: Show text
           UIDisplay --> [*]
           AmberDetection --> [*]: Stop signal
       }

**Event-Driven Mode Coordination:**

.. code-block:: python

   # Audio Service Mode Management
   async def _handle_audio_mode_change_request(self, event: AudioModeChangeRequestEvent):
       """Coordinate recorder activation based on system mode"""
       try:
           with self._lock:
               if event.mode == "dictation":
                   self._is_dictation_mode = True
                   # Both recorders active: command for amber detection, dictation for text
                   self._command_recorder.set_active(True)
                   self._dictation_recorder.set_active(True)
                   logger.info("Dictation mode: both recorders active")
               elif event.mode == "command":
                   self._is_dictation_mode = False
                   # Only command recorder active
                   self._command_recorder.set_active(True)
                   self._dictation_recorder.set_active(False)
                   logger.info("Command mode: only command recorder active")
       except Exception as e:
           logger.error(f"Error handling audio mode change request: {e}", exc_info=True)

   # STT Service Mode Management  
   async def _handle_dictation_mode_change(self, event_data: DictationModeDisableOthersEvent):
       """Handle dictation mode changes in STT processing"""
       with self._processing_lock:
           old_state = self._dictation_active
           self._dictation_active = event_data.dictation_mode_active
           
           if self._dictation_active:
               logger.info("STT service now in DICTATION mode - command audio will only check for amber triggers")
           else:
               logger.info("STT service now in COMMAND mode - normal command processing enabled")

Event-Driven Coordination
=========================

The system uses a comprehensive event bus for loose coupling and coordination:

Core Audio Events
-----------------

- ``CommandAudioSegmentReadyEvent``: Command mode audio ready for processing
- ``DictationAudioSegmentReadyEvent``: Dictation mode audio ready for processing
- ``ProcessAudioChunkForSoundRecognitionEvent``: Audio for sound recognition
- ``AudioModeChangeRequestEvent``: Request to switch between command/dictation modes

STT Events
----------

- ``CommandTextRecognizedEvent``: Text recognized in command mode
- ``DictationTextRecognizedEvent``: Text recognized in dictation mode
- ``STTProcessingStartedEvent/CompletedEvent``: Processing lifecycle events

Dictation Events
----------------

- ``DictationModeDisableOthersEvent``: Coordinate mode switching
- ``SmartDictationStartedEvent/StoppedEvent``: Smart dictation lifecycle
- ``LLMProcessingStartedEvent/CompletedEvent``: AI processing coordination
- ``LLMTokenGeneratedEvent``: Real-time LLM token streaming

Sound Events
------------

- ``CustomSoundRecognizedEvent``: Custom sound detection
- ``SoundTrainingRequestEvent``: Training mode initiation
- Various sound management events for CRUD operations

Configuration System
====================

The audio system is highly configurable through Pydantic models:

Audio Configuration (``AudioConfig``)
-------------------------------------

- Sample rates, chunk sizes, device selection
- Dual-mode processing parameters
- Command vs dictation optimizations

VAD Configuration (``VADConfig``)
---------------------------------

- Energy thresholds for different modes
- Silence timeouts and recording durations
- Progressive silence detection parameters
- Pre-roll buffer configurations

STT Configuration (``STTConfig``)
---------------------------------

- Engine selection (Vosk/Whisper)
- Model paths and device settings
- Mode-specific processing parameters
- Debounce and duplicate filtering settings

Threading and Performance
=========================

Threading Model
---------------

1. **Independent Recorder Threads**
   - Each recorder runs in its own thread
   - Continuous operation with active/inactive states
   - Thread-safe audio capture and processing

2. **Event Processing**
   - Asynchronous event handling
   - Thread-safe event publication
   - Cross-thread coordination via event bus

3. **STT Processing**
   - Separate processing threads for each engine
   - Thread-safe model access with locks
   - Concurrent processing capabilities

Performance Optimizations
--------------------------

1. **Command Mode Optimizations**
   - 60ms chunks for ultra-low latency
   - Streaming recognition for instant commands
   - Smart timeout management
   - Aggressive debouncing (20ms)

2. **Memory Management**
   - Efficient audio buffering
   - Pre-roll buffer management
   - Model memory optimization
   - Resource cleanup on shutdown

3. **Processing Efficiency**
   - Mode-specific configurations
   - Adaptive thresholds and timeouts
   - Duplicate filtering to reduce processing
   - Concurrent audio and STT processing

Error Handling and Resilience
==============================

The system implements comprehensive error handling:

1. **Audio Capture Resilience**
   - Device failure recovery
   - Stream cleanup on errors
   - Graceful degradation

2. **STT Error Handling**
   - Engine fallback mechanisms
   - Processing timeout handling
   - Model loading error recovery

3. **Event System Reliability**
   - Event publication error handling
   - Subscription management
   - Graceful service degradation

Performance Characteristics and Bottlenecks
===========================================

The system is designed with specific performance characteristics for each processing path:

**Command Mode Performance Profile:**

.. code-block:: python

   # Performance metrics for command mode
   COMMAND_MODE_METRICS = {
       'audio_latency': '60ms',      # Chunk size: 960 samples at 16kHz
       'vad_latency': '<1ms',        # RMS energy calculation
       'stt_latency': '<50ms',       # Vosk processing time
       'total_latency': '<100ms',    # End-to-end response time
       'memory_usage': '~100MB',     # Vosk model footprint
       'cpu_usage': '~5%',           # Single core utilization
       'accuracy': '~85%',           # Command vocabulary accuracy
       'streaming_enabled': True,    # Real-time partial results
       'duplicate_filter': '300ms'   # Aggressive filtering
   }

**Dictation Mode Performance Profile:**

.. code-block:: python

   # Performance metrics for dictation mode
   DICTATION_MODE_METRICS = {
       'audio_latency': '20ms',      # Chunk size: 320 samples at 16kHz
       'vad_latency': '<1ms',        # RMS energy calculation
       'stt_latency': '200-1000ms',  # Whisper processing time
       'total_latency': '300-1200ms', # End-to-end response time
       'memory_usage': '~500MB',     # Whisper base model footprint
       'cpu_usage': '~25%',          # Multi-core utilization
       'accuracy': '~95%',           # Natural language accuracy
       'streaming_enabled': False,   # Batch processing
       'duplicate_filter': '4000ms'  # Extended filtering
   }

**Sound Recognition Performance Profile:**

.. code-block:: python

   # Performance metrics for sound recognition
   SOUND_RECOGNITION_METRICS = {
       'feature_extraction': '50-100ms',  # YAMNet embedding extraction
       'classification': '<10ms',         # k-NN with cosine similarity
       'total_latency': '60-120ms',       # End-to-end sound recognition
       'memory_usage': '~200MB',          # YAMNet model + embeddings
       'training_time': '2-5s',           # 5-sample training session
       'accuracy': '>95%',                # Well-trained custom sounds
       'noise_rejection': 'High',         # ESC-50 negative examples
       'scalability': '50+ sounds'        # Practical custom sound limit
   }

**System-Wide Resource Usage:**

.. code-block:: python

   # Total system resource requirements
   TOTAL_SYSTEM_RESOURCES = {
       'peak_memory': '~800MB',      # All models loaded
       'steady_memory': '~600MB',    # Normal operation
       'cpu_command': '~10%',        # Command mode operation
       'cpu_dictation': '~30%',      # Dictation mode operation
       'disk_models': '~1GB',        # All audio models
       'disk_cache': '~50MB',        # Audio processing cache
       'startup_time': '3-5s',       # Model loading time
       'shutdown_time': '<1s'        # Cleanup time
   }

This comprehensive architecture provides a robust, high-performance audio processing system capable of handling both rapid voice commands and accurate dictation while maintaining optimal resource utilization and system responsiveness.
