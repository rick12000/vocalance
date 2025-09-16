==============================
Speech-to-Text System
==============================

This document provides comprehensive technical documentation of the IRIS Speech-to-Text (STT) system, covering the service architecture, engine implementations, and processing pipelines for both command and dictation modes.

Overview
========

The IRIS STT system implements a sophisticated dual-engine architecture optimized for different use cases:

- **Vosk Engine**: Ultra-fast command recognition with streaming capabilities
- **Whisper Engine**: High-accuracy dictation with advanced preprocessing

The system uses mode-aware processing, intelligent duplicate filtering, and comprehensive event coordination to deliver optimal performance for both rapid voice commands and accurate text dictation.

StreamlinedSpeechToTextService
==============================

The ``StreamlinedSpeechToTextService`` is the central coordinator for all STT operations, managing engine selection, mode switching, and result processing.

Architecture
------------

**Core Components**:

.. code-block:: python

   class StreamlinedSpeechToTextService:
       def __init__(self, event_bus: EventBus, config: GlobalAppConfig):
           # STT engines
           self.vosk_engine = None      # Fast command recognition
           self.whisper_engine = None   # High-accuracy dictation
           
           # Processing state
           self._dictation_active = False
           self._processing_lock = threading.RLock()
           
           # Duplicate detection
           self._duplicate_filter = DuplicateTextFilter(
               cache_size=5, 
               duplicate_threshold_ms=1000
           )

**Key Features**:
- Dual-engine management with lazy initialization
- Mode-aware processing with dictation state tracking
- Thread-safe operation with RLock protection
- Comprehensive event handling and publishing
- Intelligent duplicate filtering

Engine Initialization
---------------------

The service initializes both STT engines at startup with optimized configurations:

**Vosk Engine Initialization**:

.. code-block:: python

   # Initialize Vosk engine for fast command recognition
   self.vosk_engine = EnhancedVoskSTT(
       model_path=self.config.model_paths.vosk_model,
       sample_rate=self.stt_config.sample_rate,
       config=self.config
   )

**Whisper Engine Initialization**:

.. code-block:: python

   # Initialize Whisper engine for high-accuracy dictation
   self.whisper_engine = WhisperSpeechToText(
       model_name=self.stt_config.whisper_model,    # "base" for balance
       device=self.stt_config.whisper_device,       # "cpu" for stability
       sample_rate=self.stt_config.sample_rate,     # 16kHz
       config=self.stt_config
   )

Mode-Aware Processing
---------------------

The service implements intelligent mode switching based on dictation state:

**Command Audio Processing**:

.. code-block:: python

   async def _handle_command_audio_segment(self, event_data: CommandAudioSegmentReadyEvent):
       # If in dictation mode, only check for amber trigger words
       if self._dictation_active:
           vosk_result = self.vosk_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
           
           if self._is_amber_trigger(vosk_result):
               # Publish stop word for dictation termination
               await self._publish_recognition_result(vosk_result, 0, "vosk", STTMode.COMMAND)
           return
       
       # Normal command processing (only when NOT in dictation mode)
       processing_start = time.time()
       recognized_text = self.vosk_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
       processing_time = (time.time() - processing_start) * 1000
       
       if recognized_text and recognized_text.strip():
           if not self._duplicate_filter.is_duplicate(recognized_text):
               await self._publish_recognition_result(recognized_text, processing_time, "vosk", STTMode.COMMAND)
       else:
           # No speech detected - try sound recognition
           await self._publish_sound_recognition_event(event_data.audio_bytes, event_data.sample_rate)

**Dictation Audio Processing**:

.. code-block:: python

   async def _handle_dictation_audio_segment(self, event_data: DictationAudioSegmentReadyEvent):
       processing_start = time.time()
       recognized_text = self.whisper_engine.recognize(event_data.audio_bytes, event_data.sample_rate)
       processing_time = (time.time() - processing_start) * 1000
       
       if recognized_text and recognized_text.strip():
           if not self._duplicate_filter.is_duplicate(recognized_text):
               await self._publish_recognition_result(recognized_text, processing_time, "whisper", STTMode.DICTATION)

Event Handling
--------------

The service handles comprehensive event coordination:

**Event Subscriptions**:

.. code-block:: python

   def setup_subscriptions(self):
       self.event_bus.subscribe(CommandAudioSegmentReadyEvent, self._handle_command_audio_segment)
       self.event_bus.subscribe(DictationAudioSegmentReadyEvent, self._handle_dictation_audio_segment)
       self.event_bus.subscribe(DictationModeDisableOthersEvent, self._handle_dictation_mode_change)

**Mode Change Handling**:

.. code-block:: python

   async def _handle_dictation_mode_change(self, event_data: DictationModeDisableOthersEvent):
       with self._processing_lock:
           old_state = self._dictation_active
           self._dictation_active = event_data.dictation_mode_active
           
           if self._dictation_active:
               self.logger.info("STT service now in DICTATION mode - command audio will only check for amber triggers")
           else:
               self.logger.info("STT service now in COMMAND mode - normal command processing enabled")

EnhancedVoskSTT
===============

The ``EnhancedVoskSTT`` class provides ultra-fast speech recognition optimized for voice commands.

Architecture
------------

**Core Components**:

.. code-block:: python

   class EnhancedVoskSTT:
       def __init__(self, model_path: str, sample_rate: int, config: GlobalAppConfig):
           # Core Vosk components
           self._model = vosk.Model(model_path)
           self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
           self._recognizer_lock = threading.Lock()
           
           # Lightweight duplicate filtering
           self._duplicate_filter = DuplicateTextFilter(
               cache_size=5, 
               duplicate_threshold_ms=300  # Aggressive 300ms filtering
           )

**Performance Characteristics**:
- **Processing Time**: <50ms for typical commands
- **Memory Usage**: ~100MB model footprint
- **Latency**: Ultra-low with streaming support
- **Accuracy**: Optimized for command vocabulary

Recognition Pipeline
--------------------

**Standard Recognition**:

.. code-block:: python

   def recognize(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
       if not audio_bytes:
           return ""
       
       with self._recognizer_lock:
           self._recognizer.Reset()  # Clear previous state
           
           try:
               self._recognizer.AcceptWaveform(audio_bytes)
               result = json.loads(self._recognizer.FinalResult())
               recognized_text = result.get("text", "")
               
               if not recognized_text:
                   return ""
                   
               # Check for duplicates
               if self._duplicate_filter.is_duplicate(recognized_text):
                   return ""
               
               return recognized_text
               
           except Exception as e:
               logger.error(f"Recognition error: {e}")
               return ""

**Streaming Recognition**:

.. code-block:: python

   def recognize_streaming(self, audio_bytes: bytes, is_final: bool = False) -> Optional[str]:
       with self._recognizer_lock:
           try:
               if self._recognizer.AcceptWaveform(audio_bytes):
                   # Final result available
                   result = json.loads(self._recognizer.Result())
                   recognized_text = result.get("text", "")
               else:
                   # Partial result
                   partial_result = json.loads(self._recognizer.PartialResult())
                   recognized_text = partial_result.get("partial", "")
               
               if recognized_text:
                   normalized_text = recognized_text.lower().strip()
                   if not self._is_nonsense(normalized_text):
                       return normalized_text
               
               return None

**Nonsense Filtering**:

.. code-block:: python

   def _is_nonsense(self, text: str) -> bool:
       """Check if text appears to be nonsense"""
       if not text or len(text.strip()) < 2:
           return True
       
       # Check for repetitive patterns
       words = text.split()
       if len(words) > 1 and len(set(words)) == 1:
           return True  # All words are identical
       
       return False

WhisperSpeechToText
===================

The ``WhisperSpeechToText`` class provides high-accuracy speech recognition optimized for dictation.

Architecture
------------

**Core Components**:

.. code-block:: python

   class WhisperSpeechToText:
       def __init__(self, model_name: str = "base", device: str = "cpu", sample_rate: int = 16000, config=None):
           self._model_name = model_name
           self._device = "cpu"  # Force CPU for stability
           self._sample_rate = sample_rate
           
           # Thread safety
           self._model_lock = threading.RLock()
           
           # Quality settings from config
           self._beam_size = getattr(config, 'whisper_beam_size', 5)
           self._temperature = getattr(config, 'whisper_temperature', 0.0)
           self._no_speech_threshold = getattr(config, 'whisper_no_speech_threshold', 0.6)
           
           # Use int8 for CPU efficiency
           self._compute_type = "int8"

**Performance Characteristics**:
- **Processing Time**: 200-1000ms depending on audio length
- **Memory Usage**: ~500MB-2GB depending on model size
- **Accuracy**: High accuracy with advanced language modeling
- **Quality**: Advanced preprocessing and normalization

Model Initialization
--------------------

**faster-whisper Integration**:

.. code-block:: python

   def _load_model(self):
       """Load the faster-whisper model"""
       try:
           self._model = WhisperModel(
               self._model_name,
               device=self._device,
               compute_type=self._compute_type,
               cpu_threads=4,
               num_workers=1
           )
       except Exception as e:
           logger.error(f"Failed to load faster-whisper model: {e}")
           raise

**Model Warm-up**:

.. code-block:: python

   def _warm_up_model(self):
       """Warm up the model with dummy audio"""
       try:
           dummy_audio = np.zeros(16000, dtype=np.float32)
           
           with self._model_lock:
               segments, _ = self._model.transcribe(dummy_audio, beam_size=1, vad_filter=False)
               list(segments)  # Consume generator
               
       except Exception as e:
           logger.warning(f"Failed to warm up model: {e}")

Recognition Pipeline
--------------------

**Audio Preprocessing**:

.. code-block:: python

   def _prepare_audio(self, audio_bytes: bytes) -> np.ndarray:
       """Convert audio bytes to numpy array"""
       audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
       return audio_np

**Transcription Options**:

.. code-block:: python

   def _get_transcription_options(self, audio_duration: float) -> Dict[str, Any]:
       """Get transcription options based on audio duration"""
       options = {
           "language": "en",
           "beam_size": self._beam_size,
           "temperature": self._temperature,
           "no_speech_threshold": self._no_speech_threshold,
           "condition_on_previous_text": False,
           "word_timestamps": True,
           "vad_filter": True,
           "vad_parameters": {
               'min_silence_duration_ms': 4000,
               'speech_pad_ms': 500
           }
       }
       
       # Adjust beam size for shorter segments
       if audio_duration < 5.0:
           options["beam_size"] = max(1, self._beam_size - 1)
       
       return options

**Main Recognition Method with Detailed Processing**:

.. code-block:: python

   def recognize(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
       """
       Comprehensive Whisper recognition with advanced preprocessing and quality control
       
       Processing Pipeline:
       1. Input validation and duration filtering
       2. Audio preprocessing and normalization  
       3. Dynamic transcription parameter selection
       4. faster-whisper inference with VAD
       5. Multi-segment text extraction and confidence calculation
       6. Advanced text normalization and cleaning
       7. Duplicate detection and filtering
       """
       
       # 1. Input validation
       if not audio_bytes or not self._model:
           return ""
       
       # Sample rate validation
       if sample_rate and sample_rate != self._sample_rate:
           logger.warning(f"Sample rate mismatch. Expected {self._sample_rate}, got {sample_rate}")
       
       # Calculate duration and apply minimum threshold
       duration_sec = len(audio_bytes) / (self._sample_rate * 2)  # 2 bytes per int16 sample
       if duration_sec < 0.3:  # Skip very short segments
           logger.debug(f"Skipping short audio segment: {duration_sec:.3f}s < 0.3s minimum")
           return ""

       with self._model_lock:
           try:
               recognition_start = time.time()
               
               # 2. Audio preprocessing
               audio_np = self._prepare_audio(audio_bytes)
               logger.debug(f"Prepared audio: {len(audio_np)} samples, duration: {duration_sec:.3f}s")
               
               # 3. Dynamic transcription options based on audio characteristics
               options = self._get_transcription_options(duration_sec)
               logger.debug(f"Transcription options: beam_size={options['beam_size']}, "
                           f"vad_filter={options['vad_filter']}")
               
               # 4. faster-whisper inference with built-in VAD
               segments, info = self._model.transcribe(audio_np, **options)
               
               # Log transcription info if available
               if hasattr(info, 'language_probability'):
                   logger.debug(f"Language detection: {info.language} "
                               f"(confidence: {info.language_probability:.3f})")
               
               # 5. Extract text from segments with confidence calculation
               recognized_text, avg_confidence = self._extract_text_from_segments(segments)
               
               recognition_time = time.time() - recognition_start
               
               if not recognized_text:
                   logger.debug("No text extracted from Whisper segments")
                   return ""
               
               # 6. Advanced text normalization
               original_text = recognized_text
               recognized_text = self._normalize_text(recognized_text)
               
               if not recognized_text:
                   logger.debug(f"Text filtered out during normalization: '{original_text}'")
                   return ""
               
               # 7. Duplicate detection
               current_time_ms = time.time() * 1000
               if self._duplicate_filter.is_duplicate(recognized_text, current_time_ms):
                   logger.debug(f"Duplicate text filtered: '{recognized_text}'")
                   return ""
               
               # Success - log comprehensive recognition info
               logger.info(
                   f"Whisper recognized: '{recognized_text}' "
                   f"(confidence: {avg_confidence:.3f}, "
                   f"processing_time: {recognition_time:.3f}s, "
                   f"audio_duration: {duration_sec:.3f}s, "
                   f"real_time_factor: {recognition_time/duration_sec:.2f}x)"
               )
               
               return recognized_text
               
           except Exception as e:
               logger.error(f"Whisper recognition error: {e}", exc_info=True)
               return ""

**Advanced Audio Preprocessing Pipeline**:

.. code-block:: python

   def _prepare_audio(self, audio_bytes: bytes) -> np.ndarray:
       """Advanced audio preprocessing for optimal Whisper performance"""
       
       # 1. Convert bytes to numpy array
       audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
       
       # 2. Convert to float32 in range [-1, 1]
       audio_float32 = audio_int16.astype(np.float32) / 32768.0
       
       # 3. Audio quality analysis
       peak_amplitude = np.max(np.abs(audio_float32))
       rms_energy = np.sqrt(np.mean(audio_float32 ** 2))
       dynamic_range = peak_amplitude / (rms_energy + 1e-8)  # Avoid division by zero
       
       # 4. Adaptive normalization based on audio characteristics
       if peak_amplitude > 0.95:  # Clipping detected
           logger.warning(f"Audio clipping detected: peak={peak_amplitude:.3f}")
           # Reduce gain slightly to prevent artifacts
           audio_float32 *= 0.9
       elif peak_amplitude < 0.01:  # Very quiet audio
           logger.debug(f"Very quiet audio detected: peak={peak_amplitude:.3f}")
           # Gentle amplification for very quiet speech
           audio_float32 *= min(2.0, 0.1 / peak_amplitude)
       
       # 5. Optional DC offset removal
       dc_offset = np.mean(audio_float32)
       if abs(dc_offset) > 0.01:  # Significant DC offset
           audio_float32 -= dc_offset
           logger.debug(f"Removed DC offset: {dc_offset:.4f}")
       
       # 6. Quality metrics logging
       logger.debug(
           f"Audio preprocessing: peak={peak_amplitude:.3f}, "
           f"rms={rms_energy:.4f}, dynamic_range={dynamic_range:.1f}, "
           f"length={len(audio_float32)} samples"
       )
       
       return audio_float32

**Dynamic Transcription Parameter Selection**:

.. code-block:: python

   def _get_transcription_options(self, audio_duration: float) -> Dict[str, Any]:
       """Dynamically adjust transcription parameters based on audio characteristics"""
       
       # Base configuration optimized for dictation
       options = {
           "language": "en",
           "beam_size": self._beam_size,  # Default: 5
           "temperature": self._temperature,  # Default: 0.0 for deterministic results
           "no_speech_threshold": self._no_speech_threshold,  # Default: 0.6
           "condition_on_previous_text": False,  # Avoid context bleeding
           "word_timestamps": True,  # Enable word-level timing
           "vad_filter": True,  # Use built-in VAD
           "vad_parameters": {
               'min_silence_duration_ms': 4000,  # Long silence = segment boundary
               'speech_pad_ms': 500  # Padding around speech
           }
       }
       
       # Dynamic adjustments based on audio duration
       if audio_duration < 2.0:
           # Short segments: reduce beam size for speed
           options["beam_size"] = max(1, self._beam_size - 2)
           options["vad_parameters"]["min_silence_duration_ms"] = 2000
           logger.debug("Short audio: reduced beam size and VAD sensitivity")
           
       elif audio_duration < 5.0:
           # Medium segments: slight beam reduction
           options["beam_size"] = max(1, self._beam_size - 1)
           logger.debug("Medium audio: slightly reduced beam size")
           
       elif audio_duration > 15.0:
           # Long segments: increase beam size for accuracy
           options["beam_size"] = min(10, self._beam_size + 2)
           options["vad_parameters"]["min_silence_duration_ms"] = 6000
           logger.debug("Long audio: increased beam size and VAD sensitivity")
       
       # Temperature adjustment for difficult audio
       if hasattr(self, '_recent_failures') and self._recent_failures > 2:
           # Increase temperature for more diverse outputs if recent failures
           options["temperature"] = min(0.2, self._temperature + 0.1)
           logger.debug("Recent failures detected: increased temperature for diversity")
       
       return options

Text Processing
---------------

**Segment Text Extraction**:

.. code-block:: python

   def _extract_text_from_segments(self, segments: List[Any]) -> Tuple[str, float]:
       """Extract text from faster-whisper segments"""
       all_text_parts = []
       total_confidence = 0.0
       segment_count = 0
       
       for segment in segments:
           segment_text = segment.text.strip()
           if not segment_text:
               continue
           
           segment_count += 1
           all_text_parts.append(segment_text)
           
           # Calculate confidence from avg_logprob
           if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
               confidence = min(1.0, max(0.0, (segment.avg_logprob + 1.0) / 1.0))
               total_confidence += confidence
           else:
               total_confidence += 0.8
       
       combined_text = " ".join(all_text_parts).strip()
       avg_confidence = total_confidence / max(1, segment_count)
       
       return combined_text, avg_confidence

**Text Normalization**:

.. code-block:: python

   def _normalize_text(self, text: str) -> str:
       """Advanced text normalization for dictation quality"""
       if not text:
           return ""
       
       text = text.strip()
       if not text:
           return ""
       
       # Remove excessive whitespace
       import re
       text = re.sub(r'\s+', ' ', text)
       
       # Remove common artifacts at boundaries
       text = re.sub(r'^(um|uh|like|so)\s+', '', text, flags=re.IGNORECASE)
       text = re.sub(r'\s+(um|uh|like|so)$', '', text, flags=re.IGNORECASE)
       
       # Handle simple repetitions
       words = text.split()
       if len(words) > 1:
           # Remove consecutive duplicates
           result = [words[0]]
           for word in words[1:]:
               if word.lower() != result[-1].lower():
                   result.append(word)
           text = " ".join(result)
       
       return text.strip()

Duplicate Text Filtering
========================

The ``DuplicateTextFilter`` class provides intelligent duplicate detection across both engines.

Architecture
------------

**Core Components**:

.. code-block:: python

   class DuplicateTextFilter:
       def __init__(self, cache_size: int = 5, duplicate_threshold_ms: float = 300):
           self._text_cache: deque = deque(maxlen=cache_size)
           self._duplicate_threshold_ms = duplicate_threshold_ms
           self._last_recognized_text = ""
           self._last_text_time = 0.0

**Filtering Algorithm**:

.. code-block:: python

   def is_duplicate(self, text: str, current_time_ms: Optional[float] = None) -> bool:
       if not text or not text.strip():
           return True
           
       normalized_text = text.lower().strip()
       
       if current_time_ms is None:
           current_time_ms = time.time() * 1000
       
       # Simple recent duplicate check
       current_time_s = current_time_ms / 1000
       if (normalized_text == self._last_recognized_text.lower() and 
           current_time_s - self._last_text_time < self._duplicate_threshold_ms / 1000):
           return True
       
       # Cache-based duplicate check
       for entry_timestamp, entry_text in reversed(self._text_cache):
           time_diff_ms = current_time_ms - entry_timestamp
           
           # Skip if too old
           if time_diff_ms > self._duplicate_threshold_ms:
               continue
               
           # Exact match
           if entry_text.lower().strip() == normalized_text:
               return True
               
           # High similarity check for longer texts
           if len(normalized_text) > 10 and len(entry_text) > 10:
               text_words = set(normalized_text.split())
               entry_words = set(entry_text.lower().split())
               
               if len(text_words) >= 3 and len(entry_words) >= 3:
                   overlap = text_words.intersection(entry_words)
                   overlap_ratio = len(overlap) / min(len(text_words), len(entry_words))
                   
                   if overlap_ratio > 0.8 and time_diff_ms < self._duplicate_threshold_ms * 0.5:
                       return True
       
       # Update cache and tracking
       self._text_cache.append((current_time_ms, text))
       self._last_recognized_text = text
       self._last_text_time = current_time_s
       
       return False

Smart Timeout Management
========================

The ``SmartTimeoutManager`` provides context-aware timeout optimization for command recognition.

Architecture
------------

**Command Analysis**:

.. code-block:: python

   class SmartTimeoutManager:
       # Optimized timeout constants
       ZERO_TIMEOUT = 0.02      # Instant execution
       INSTANT_TIMEOUT = 0.05   # Near-instant for recognized commands
       QUICK_TIMEOUT = 0.15     # Fast for single words
       DEFAULT_TIMEOUT = 0.4    # Standard timeout
       AMBIGUOUS_TIMEOUT = 0.6  # Longer for ambiguous commands

**Command Categorization**:

.. code-block:: python

   def _categorize_commands(self, commands: List[str]) -> None:
       """Categorize commands for optimized timeout handling"""
       for command in commands:
           words = command.split()
           
           # Instant execution: single short words and common actions
           if len(words) == 1:
               if len(command) <= 4 or command in ["click", "enter", "escape", "space", "tab"]:
                   self._instant_commands.add(command)
               elif command.isdigit() and int(command) <= 20:
                   self._instant_commands.add(command)
               else:
                   self._quick_commands.add(command)
           
           # Quick commands: common multi-key combinations
           elif command in ["ctrl c", "ctrl v", "ctrl z", "ctrl s", "alt tab"]:
               self._quick_commands.add(command)

**Timeout Calculation**:

.. code-block:: python

   def get_timeout_for_text(self, recognized_text: str) -> float:
       """Get optimized timeout based on recognized text"""
       if not recognized_text:
           return self.ZERO_TIMEOUT
           
       text_lower = recognized_text.lower().strip()
       
       # Instant execution for complete, unambiguous commands
       if text_lower in self._instant_commands:
           return self.ZERO_TIMEOUT
       
       # Quick timeout for single-word commands
       if text_lower in self._quick_commands:
           return self.INSTANT_TIMEOUT
       
       # Check for numeric commands
       if text_lower.isdigit():
           return self.ZERO_TIMEOUT if int(text_lower) <= 20 else self.QUICK_TIMEOUT
       
       # Check ambiguous prefixes
       for prefix, group in self._ambiguity_groups.items():
           if text_lower.startswith(prefix):
               # If we have the complete command, execute quickly
               if text_lower in [cmd.lower() for cmd in group.commands]:
                   return self.INSTANT_TIMEOUT
               # Still ambiguous, need more time
               return self.AMBIGUOUS_TIMEOUT
       
       return self.DEFAULT_TIMEOUT

Performance Optimization
========================

The STT system implements numerous performance optimizations:

Processing Efficiency
--------------------

**Mode-Specific Optimization**:
- Command mode: Ultra-fast Vosk processing (<50ms)
- Dictation mode: High-accuracy Whisper processing (200-1000ms)
- Streaming recognition for instant commands
- Adaptive timeout management

**Memory Management**:
- Efficient model loading and caching
- Proper resource cleanup on shutdown
- Thread-safe operation with minimal locking
- Optimized audio buffer handling

**Threading Model**:
- Separate processing threads for each engine
- Thread-safe model access with RLock protection
- Concurrent processing capabilities
- Event-driven coordination

Quality Assurance
-----------------

**Duplicate Prevention**:
- Time-based duplicate filtering
- Similarity analysis for longer texts
- Cache-based recent history tracking
- Mode-specific filtering thresholds

**Text Quality**:
- Advanced normalization and cleaning
- Artifact removal and repetition handling
- Confidence-based result filtering
- Minimum duration validation

**Error Handling**:
- Comprehensive exception handling
- Graceful degradation on failures
- Detailed logging for debugging
- Resource cleanup guarantees

Configuration Management
========================

The STT system is highly configurable through the ``STTConfig`` class:

Engine Configuration
--------------------

.. code-block:: python

   class STTConfig(BaseModel):
       # Engine selection
       default_engine: Literal["vosk", "whisper"] = Field("vosk")
       whisper_model: Literal["tiny", "base", "small", "medium"] = Field("base")
       whisper_device: str = Field("cpu")
       
       # Common configuration
       sample_rate: int = Field(16000)
       max_segment_duration_sec: int = Field(3)
       
       # Mode-specific settings
       command_debounce_interval: float = Field(0.02)      # Ultra-aggressive
       command_duplicate_text_interval: float = Field(0.2)  # Very short
       
       dictation_debounce_interval: float = Field(0.1)     # Reduced
       dictation_duplicate_text_interval: float = Field(4.0) # Extended

Performance Tuning
-------------------

**Command Mode Tuning**:
- Ultra-aggressive debounce (20ms)
- Short duplicate suppression (200ms)
- Fast processing timeouts
- Streaming recognition enabled

**Dictation Mode Tuning**:
- Balanced debounce (100ms)
- Extended duplicate suppression (4s)
- Quality-focused processing
- Context preservation enabled

This comprehensive STT system provides the foundation for all speech recognition in IRIS, delivering both ultra-fast command recognition and high-accuracy dictation through its sophisticated dual-engine architecture with intelligent processing optimization.
