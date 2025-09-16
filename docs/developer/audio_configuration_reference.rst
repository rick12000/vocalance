====================================
Audio Configuration Reference
====================================

This document provides comprehensive reference documentation for all audio system configuration parameters in IRIS, including detailed explanations of their impact on performance and behavior.

Overview
========

The IRIS audio system is highly configurable through Pydantic configuration models that control every aspect of audio processing, from hardware settings to advanced AI parameters. The configuration system is organized into several specialized sections:

- **AudioConfig**: Core audio hardware and processing settings
- **VADConfig**: Voice Activity Detection parameters
- **STTConfig**: Speech-to-Text engine configuration
- **DictationConfig**: Dictation mode settings
- **SoundRecognizerConfig**: Sound recognition ML parameters

All configurations support both default values and runtime customization through YAML configuration files.

AudioConfig
===========

The ``AudioConfig`` class controls fundamental audio capture and processing parameters.

Core Audio Settings
-------------------

**sample_rate**
  - **Type**: ``int``
  - **Default**: ``16000``
  - **Description**: Audio sample rate in Hz
  - **Impact**: Higher rates improve quality but increase processing overhead
  - **Recommended**: 16kHz for optimal STT performance

**channels**
  - **Type**: ``int``
  - **Default**: ``1``
  - **Description**: Number of audio channels
  - **Impact**: Mono (1) is sufficient for voice; stereo (2) increases data volume
  - **Recommended**: 1 for voice applications

**dtype**
  - **Type**: ``str``
  - **Default**: ``"int16"``
  - **Description**: Audio sample data type
  - **Impact**: int16 provides good quality with reasonable memory usage
  - **Options**: "int16", "float32"

**device**
  - **Type**: ``int | None``
  - **Default**: ``None``
  - **Description**: Input device index for sounddevice
  - **Impact**: None uses system default; specific index for device selection
  - **Usage**: Set to specific device ID for non-default microphones

Dual-Mode Processing
--------------------

**enable_dual_mode_processing**
  - **Type**: ``bool``
  - **Default**: ``True``
  - **Description**: Enable separate processing paths for commands vs dictation
  - **Impact**: Allows independent optimization for speed vs accuracy
  - **Recommended**: True for optimal performance

**chunk_size**
  - **Type**: ``int``
  - **Default**: ``320``
  - **Description**: Audio chunk size for dictation mode (samples)
  - **Impact**: Smaller chunks reduce latency but increase overhead
  - **Calculation**: 320 samples = 20ms at 16kHz

**command_chunk_size**
  - **Type**: ``int``
  - **Default**: ``960``
  - **Description**: Ultra-optimized chunk size for command mode (samples)
  - **Impact**: Larger chunks for command mode enable better short-word detection
  - **Calculation**: 960 samples = 60ms at 16kHz
  - **Optimization**: Sized for maximum short-word performance

Example Configuration
---------------------

.. code-block:: yaml

   audio:
     sample_rate: 16000
     chunk_size: 320              # Dictation mode: 20ms chunks
     command_chunk_size: 960      # Command mode: 60ms chunks
     channels: 1
     dtype: "int16"
     device: null                 # Use system default
     enable_dual_mode_processing: true

VADConfig
=========

The ``VADConfig`` class controls Voice Activity Detection parameters for both command and dictation modes.

Base VAD Settings
-----------------

**energy_threshold**
  - **Type**: ``float``
  - **Default**: ``0.006``
  - **Description**: Base energy threshold for speech detection
  - **Impact**: Lower values increase sensitivity but may capture noise
  - **Range**: 0.001-0.01 typical

**silence_timeout**
  - **Type**: ``float``
  - **Default**: ``0.6``
  - **Description**: Base silence timeout in seconds
  - **Impact**: Shorter timeouts increase responsiveness but may cut off speech
  - **Usage**: Fallback value when mode-specific timeouts not used

**max_recording_duration**
  - **Type**: ``float``
  - **Default**: ``4.0``
  - **Description**: Maximum recording duration in seconds
  - **Impact**: Prevents runaway recordings but may truncate long speech
  - **Safety**: Important for memory management

**pre_roll_buffers**
  - **Type**: ``int``
  - **Default**: ``2``
  - **Description**: Number of audio chunks to buffer before speech detection
  - **Impact**: More buffers capture speech onset better but increase latency
  - **Calculation**: Buffer duration = chunks × chunk_duration

Command Mode VAD (Ultra-Optimized)
-----------------------------------

**command_energy_threshold**
  - **Type**: ``float``
  - **Default**: ``0.002``
  - **Description**: Ultra-sensitive threshold for command speech detection
  - **Impact**: Extremely low threshold to catch initial consonants
  - **Optimization**: Tuned for capturing speech onset including soft sounds
  - **Trade-off**: May be sensitive to environmental noise

**command_silence_timeout**
  - **Type**: ``float``
  - **Default**: ``0.35``
  - **Description**: Ultra-fast timeout for command completion
  - **Impact**: Enables rapid command execution but may cut off longer commands
  - **Adaptive**: Works with smart timeout manager for context-aware adjustment
  - **Range**: 0.25-0.6s with adaptive extension

**command_max_recording_duration**
  - **Type**: ``float``
  - **Default**: ``3.0``
  - **Description**: Maximum command recording duration
  - **Impact**: Shorter than dictation to prioritize speed
  - **Usage**: Sufficient for multi-word commands without excessive delay

**command_pre_roll_buffers**
  - **Type**: ``int``
  - **Default**: ``4``
  - **Description**: Pre-roll buffers for command mode
  - **Impact**: 4 × 60ms = 240ms pre-roll to capture speech onset
  - **Critical**: Essential for not missing command beginnings

Dictation Mode VAD (Accuracy-Optimized)
----------------------------------------

**dictation_energy_threshold**
  - **Type**: ``float``
  - **Default**: ``0.0035``
  - **Description**: Balanced threshold for dictation accuracy
  - **Impact**: Higher than command mode to reduce noise while maintaining sensitivity
  - **Optimization**: Balances accuracy with noise rejection

**dictation_silence_timeout**
  - **Type**: ``float``
  - **Default**: ``0.5``
  - **Description**: Timeout optimized for natural speech patterns
  - **Impact**: Allows for natural pauses in speech
  - **Reduced**: From previous 4.0s for faster processing

**dictation_max_recording_duration**
  - **Type**: ``float``
  - **Default**: ``8.0``
  - **Description**: Extended duration for longer dictation segments
  - **Impact**: Accommodates longer sentences and paragraphs
  - **Reduced**: From previous 20s for faster processing

**dictation_pre_roll_buffers**
  - **Type**: ``int``
  - **Default**: ``2``
  - **Description**: Minimal pre-roll for reduced latency
  - **Impact**: 2 × 20ms = 40ms pre-roll for quick response
  - **Trade-off**: Less pre-roll than command mode but faster processing

Advanced Dictation VAD
-----------------------

**dictation_progressive_silence**
  - **Type**: ``bool``
  - **Default**: ``True``
  - **Description**: Enable progressive silence detection
  - **Impact**: Adapts timeout based on speech patterns
  - **Benefit**: Better handling of natural speech rhythms

**dictation_inter_sentence_pause_threshold**
  - **Type**: ``float``
  - **Default**: ``0.35``
  - **Description**: Pause threshold for sentence boundaries
  - **Impact**: Helps detect natural sentence breaks
  - **Usage**: For segmenting continuous speech

**dictation_continuation_energy_threshold**
  - **Type**: ``float``
  - **Default**: ``0.001``
  - **Description**: Ultra-low threshold for detecting speech continuation
  - **Impact**: Catches quiet speech and fast transitions
  - **Critical**: For continuous speech processing

Example VAD Configuration
-------------------------

.. code-block:: yaml

   vad:
     # Command mode - ultra-optimized for speed
     command_energy_threshold: 0.002     # Ultra-sensitive
     command_silence_timeout: 0.35       # Fast timeout
     command_max_recording_duration: 3.0
     command_pre_roll_buffers: 4         # 240ms pre-roll
     
     # Dictation mode - optimized for accuracy
     dictation_energy_threshold: 0.0035  # Balanced sensitivity
     dictation_silence_timeout: 0.5      # Natural speech timeout
     dictation_max_recording_duration: 8.0
     dictation_pre_roll_buffers: 2       # 40ms pre-roll
     
     # Advanced features
     dictation_progressive_silence: true
     dictation_adaptive_timeout_enabled: true

STTConfig
=========

The ``STTConfig`` class controls Speech-to-Text engine selection and processing parameters.

Engine Selection
----------------

**default_engine**
  - **Type**: ``Literal["vosk", "whisper"]``
  - **Default**: ``"vosk"``
  - **Description**: Default STT engine for general use
  - **Impact**: Vosk for speed, Whisper for accuracy
  - **Recommendation**: "vosk" for command-focused applications

**command_engine**
  - **Type**: ``Literal["vosk", "whisper"]``
  - **Default**: ``"vosk"``
  - **Description**: STT engine for command recognition
  - **Impact**: Vosk provides <50ms processing time
  - **Optimization**: Vosk optimized for command vocabulary

**dictation_engine**
  - **Type**: ``Literal["vosk", "whisper"]``
  - **Default**: ``"whisper"``
  - **Description**: STT engine for dictation mode
  - **Impact**: Whisper provides superior accuracy for text
  - **Trade-off**: Higher latency but better text quality

**enable_engine_switching**
  - **Type**: ``bool``
  - **Default**: ``True``
  - **Description**: Enable automatic engine switching based on mode
  - **Impact**: Allows optimal engine per use case
  - **Recommended**: True for best performance

Model Configuration
-------------------

**model_path**
  - **Type**: ``str``
  - **Default**: ``"assets/vosk-model-small-en-us-0.15"``
  - **Description**: Path to Vosk model directory
  - **Impact**: Model size affects accuracy and memory usage
  - **Options**: small (50MB), medium (1.5GB), large (3GB+)

**whisper_model**
  - **Type**: ``Literal["tiny", "base", "small", "medium"]``
  - **Default**: ``"base"``
  - **Description**: Whisper model size selection
  - **Impact**: Larger models = better accuracy + more memory/time
  - **Recommendations**:
    - tiny: 39MB, fastest, lowest accuracy
    - base: 142MB, good balance (recommended)
    - small: 466MB, better accuracy
    - medium: 1.5GB, best accuracy

**whisper_device**
  - **Type**: ``str``
  - **Default**: ``"cpu"``
  - **Description**: Device for Whisper inference
  - **Impact**: CPU for stability, GPU for speed (if available)
  - **Stability**: CPU forced for reliability in current implementation

Processing Parameters
---------------------

**sample_rate**
  - **Type**: ``int``
  - **Default**: ``16000``
  - **Description**: Sample rate for STT processing
  - **Impact**: Must match audio system sample rate
  - **Standard**: 16kHz is optimal for most STT engines

**max_segment_duration_sec**
  - **Type**: ``int``
  - **Default**: ``3``
  - **Description**: Maximum audio segment duration for processing
  - **Impact**: Longer segments may improve accuracy but increase latency
  - **Usage**: Balances processing time with context

Command Mode Processing (Ultra-Low Latency)
--------------------------------------------

**command_debounce_interval**
  - **Type**: ``float``
  - **Default**: ``0.02``
  - **Description**: Ultra-aggressive debounce for command mode (20ms)
  - **Impact**: Prevents duplicate processing but may miss rapid commands
  - **Optimization**: Tuned for real-time response

**command_duplicate_text_interval**
  - **Type**: ``float``
  - **Default**: ``0.2``
  - **Description**: Very short duplicate suppression (200ms)
  - **Impact**: Reduces repeated commands while allowing quick re-execution
  - **Usage**: For responsive command interaction

**command_max_segment_duration_sec**
  - **Type**: ``float``
  - **Default**: ``1.5``
  - **Description**: Short max duration for fast command execution
  - **Impact**: Prioritizes speed over handling very long commands
  - **Trade-off**: May truncate complex commands but ensures responsiveness

Dictation Mode Processing (Accuracy-Focused)
---------------------------------------------

**dictation_debounce_interval**
  - **Type**: ``float``
  - **Default**: ``0.1``
  - **Description**: Reduced debounce for better dictation responsiveness
  - **Impact**: Balances duplicate prevention with continuous speech
  - **Improvement**: Better than previous higher values

**dictation_duplicate_text_interval**
  - **Type**: ``float``
  - **Default**: ``4.0``
  - **Description**: Extended duplicate suppression for phrase repetitions
  - **Impact**: Prevents accidental text duplication during dictation
  - **Usage**: Accounts for natural speech repetitions and corrections

**dictation_max_segment_duration_sec**
  - **Type**: ``float``
  - **Default**: ``20.0``
  - **Description**: Extended duration for longer dictation segments
  - **Impact**: Allows for complete thoughts and paragraphs
  - **Context**: Provides sufficient context for accurate transcription

**dictation_context_enabled**
  - **Type**: ``bool``
  - **Default**: ``True``
  - **Description**: Enable context preservation between segments
  - **Impact**: Improves accuracy by maintaining speech context
  - **Benefit**: Better handling of continuous dictation

**dictation_min_segment_duration_sec**
  - **Type**: ``float``
  - **Default**: ``0.8``
  - **Description**: Minimum segment duration for quality dictation
  - **Impact**: Ensures sufficient context for accurate transcription
  - **Quality**: Filters out very short, potentially noisy segments

Example STT Configuration
-------------------------

.. code-block:: yaml

   stt:
     # Engine selection
     default_engine: "vosk"
     command_engine: "vosk"        # Fast for commands
     dictation_engine: "whisper"  # Accurate for dictation
     enable_engine_switching: true
     
     # Model configuration
     whisper_model: "base"         # Good balance
     whisper_device: "cpu"         # Stable
     
     # Command mode - ultra-low latency
     command_debounce_interval: 0.02      # 20ms
     command_duplicate_text_interval: 0.2 # 200ms
     command_max_segment_duration_sec: 1.5
     
     # Dictation mode - accuracy focused
     dictation_debounce_interval: 0.1     # 100ms
     dictation_duplicate_text_interval: 4.0 # 4s
     dictation_max_segment_duration_sec: 20.0
     dictation_context_enabled: true

DictationConfig
===============

The ``DictationConfig`` class controls dictation mode behavior and text processing.

Trigger Words
-------------

**start_trigger**
  - **Type**: ``str``
  - **Default**: ``"green"``
  - **Description**: Trigger word to start standard dictation
  - **Impact**: Must be easily recognizable and distinct
  - **Usage**: Activates continuous dictation mode

**stop_trigger**
  - **Type**: ``str``
  - **Default**: ``"amber"``
  - **Description**: Trigger word to stop any dictation mode
  - **Impact**: Universal stop command for all dictation modes
  - **Critical**: Must be reliably recognized during dictation

**type_trigger**
  - **Type**: ``str``
  - **Default**: ``"type"``
  - **Description**: Trigger word to start type mode
  - **Impact**: Activates character-by-character typing mode
  - **Usage**: For applications where clipboard paste is problematic

**smart_start_trigger**
  - **Type**: ``str``
  - **Default**: ``"smart green"``
  - **Description**: Trigger phrase to start LLM-assisted dictation
  - **Impact**: Activates AI-enhanced text processing
  - **Feature**: Enables grammar correction and clarity improvement

STT Engine Configuration
------------------------

**enable_stt_switching**
  - **Type**: ``bool``
  - **Default**: ``True``
  - **Description**: Enable automatic STT engine switching for dictation
  - **Impact**: Allows optimal engine selection per mode
  - **Benefit**: Whisper for dictation accuracy, Vosk for commands

**dictation_stt_engine**
  - **Type**: ``str``
  - **Default**: ``"whisper"``
  - **Description**: STT engine for dictation processing
  - **Impact**: Whisper provides superior text accuracy
  - **Recommendation**: Whisper for best dictation quality

**command_stt_engine**
  - **Type**: ``str``
  - **Default**: ``"vosk"``
  - **Description**: STT engine for command recognition during dictation
  - **Impact**: Vosk provides fast amber trigger detection
  - **Usage**: Maintains command responsiveness during dictation

Text Processing
---------------

**min_text_length**
  - **Type**: ``int``
  - **Default**: ``1``
  - **Description**: Minimum length of text to process
  - **Impact**: Filters out very short, potentially erroneous recognition
  - **Usage**: Quality control for text input

**remove_trigger_words**
  - **Type**: ``bool``
  - **Default**: ``True``
  - **Description**: Remove trigger words from dictated text
  - **Impact**: Cleans output text of activation commands
  - **Benefit**: Produces clean dictated text

Text Input Methods
------------------

**use_clipboard**
  - **Type**: ``bool``
  - **Default**: ``True``
  - **Description**: Use clipboard for text input instead of typing
  - **Impact**: Clipboard is faster and more reliable than character-by-character
  - **Benefits**:
    - Instant text insertion
    - Preserves formatting
    - Works with all applications
  - **Fallback**: Character typing available if clipboard fails

**typing_delay**
  - **Type**: ``float``
  - **Default**: ``0.01``
  - **Description**: Delay between keystrokes when typing
  - **Impact**: Slower typing may be more reliable in some applications
  - **Usage**: Only relevant when use_clipboard is False

Example Dictation Configuration
-------------------------------

.. code-block:: yaml

   dictation:
     # Trigger words
     start_trigger: "green"
     stop_trigger: "amber"
     type_trigger: "type"
     smart_start_trigger: "smart green"
     
     # Engine configuration
     enable_stt_switching: true
     dictation_stt_engine: "whisper"  # Accurate
     command_stt_engine: "vosk"       # Fast for triggers
     
     # Text processing
     min_text_length: 1
     remove_trigger_words: true
     
     # Input method
     use_clipboard: true              # Fast and reliable
     typing_delay: 0.01

SoundRecognizerConfig
=====================

The ``SoundRecognizerConfig`` class controls machine learning parameters for custom sound recognition.

Core ML Parameters
------------------

**confidence_threshold**
  - **Type**: ``float``
  - **Default**: ``0.7``
  - **Description**: Minimum confidence for sound recognition
  - **Impact**: Higher values reduce false positives but may miss valid sounds
  - **Range**: 0.5-0.9 typical
  - **Tuning**: Adjust based on training data quality

**k_neighbors**
  - **Type**: ``int``
  - **Default**: ``5``
  - **Description**: Number of neighbors for k-NN classification
  - **Impact**: More neighbors = more stable but potentially less precise
  - **Range**: 3-10 typical
  - **Trade-off**: Stability vs precision

**vote_threshold**
  - **Type**: ``float``
  - **Default**: ``0.6``
  - **Description**: Minimum vote ratio for classification
  - **Impact**: Higher values require stronger consensus among neighbors
  - **Range**: 0.5-0.8 typical
  - **Usage**: Prevents classification on weak evidence

Audio Processing
----------------

**target_sample_rate**
  - **Type**: ``int``
  - **Default**: ``16000``
  - **Description**: Target sample rate for sound processing
  - **Impact**: Must match YAMNet model requirements
  - **Standard**: 16kHz is optimal for YAMNet embeddings

ESC-50 Configuration
--------------------

**esc50_categories**
  - **Type**: ``Dict[str, int]``
  - **Default**: ``{"breathing": 15, "coughing": 15, "keyboard_typing": 15, "mouse_click": 15, "wind": 15, "brushing_teeth": 15, "drinking_sipping": 15}``
  - **Description**: ESC-50 categories and sample counts for negative examples
  - **Impact**: More negative examples improve noise rejection
  - **Usage**: Helps distinguish custom sounds from background noise

**max_esc50_samples_per_category**
  - **Type**: ``int``
  - **Default**: ``15``
  - **Description**: Maximum ESC-50 samples per category
  - **Impact**: Limits memory usage while providing diverse negative examples
  - **Balance**: Sufficient diversity without excessive memory

**max_total_esc50_samples**
  - **Type**: ``int``
  - **Default**: ``100``
  - **Description**: Maximum total ESC-50 samples
  - **Impact**: Overall limit on negative example dataset size
  - **Performance**: Affects training time and memory usage

Example Sound Recognizer Configuration
--------------------------------------

.. code-block:: yaml

   sound_recognizer:
     # ML parameters
     confidence_threshold: 0.7        # Balanced sensitivity
     k_neighbors: 5                   # Good stability
     vote_threshold: 0.6              # Reasonable consensus
     
     # Audio processing
     target_sample_rate: 16000        # YAMNet standard
     
     # ESC-50 negative examples
     max_esc50_samples_per_category: 15
     max_total_esc50_samples: 100
     esc50_categories:
       breathing: 15
       coughing: 15
       keyboard_typing: 15
       mouse_click: 15
       wind: 15
       brushing_teeth: 15
       drinking_sipping: 15

Performance Tuning Guidelines
=============================

The audio system configuration can be tuned for different performance profiles:

Ultra-Low Latency (Command-Focused)
------------------------------------

.. code-block:: yaml

   # Optimize for absolute minimum latency
   audio:
     command_chunk_size: 480          # 30ms chunks
   
   vad:
     command_energy_threshold: 0.001  # Ultra-sensitive
     command_silence_timeout: 0.25    # Very fast timeout
     command_pre_roll_buffers: 3      # Minimal pre-roll
   
   stt:
     command_debounce_interval: 0.01  # 10ms debounce
     command_duplicate_text_interval: 0.1 # 100ms duplicate filter

High Accuracy (Dictation-Focused)
----------------------------------

.. code-block:: yaml

   # Optimize for maximum accuracy
   vad:
     dictation_energy_threshold: 0.005    # Less sensitive
     dictation_silence_timeout: 0.8       # Longer timeout
     dictation_max_recording_duration: 15.0 # Extended duration
   
   stt:
     whisper_model: "small"               # Better model
     dictation_max_segment_duration_sec: 30.0 # Longer context

Balanced Performance (Recommended)
-----------------------------------

.. code-block:: yaml

   # Default configuration provides good balance
   # of speed and accuracy for most use cases
   audio:
     enable_dual_mode_processing: true
   
   vad:
     command_energy_threshold: 0.002
     dictation_energy_threshold: 0.0035
   
   stt:
     enable_engine_switching: true
     whisper_model: "base"

Environment-Specific Tuning
===========================

Noisy Environment
-----------------

.. code-block:: yaml

   # Adjust for background noise
   vad:
     command_energy_threshold: 0.004     # Less sensitive
     dictation_energy_threshold: 0.006   # Less sensitive
   
   sound_recognizer:
     confidence_threshold: 0.8           # Higher confidence required

Quiet Environment
-----------------

.. code-block:: yaml

   # Optimize for quiet environment
   vad:
     command_energy_threshold: 0.001     # More sensitive
     dictation_energy_threshold: 0.002   # More sensitive
   
   sound_recognizer:
     confidence_threshold: 0.6           # Lower confidence acceptable

Configuration File Example
==========================

Complete example configuration file:

.. code-block:: yaml

   app:
     audio:
       sample_rate: 16000
       chunk_size: 320
       command_chunk_size: 960
       channels: 1
       dtype: "int16"
       device: null
       enable_dual_mode_processing: true
     
     vad:
       # Command mode optimization
       command_energy_threshold: 0.002
       command_silence_timeout: 0.35
       command_max_recording_duration: 3.0
       command_pre_roll_buffers: 4
       
       # Dictation mode optimization
       dictation_energy_threshold: 0.0035
       dictation_silence_timeout: 0.5
       dictation_max_recording_duration: 8.0
       dictation_pre_roll_buffers: 2
       dictation_progressive_silence: true
       dictation_adaptive_timeout_enabled: true
     
     stt:
       default_engine: "vosk"
       command_engine: "vosk"
       dictation_engine: "whisper"
       enable_engine_switching: true
       whisper_model: "base"
       whisper_device: "cpu"
       
       # Command mode processing
       command_debounce_interval: 0.02
       command_duplicate_text_interval: 0.2
       command_max_segment_duration_sec: 1.5
       
       # Dictation mode processing
       dictation_debounce_interval: 0.1
       dictation_duplicate_text_interval: 4.0
       dictation_max_segment_duration_sec: 20.0
       dictation_context_enabled: true
     
     dictation:
       start_trigger: "green"
       stop_trigger: "amber"
       type_trigger: "type"
       smart_start_trigger: "smart green"
       enable_stt_switching: true
       dictation_stt_engine: "whisper"
       command_stt_engine: "vosk"
       min_text_length: 1
       remove_trigger_words: true
       use_clipboard: true
       typing_delay: 0.01
     
     sound_recognizer:
       confidence_threshold: 0.7
       k_neighbors: 5
       vote_threshold: 0.6
       target_sample_rate: 16000
       max_esc50_samples_per_category: 15
       max_total_esc50_samples: 100

This comprehensive configuration reference provides all the parameters needed to tune the IRIS audio system for optimal performance in any environment or use case.
