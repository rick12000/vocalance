=============================
Audio Recording System
=============================

This document provides detailed technical documentation of the IRIS audio recording system, covering the ``SimpleAudioService`` and ``AudioRecorder`` components that handle microphone input and voice activity detection.

Overview
========

The audio recording system implements a dual-recorder architecture optimized for both ultra-low latency command recognition and high-accuracy dictation processing. The system uses independent recorders that can operate simultaneously, with sophisticated Voice Activity Detection (VAD) and adaptive timeout management.

SimpleAudioService
==================

The ``SimpleAudioService`` is the main coordinator for audio recording operations, managing two independent ``AudioRecorder`` instances.

Architecture
------------

**Dual Recorder Design**:

- **Command Recorder**: Optimized for speed with streaming support
- **Dictation Recorder**: Optimized for accuracy with extended capture

**Key Features**:

- Independent recorder lifecycles
- Mode-based activation control
- Event-driven audio segment publication
- Streaming recognition integration
- Thread-safe operation

Initialization
--------------

The service initializes both recorders with mode-specific configurations:

.. code-block:: python

   # Command recorder - optimized for speed
   self._command_recorder = AudioRecorder(
       app_config=self._config,
       mode="command",
       on_audio_segment=self._on_command_audio_segment,
       on_streaming_chunk=self._on_command_streaming_chunk
   )
   
   # Dictation recorder - optimized for accuracy
   self._dictation_recorder = AudioRecorder(
       app_config=self._config,
       mode="dictation", 
       on_audio_segment=self._on_dictation_audio_segment
   )

Mode Management
---------------

The service manages recorder activation based on system mode:

**Command Mode** (Default):
- Command recorder: ACTIVE
- Dictation recorder: INACTIVE

**Dictation Mode**:
- Command recorder: ACTIVE (for amber detection)
- Dictation recorder: ACTIVE (for text capture)

**Mode Switching Events**:

The service listens for ``AudioModeChangeRequestEvent`` to coordinate mode changes:

.. code-block:: python

   async def _handle_audio_mode_change_request(self, event: AudioModeChangeRequestEvent):
       if event.mode == "dictation":
           self._command_recorder.set_active(True)   # Keep for amber detection
           self._dictation_recorder.set_active(True) # Activate for dictation
       elif event.mode == "command":
           self._command_recorder.set_active(True)   # Normal command processing
           self._dictation_recorder.set_active(False) # Deactivate dictation

AudioRecorder
=============

The ``AudioRecorder`` class implements sophisticated audio capture with Voice Activity Detection (VAD) and adaptive timeout management.

Core Architecture
-----------------

**Threading Model**:
- Main recording thread for continuous audio capture
- Thread-safe state management with locks
- Graceful shutdown with proper cleanup

**Audio Processing Pipeline**:
1. Continuous audio capture from microphone
2. Real-time energy calculation and VAD
3. Pre-roll buffer management
4. Speech detection and segment collection
5. Silence detection and timeout management
6. Audio segment publication

Mode-Specific Configuration
---------------------------

The recorder adapts its behavior based on the configured mode:

Command Mode Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optimized for Ultra-Low Latency**:

.. code-block:: python

   # Ultra-optimized chunk size for maximum short-word performance
   self.chunk_size = app_config.audio.command_chunk_size  # 960 samples (60ms at 16kHz)
   
   # Ultra-sensitive energy threshold for speech onset detection
   self.energy_threshold = app_config.vad.command_energy_threshold  # 0.002
   
   # Fast silence timeout with adaptive extension
   self.silence_timeout = app_config.vad.command_silence_timeout  # 0.35s
   
   # Optimized duration for multi-word commands
   self.max_duration = app_config.vad.command_max_recording_duration  # 3.0s
   
   # Sufficient pre-roll buffers to capture speech onset (240ms)
   self.pre_roll_chunks = app_config.vad.command_pre_roll_buffers  # 4 chunks
   
   # Enable streaming recognition for instant commands
   self.enable_streaming = True

Dictation Mode Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optimized for Accuracy and Natural Speech**:

.. code-block:: python

   # Standard chunk size for balanced processing
   self.chunk_size = app_config.audio.chunk_size  # 320 samples (20ms at 16kHz)
   
   # Balanced energy threshold for accuracy
   self.energy_threshold = app_config.vad.dictation_energy_threshold  # 0.0035
   
   # Extended silence timeout for natural speech patterns
   self.silence_timeout = app_config.vad.dictation_silence_timeout  # 0.5s
   
   # Extended duration for longer dictation segments
   self.max_duration = app_config.vad.dictation_max_recording_duration  # 8.0s
   
   # Minimal pre-roll buffers for reduced latency
   self.pre_roll_chunks = app_config.vad.dictation_pre_roll_buffers  # 2 chunks
   
   # No streaming for dictation (accuracy focus)
   self.enable_streaming = False

Voice Activity Detection (VAD)
==============================

The VAD system implements sophisticated speech detection with adaptive thresholds and noise floor estimation.

Energy Calculation
------------------

**RMS Energy Calculation**:

.. code-block:: python

   def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
       """Calculate RMS energy for VAD with format handling"""
       if audio_chunk.dtype == np.int16:
           # Convert int16 [-32768, 32767] to float32 [-1, 1]
           normalized = audio_chunk.astype(np.float32) / 32768.0
           return np.sqrt(np.mean(normalized ** 2))
       elif audio_chunk.dtype == np.float32:
           # Already normalized, calculate RMS directly
           return np.sqrt(np.mean(audio_chunk ** 2))
       else:
           # Handle other formats by converting to float32
           normalized = audio_chunk.astype(np.float32)
           # Assume range is [0, 1] or [-1, 1] based on min/max
           if np.max(normalized) > 1.0:
               normalized = normalized / np.max(np.abs(normalized))
           return np.sqrt(np.mean(normalized ** 2))

**Energy Calculation Deep Dive:**

The RMS (Root Mean Square) energy calculation is critical for reliable speech detection:

.. code-block:: python

   # Mathematical breakdown of RMS energy calculation:
   # 1. Square each sample: x[n]²
   # 2. Calculate mean: (1/N) * Σ(x[n]²)
   # 3. Take square root: √(mean)
   
   # For a 960-sample chunk (60ms at 16kHz):
   chunk_samples = 960
   sample_rate = 16000
   duration_ms = (chunk_samples / sample_rate) * 1000  # 60ms
   
   # Energy range interpretation:
   # 0.001-0.002: Very quiet (background noise)
   # 0.002-0.005: Soft speech (whispers, consonants)
   # 0.005-0.020: Normal speech
   # 0.020-0.100: Loud speech
   # >0.100: Very loud (shouting, music)

**Key Characteristics**:
- Normalized to [-1, 1] range for int16 audio
- RMS provides better speech detection than peak amplitude
- Handles different audio formats consistently
- Computationally efficient for real-time processing

Adaptive Threshold Management
-----------------------------

**Noise Floor Estimation with Environmental Adaptation**:

The system continuously estimates the noise floor and adapts thresholds to environmental conditions:

.. code-block:: python

   def _update_noise_floor(self, energy: float):
       """Advanced noise floor estimation with environmental adaptation"""
       # Only collect noise samples when energy is below current threshold
       if energy <= self.energy_threshold and len(self._noise_samples) < self._max_noise_samples:
           self._noise_samples.append(energy)
           
           # Once we have enough samples, calculate adaptive thresholds
           if len(self._noise_samples) == self._max_noise_samples:  # 20 samples
               # Use 75th percentile to avoid outliers but capture typical noise
               self._noise_floor = np.percentile(self._noise_samples, 75)
               
               # Mode-specific margin multipliers for optimal performance
               if self.mode == "command":
                   margin_multiplier = 3.0    # Aggressive for ultra-sensitivity
                   max_threshold_multiplier = 2.0  # Allow 2x increase maximum
               else:  # dictation mode
                   margin_multiplier = 2.5    # Conservative for accuracy
                   max_threshold_multiplier = 1.5  # Limit adaptation
               
               # Calculate adaptive threshold
               adaptive_threshold = self._noise_floor * margin_multiplier
               
               # Safety check: don't increase threshold too dramatically
               max_allowed_threshold = self.energy_threshold * max_threshold_multiplier
               if adaptive_threshold > max_allowed_threshold:
                   adaptive_threshold = max_allowed_threshold
               
               # Update thresholds if environment is significantly noisier
               if adaptive_threshold > self.energy_threshold:
                   old_threshold = self.energy_threshold
                   self.energy_threshold = adaptive_threshold
                   self.silence_threshold = self.energy_threshold * 0.35  # 35% ratio
                   
                   self.logger.info(
                       f"Adapted thresholds for {self.mode} mode: "
                       f"{old_threshold:.6f} -> {self.energy_threshold:.6f} "
                       f"(noise floor: {self._noise_floor:.6f})"
                   )

**Threshold Hierarchy and Relationships**:

.. code-block:: python

   # Threshold hierarchy (from most to least sensitive):
   THRESHOLD_HIERARCHY = {
       'noise_floor': 'Estimated background noise level',
       'silence_threshold': 'energy_threshold * 0.35',  # 35% of speech threshold
       'energy_threshold': 'Primary speech detection threshold',  
       'continuation_threshold': 'energy_threshold * 0.5',  # 50% for speech continuation
   }
   
   # Example threshold values in different environments:
   ENVIRONMENT_EXAMPLES = {
       'quiet_room': {
           'noise_floor': 0.0008,
           'silence_threshold': 0.0007,  # Below noise floor for clean detection
           'energy_threshold': 0.002,    # Default command mode
           'adaptation': 'No adaptation needed'
       },
       'office_environment': {
           'noise_floor': 0.0025,
           'silence_threshold': 0.0026,  # Slightly above noise floor
           'energy_threshold': 0.0075,   # 3x noise floor
           'adaptation': 'Moderate adaptation'
       },
       'noisy_environment': {
           'noise_floor': 0.006,
           'silence_threshold': 0.0042,  # 35% of adapted threshold
           'energy_threshold': 0.012,    # 2x original threshold (safety limit)
           'adaptation': 'Maximum adaptation applied'
       }
   }

**Environmental Adaptation Algorithm**:

.. code-block:: python

   def _analyze_environment(self) -> dict:
       """Analyze current acoustic environment"""
       if len(self._noise_samples) < self._max_noise_samples:
           return {'status': 'collecting_samples', 'samples': len(self._noise_samples)}
       
       # Statistical analysis of noise floor
       noise_std = np.std(self._noise_samples)
       noise_mean = np.mean(self._noise_samples)
       noise_percentiles = {
           '25th': np.percentile(self._noise_samples, 25),
           '50th': np.percentile(self._noise_samples, 50),
           '75th': np.percentile(self._noise_samples, 75),
           '95th': np.percentile(self._noise_samples, 95)
       }
       
       # Classify environment based on noise characteristics
       if noise_std < 0.0005:
           environment_type = "stable_quiet"
       elif noise_std < 0.002:
           environment_type = "stable_moderate"
       else:
           environment_type = "variable_noisy"
       
       return {
           'environment_type': environment_type,
           'noise_floor': self._noise_floor,
           'noise_std': noise_std,
           'noise_mean': noise_mean,
           'percentiles': noise_percentiles,
           'current_threshold': self.energy_threshold,
           'adaptation_factor': self.energy_threshold / self._original_threshold
       }

**Threshold Update Strategy**:
- Collects 20 noise samples during quiet periods
- Uses 75th percentile to avoid outliers while capturing typical noise
- Mode-specific margin multipliers (3.0x for command, 2.5x for dictation)
- Safety limits prevent excessive threshold increases
- Maintains 35% ratio between energy and silence thresholds
- Logs threshold changes for debugging and monitoring

Speech Detection Algorithm
--------------------------

**Pre-roll Buffer Management**:

The system maintains a circular buffer of recent audio chunks to capture speech onset:

.. code-block:: python

   # Maintain pre-roll buffer
   pre_roll_buffer.append(data)
   if len(pre_roll_buffer) > self.pre_roll_chunks:
       pre_roll_buffer.pop(0)
   
   # Speech detection
   if energy > self.energy_threshold:
       speech_detected = True
       audio_buffer.extend(pre_roll_buffer)  # Include pre-roll in segment

**Speech Segment Collection**:

Once speech is detected, the system collects audio until silence is detected:

.. code-block:: python

   while self._is_recording and self._is_active:
       data, _ = self._stream.read(self.chunk_size)
       energy = self._calculate_energy(data)
       audio_buffer.append(data)
       
       # Streaming recognition for command mode
       if (self.enable_streaming and self.on_streaming_chunk and 
           chunks_collected >= 3 and chunks_collected % 2 == 0):
           current_audio = np.concatenate(audio_buffer)
           recognized_command = self.on_streaming_chunk(current_audio.tobytes(), False)
           
           if recognized_command and self._is_instant_command(recognized_command):
               break  # Early termination for instant commands
       
       # Silence detection with adaptive timeout
       current_timeout = self._get_timeout(chunks_collected)
       if energy < self.silence_threshold:
           if silence_start is None:
               silence_start = time.time()
           elif time.time() - silence_start > current_timeout:
               break  # End of speech detected
       else:
           silence_start = None  # Reset silence timer

Adaptive Timeout Management
===========================

The system implements intelligent timeout management that adapts based on speech characteristics and context.

Smart Timeout Algorithm
-----------------------

**Command Mode Adaptive Timeouts**:

.. code-block:: python

   def _get_timeout(self, chunks_collected: int) -> float:
       """Get adaptive timeout based on speech length"""
       if self.mode != "command":
           return self.silence_timeout
           
       # Progressive timeout based on speech duration
       if chunks_collected <= 4:      # < 240ms - very short
           return 0.25                # Quick timeout for single words
       elif chunks_collected <= 8:    # < 480ms - short phrase
           return self.silence_timeout # Standard timeout
       else:                          # > 480ms - longer command
           return 0.6                 # Extended timeout

**Smart Timeout Manager Integration**:

For command mode, the system integrates with ``SmartTimeoutManager`` for context-aware timeouts:

.. code-block:: python

   def _is_instant_command(self, recognized_text: str) -> bool:
       """Check if command should execute instantly"""
       if not self._smart_timeout_manager or not recognized_text:
           return False
       timeout = self._smart_timeout_manager.get_timeout_for_text(recognized_text)
       return timeout <= 0.05  # Execute immediately for known complete commands

Streaming Recognition Integration
=================================

Command mode supports streaming recognition for ultra-low latency response.

Streaming Architecture
----------------------

**Real-time Processing**:
- Audio chunks processed every 2nd chunk after initial 3 chunks
- Direct STT engine integration bypassing event bus for speed
- Early termination for recognized instant commands

**Implementation**:

.. code-block:: python

   # Streaming recognition for command mode
   if (self.enable_streaming and self.on_streaming_chunk and 
       chunks_collected >= 3 and chunks_collected % 2 == 0):
       current_audio = np.concatenate(audio_buffer)
       recognized_command = self.on_streaming_chunk(current_audio.tobytes(), False)
       
       if recognized_command and self._is_instant_command(recognized_command):
           self.logger.info(f"Instant command '{recognized_command}' detected")
           break  # Stop recording immediately

**Benefits**:
- Sub-200ms response time for common commands
- Reduced audio processing overhead
- Improved user experience for frequent operations

Audio Stream Management
=======================

The system implements robust audio stream management with proper cleanup and error handling.

Stream Lifecycle
-----------------

**Initialization**:

.. code-block:: python

   self._stream = sd.InputStream(
       samplerate=self.sample_rate,    # 16kHz
       blocksize=self.chunk_size,      # Mode-specific chunk size
       channels=1,                     # Mono audio
       dtype='int16',                  # 16-bit PCM
       device=self.device              # Configurable audio device
   )
   self._stream.start()

**Cleanup**:

.. code-block:: python

   def _cleanup_stream(self):
       """Properly cleanup audio stream with detailed error handling"""
       if self._stream:
           try:
               if hasattr(self._stream, 'active') and self._stream.active:
                   self._stream.stop()
               self._stream.close()
           except Exception as e:
               self.logger.error(f"Error cleaning up {self.mode} audio stream: {e}")
           finally:
               self._stream = None

Error Handling
--------------

**Robust Error Recovery**:
- Stream failure detection and recovery
- Device disconnection handling
- Graceful degradation on errors
- Comprehensive logging for debugging

**Thread Safety**:
- Lock-protected state management
- Safe shutdown procedures
- Resource cleanup guarantees

Audio Segment Publication
=========================

Processed audio segments are published as events for downstream processing.

Event Publication
-----------------

**Command Audio Segments**:

.. code-block:: python

   def _on_command_audio_segment(self, segment_bytes: bytes):
       """Handle command mode audio segments - optimized for speed"""
       event = CommandAudioSegmentReadyEvent(
           audio_bytes=segment_bytes,
           sample_rate=self._config.audio.sample_rate
       )
       self._publish_audio_event(event)

**Dictation Audio Segments**:

.. code-block:: python

   def _on_dictation_audio_segment(self, segment_bytes: bytes):
       """Handle dictation mode audio segments - optimized for accuracy"""
       event = DictationAudioSegmentReadyEvent(
           audio_bytes=segment_bytes,
           sample_rate=self._config.audio.sample_rate
       )
       self._publish_audio_event(event)

**Thread-Safe Publication**:

.. code-block:: python

   def _publish_audio_event(self, event_data: BaseEvent):
       """Unified event publication method"""
       if self._main_event_loop and not self._main_event_loop.is_closed():
           asyncio.run_coroutine_threadsafe(
               self._event_bus.publish(event_data),
               self._main_event_loop
           )

Quality Control
===============

The system implements audio quality control to ensure reliable processing.

Segment Filtering
-----------------

**Minimum Duration Filter**:

.. code-block:: python

   # Process collected audio
   if audio_buffer and self.on_audio_segment:
       audio_data = np.concatenate(audio_buffer)
       duration = len(audio_data) / self.sample_rate
       
       if duration >= 0.1:  # Minimum 100ms duration filter
           audio_bytes = audio_data.tobytes()
           self.logger.info(f"Segment captured: {duration:.3f}s")
           self.on_audio_segment(audio_bytes)

**Quality Metrics**:
- Duration validation (minimum 100ms)
- Energy level validation
- Audio format consistency
- Proper segment boundaries

Performance Characteristics
===========================

The recording system is optimized for different performance profiles:

Command Mode Performance
------------------------

- **Latency**: <100ms from speech to recognition start
- **Chunk Size**: 60ms (960 samples at 16kHz)
- **Pre-roll**: 240ms (4 chunks)
- **Timeout**: Adaptive 0.25-0.6s
- **Streaming**: Enabled for instant commands

Dictation Mode Performance
--------------------------

- **Latency**: ~200ms from speech to recognition start
- **Chunk Size**: 20ms (320 samples at 16kHz)
- **Pre-roll**: 40ms (2 chunks)
- **Timeout**: 0.5s for natural speech
- **Accuracy**: Optimized for extended segments

Memory Usage
------------

- **Buffer Management**: Efficient circular buffers
- **Memory Footprint**: <10MB per recorder
- **Cleanup**: Automatic resource management
- **Scaling**: Linear with audio duration

This recording system provides the foundation for all audio processing in IRIS, delivering both ultra-low latency command recognition and high-accuracy dictation capabilities through its sophisticated dual-recorder architecture.
