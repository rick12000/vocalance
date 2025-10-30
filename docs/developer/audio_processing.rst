Audio Processing Pipeline
==========================

Vocalance transforms raw microphone input into actionable commands through a sophisticated multi-stage audio processing pipeline. Audio flows from continuous capture through voice activity detection, dual-mode recording coordination, multi-engine speech recognition, non-verbal sound classification, and predictive command execution. Each stage is optimized for minimal latency while maintaining high accuracy, enabling natural voice-driven computer control.

.. mermaid::

   flowchart TD
       A[Microphone Input] --> B[Voice Activity Detection]
       B --> C{Speech Detected?}
       C -->|Yes| D[Dual-Recorder System]
       C -->|No| B

       D --> E{Current Mode?}
       E -->|Command| F[Command Recorder<br/>60ms chunks, 180ms timeout]
       E -->|Dictation| G[Dictation Recorder<br/>20ms chunks, 800ms timeout]
       E -->|Dictation| F

       F --> H[Audio Segment Ready]
       G --> I[Audio Segment Ready]

       H --> J{Mode Check}
       J -->|Command Mode| K[Vosk STT<br/>Fast Recognition]
       J -->|Dictation Mode| L[Stop Word Detection Only]

       I --> M[Whisper STT<br/>High Accuracy]

       K --> N{Text Result?}
       N -->|Empty| O[Sound Recognition<br/>YAMNet + KNN]
       N -->|Text| P[Command Parser]

       L --> Q{Stop Word?}
       Q -->|Yes| P
       Q -->|No| R[Ignore]

       M --> P
       O --> S{Custom Sound?}
       S -->|Yes| T[Mapped Command]
       T --> P

       P --> U[Command Execution]

       style B fill:#e1f5ff
       style D fill:#fff4e1
       style K fill:#e8f5e9
       style M fill:#e8f5e9
       style O fill:#f3e5f5
       style P fill:#ffe0e0

This overview shows how audio flows through the system. The following sections detail each stage, starting with the foundational voice activity detection engine that determines when speech occurs.

Voice Activity Detection Engine
--------------------------------

At the heart of audio capture sits the ``AudioRecorder`` class, implementing energy-based Voice Activity Detection (VAD). VAD continuously monitors microphone input to distinguish speech from silence, triggering recording only when someone speaks. This prevents wasted processing on ambient noise while ensuring complete utterances are captured.

The High-Level Flow
~~~~~~~~~~~~~~~~~~~

.. mermaid::

   flowchart LR
       A[Microphone Stream] --> B[Calculate RMS Energy]
       B --> C{Energy ><br/>Threshold?}
       C -->|Below| D[Buffer Pre-roll<br/>Update Noise Floor]
       C -->|Above| E[Start Recording]
       D --> B
       E --> F[Collect Audio]
       F --> G{Silence Detected<br/>or Max Duration?}
       G -->|No| F
       G -->|Yes| H[Publish Audio Segment]
       H --> B

The VAD continuously measures audio energy (volume) in small chunks. When energy exceeds a threshold, recording begins. When consecutive silent chunks are detected or maximum duration is reached, the complete audio segment is published for speech recognition.

Configuration: Command vs Dictation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recorder adapts its behavior based on mode, trading off latency for accuracy:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Parameter
     - Command Mode
     - Dictation Mode
   * - Chunk Size
     - 960 samples (60ms)
     - 320 samples (20ms)
   * - Energy Threshold
     - 0.002 RMS
     - 0.0035 RMS
   * - Silent Chunks to End
     - 3 chunks (180ms)
     - 40 chunks (800ms)
   * - Max Duration
     - 4 seconds
     - 30 seconds
   * - Pre-roll Buffers
     - 4 chunks (240ms)
     - 8 chunks (160ms)
   * - Min Duration
     - 50ms
     - 100ms

**Command mode** prioritizes speed: larger chunks (60ms) for faster processing, aggressive silence timeout (180ms) to avoid delays, and shorter maximum duration for rapid-fire commands.

**Dictation mode** prioritizes completeness: smaller chunks (20ms) for finer granularity, long silence timeout (800ms) to accommodate natural pauses, and extended duration to capture full sentences.

The VAD State Machine
~~~~~~~~~~~~~~~~~~~~~~

Internally, VAD operates as a two-state machine with adaptive noise handling:

.. mermaid::

   stateDiagram-v2
       [*] --> Waiting: recorder starts
       Waiting --> Recording: energy > energy_threshold
       Recording --> Waiting: silent_chunks >= threshold
       Recording --> Waiting: duration >= max_duration
       Waiting --> Waiting: buffer pre-roll chunks<br/>update noise floor
       Recording --> Recording: collect audio chunks<br/>track consecutive silence

       note right of Waiting
           • Calculate RMS energy per chunk
           • Maintain circular pre-roll buffer
           • Collect noise samples (< threshold)
           • Adapt thresholds if noisy environment
       end note

       note right of Recording
           • Prepend pre-roll buffer (capture speech onset)
           • Continue collecting chunks
           • Count consecutive silent chunks
           • Publish complete segment on exit
       end note

**Waiting State**: The recorder continuously reads audio chunks, calculating their RMS (Root Mean Square) energy. Chunks are stored in a circular pre-roll buffer. Low-energy chunks update the noise floor estimate. When energy exceeds the threshold, transition to Recording.

**Recording State**: The pre-roll buffer is prepended to the recording (solving the "first syllable problem"—by the time VAD detects speech, the beginning has passed). Audio continues to be collected. Silent chunks are counted. When consecutive silence reaches the configured threshold, or maximum duration is hit, the complete audio segment is published via callback.

The Pre-Roll Mechanism
~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
       participant Mic as Microphone
       participant VAD as VAD Engine
       participant Buffer as Pre-roll Buffer<br/>(circular, size=4)
       participant Recording as Recording Buffer

       Note over Mic,Buffer: User silent, VAD in Waiting state

       Mic->>VAD: chunk 1 (low energy)
       VAD->>Buffer: store chunk 1
       Mic->>VAD: chunk 2 (low energy)
       VAD->>Buffer: store chunk 2
       Mic->>VAD: chunk 3 (low energy)
       VAD->>Buffer: store chunk 3
       Mic->>VAD: chunk 4 (low energy)
       VAD->>Buffer: store chunk 4

       Note over Mic,Recording: User starts speaking "Hello"

       Mic->>VAD: chunk 5 ("He-", HIGH energy)
       VAD->>VAD: energy > threshold<br/>SPEECH DETECTED!
       VAD->>Recording: prepend chunks 2,3,4,5
       Note over Recording: Captures full "Hello"<br/>including onset

       Mic->>VAD: chunk 6 ("-llo")
       VAD->>Recording: append chunk 6

Without pre-roll, the first syllable "He-" would be partially lost. With pre-roll, the complete word is captured.

Adaptive Noise Floor
~~~~~~~~~~~~~~~~~~~~~

To handle varying acoustic environments (quiet room vs noisy office), VAD adapts its thresholds:

.. mermaid::

   flowchart TD
       A[Chunk Energy < Threshold] --> B{Collected < 20<br/>Noise Samples?}
       B -->|Yes| C[Store Energy as Noise Sample]
       B -->|No| D[Skip]
       C --> E{Collected Exactly<br/>20 Samples?}
       E -->|Yes| F[Calculate 75th Percentile<br/>= Noise Floor]
       E -->|No| D
       F --> G[Adaptive Threshold =<br/>Noise Floor × Margin Multiplier]
       G --> H{Adaptive > Current × 2?}
       H -->|Yes| I[Update Energy Threshold<br/>Update Silence Threshold]
       H -->|No| J[Keep Original Thresholds]

When the environment is consistently noisy (e.g., fan noise, keyboard typing), the first 20 low-energy chunks establish a noise floor. If the adaptive threshold is significantly higher than the default, thresholds are raised to prevent false triggers. This happens automatically per recorder instance, transparently to the user.

From VAD to Service Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ``AudioRecorder`` runs in its own background thread, invoking callbacks when audio segments are complete. The ``AudioService`` orchestrates two recorder instances, routing their output appropriately. This dual-recorder design enables seamless mode switching, which we'll explore next.

Dual-Recorder Coordination
---------------------------

Vocalance doesn't start/stop recorders when switching between command and dictation modes—instead, it runs **two recorders simultaneously** and activates them selectively. This eliminates thread startup latency and audio device reinitialization, enabling instant mode transitions.

Architecture Overview
~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   flowchart TD
       subgraph AudioService
           direction TB
           A[Command Recorder Thread<br/>Mode: command<br/>60ms chunks]
           B[Dictation Recorder Thread<br/>Mode: dictation<br/>20ms chunks]
       end

       C[Microphone] --> A
       C --> B

       A -->|set_active True/False| D{Active?}
       B -->|set_active True/False| E{Active?}

       D -->|True| F[VAD Processing<br/>Publish CommandAudioSegmentReady]
       D -->|False| G[Thread Sleeps]

       E -->|True| H[VAD Processing<br/>Publish DictationAudioSegmentReady]
       E -->|False| I[Thread Sleeps]

       F --> J[Event Bus]
       H --> J

       style A fill:#bbdefb
       style B fill:#c8e6c9

Both recorders continuously read from the microphone, but only **active** recorders perform VAD and publish events. Inactive recorders sleep, consuming minimal CPU.

Mode Switching Logic
~~~~~~~~~~~~~~~~~~~~

The ``AudioService`` responds to ``AudioModeChangeRequestEvent``, adjusting recorder activation:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Mode
     - Command Recorder
     - Dictation Recorder
   * - **Command** (default)
     - **ACTIVE** (captures commands)
     - INACTIVE (sleeping)
   * - **Dictation** (during dictation)
     - **ACTIVE** (captures stop words only)
     - **ACTIVE** (captures dictation text)

**Why keep command recorder active during dictation?** To detect stop trigger words like "stop dictation" while the user is dictating. The command recorder remains ready to interrupt dictation, while the dictation recorder captures the full text.

Complete Mode Switch Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This sequence diagram shows the full lifecycle of entering and exiting dictation mode:

.. mermaid::

   sequenceDiagram
       participant User
       participant CmdRec as Command Recorder
       participant DictRec as Dictation Recorder
       participant EventBus
       participant STT as STT Service
       participant Parser as Command Parser

       Note over CmdRec,DictRec: Initial: Command Mode
       Note over CmdRec: VAD Active
       Note over DictRec: VAD Inactive

       User->>CmdRec: Says "dictate"
       CmdRec->>EventBus: CommandAudioSegmentReady
       EventBus->>STT: Process audio
       STT->>EventBus: CommandTextRecognized("dictate")
       EventBus->>Parser: Parse command
       Parser->>EventBus: DictationStartCommand
       EventBus->>CmdRec: AudioModeChangeRequest(mode="dictation")
       EventBus->>DictRec: AudioModeChangeRequest(mode="dictation")

       Note over CmdRec,DictRec: Now: Dictation Mode
       Note over CmdRec: VAD Active (stop words)
       Note over DictRec: VAD Active (full capture)

       User->>DictRec: Says "Hello world..."
       DictRec->>EventBus: DictationAudioSegmentReady

       User->>CmdRec: Says "stop dictation"
       CmdRec->>EventBus: CommandAudioSegmentReady
       EventBus->>STT: Process audio
       STT->>STT: _is_stop_trigger() returns True
       STT->>EventBus: CommandTextRecognized("stop dictation")
       EventBus->>Parser: Parse command
       Parser->>EventBus: DictationStopCommand
       EventBus->>CmdRec: AudioModeChangeRequest(mode="command")
       EventBus->>DictRec: AudioModeChangeRequest(mode="command")

       Note over CmdRec,DictRec: Back to: Command Mode

**Key Observations**:

1. Mode switches happen via event bus broadcasts—both recorders receive the same ``AudioModeChangeRequestEvent``
2. During dictation, **both recorders are active**: dictation recorder captures full text, command recorder listens for stop words
3. The STT service implements stop word detection logic, filtering command audio during dictation mode
4. Thread overhead is eliminated—recorders simply wake/sleep rather than start/stop

This dual-recorder architecture provides the foundation for responsive mode switching. Next, we examine how captured audio is converted to text through specialized STT engines.


Speech-to-Text Processing
==========================

Once audio segments are captured and published by the recorders, they must be converted to text. The ``SpeechToTextService`` orchestrates two complementary STT engines, each optimized for different use cases:

- **Vosk Engine**: Lightweight Kaldi-based model delivering sub-100ms recognition for rapid command execution (CPU-efficient)
- **Whisper Engine**: OpenAI Whisper model providing superior accuracy with natural punctuation for dictation (GPU/CPU)

Engine Selection Strategy
--------------------------

The choice of engine represents a fundamental speed/accuracy tradeoff:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Vosk (Command Mode)
     - Whisper (Dictation Mode)
   * - Latency
     - 50-100ms typical
     - 200-500ms typical
   * - Accuracy
     - Good for short commands
     - Excellent for natural speech
   * - Punctuation
     - No punctuation
     - Full punctuation support
   * - Model Size
     - ~40 MB (small-en-us-0.15)
     - ~140 MB (base model)
   * - Use Case
     - "scroll down", "click", "mark"
     - Full sentences and paragraphs
   * - Processing
     - Synchronous, fast
     - Slower but accurate

Mode-Aware Audio Routing
-------------------------

The STT service implements intelligent routing based on the current system mode and audio source. This diagram shows the complete decision tree:

.. mermaid::

   flowchart TD
       A[Audio Segment Event] --> B{Event Type?}
       B -->|CommandAudioSegmentReady| C[Command Audio Handler]
       B -->|DictationAudioSegmentReady| D[Dictation Audio Handler]

       C --> E{dictation_active<br/>flag?}
       E -->|True| F[Vosk: Stop Word<br/>Detection Only]
       E -->|False| G[Vosk: Full<br/>Recognition]

       F --> H{Text matches<br/>stop_trigger?}
       H -->|Yes| I[Publish<br/>CommandTextRecognized<br/>Exit dictation mode]
       H -->|No| J[Ignore<br/>silently]

       G --> K{Text<br/>recognized?}
       K -->|Yes| L[Duplicate Filter<br/>1 second window]
       K -->|No| M[Forward to<br/>Sound Recognition]

       L --> N{Is<br/>duplicate?}
       N -->|No| I
       N -->|Yes| J

       D --> O[Whisper:<br/>Full Transcription]
       O --> P{Text<br/>recognized?}
       P -->|Yes| Q[Duplicate Filter<br/>1 second window]
       P -->|No| R[End<br/>quietly]

       Q --> S{Is<br/>duplicate?}
       S -->|No| T[Publish<br/>DictationTextRecognized]
       S -->|Yes| R

       style F fill:#fff3cd
       style G fill:#d1ecf1
       style O fill:#d1ecf1
       style M fill:#e7d4f5

**Key Routing Rules**:

1. **Command audio in command mode** → Vosk full recognition → publish text or forward to sound recognition if empty
2. **Command audio in dictation mode** → Vosk stop word check only → publish if stop word detected, otherwise ignore
3. **Dictation audio** → Whisper full recognition → publish text if recognized
4. **Empty Vosk results** → forward to sound recognition system (enables non-verbal commands like finger snaps)

Duplicate Filtering
-------------------

To prevent stuttering from double-detection or echo, the STT service implements a temporal duplicate filter:

.. mermaid::

   sequenceDiagram
       participant VAD as Audio Recorder
       participant STT as STT Service
       participant Filter as Duplicate Filter<br/>(1-second window)
       participant EventBus as Event Bus

       VAD->>STT: Audio segment 1
       STT->>STT: Recognize: "scroll down"
       STT->>Filter: Check: "scroll down"
       Filter->>Filter: Not in cache
       Filter->>STT: OK to publish
       STT->>EventBus: CommandTextRecognized("scroll down")
       Filter->>Filter: Cache: "scroll down" @ t=1000ms

       Note over VAD,Filter: 200ms later

       VAD->>STT: Audio segment 2 (echo)
       STT->>STT: Recognize: "scroll down"
       STT->>Filter: Check: "scroll down"
       Filter->>Filter: Found in cache @ t=1000ms<br/>Current: t=1200ms<br/>Δ = 200ms < 1000ms threshold
       Filter->>STT: DUPLICATE - block
       STT->>STT: Silently drop event

       Note over VAD,Filter: 900ms later (1100ms total)

       VAD->>STT: Audio segment 3
       STT->>STT: Recognize: "scroll down"
       STT->>Filter: Check: "scroll down"
       Filter->>Filter: Cache expired (Δ > 1000ms)
       Filter->>STT: OK to publish
       STT->>EventBus: CommandTextRecognized("scroll down")

The filter maintains a 5-entry LRU cache with 1-second expiry per unique text string. This prevents rapid repeats while allowing intentional command repetition after a brief pause.

Stop Word Detection During Dictation
-------------------------------------

A critical feature: during dictation mode, the command recorder remains active but STT filters everything except configured stop words. This prevents normal speech from interrupting dictation while enabling clean exit:

.. code-block:: python

   # From SpeechToTextService._handle_command_audio_segment()
   if is_dictation_active:
       vosk_result = await self.vosk_engine.recognize(audio_bytes, sample_rate)

       if self._is_stop_trigger(vosk_result):  # Checks for "stop dictation"
           await self._publish_recognition_result(vosk_result, 0, "vosk", STTMode.COMMAND)
       else:
           # Silently ignore - prevents normal speech from being treated as commands
           logger.debug(f"No stop trigger in '{vosk_result}' - ignoring during dictation")
       return

Without this filtering, saying "Hello world" during dictation could accidentally trigger a "hello" command. The stop word filter ensures only intentional control phrases interrupt dictation flow.

From STT to Action
-------------------

Recognized text events (``CommandTextRecognized`` or ``DictationTextRecognized``) are published to the event bus for downstream processing. Command text proceeds to the command parser for interpretation and execution. Dictation text is assembled, formatted (optionally via LLM), and typed into the active application.

But what happens when Vosk returns empty—when the user makes a sound that isn't speech? This leads us to the sound recognition system.

Sound Recognition
=================

Not all commands require speech. Vocalance supports **non-verbal commands** through sound recognition—enabling control via finger snaps, whistles, tongue clicks, or any distinctive sound. When Vosk STT returns empty (no speech detected), audio is forwarded to the ``SoundService`` for acoustic pattern matching.

Why Sound Recognition?
-----------------------

Non-verbal commands offer several advantages:

- **Quieter**: Use voice control in noise-sensitive environments
- **Faster**: A snap is quicker than saying "click"
- **Discrete**: Less disruptive in shared spaces
- **Accessible**: Alternative input for users with speech difficulties

The Challenge of Custom Sound Detection
----------------------------------------

Unlike speech recognition (trained on millions of hours of speech), custom sound detection must work with just a handful of user samples. The system must:

1. Learn the user's specific sound from 10-12 examples
2. Distinguish it from ambient noise (keyboard typing, mouse clicks, coughing)
3. Generalize despite variation in volume, distance, and acoustic environment

Vocalance solves this through **transfer learning** with YAMNet and **few-shot classification** with k-NN.

YAMNet Embeddings + k-NN Classification
----------------------------------------

The architecture combines Google's pre-trained YAMNet audio embedding model with a k-Nearest Neighbors classifier:

.. mermaid::

   flowchart LR
       A[Audio Waveform] --> B[YAMNet Model<br/>Pre-trained on<br/>AudioSet 521 classes]
       B --> C[1024-dim<br/>Embedding Vector]

       subgraph Training
           D[12x Finger Snap<br/>Samples] --> B
           B --> E[12x Embeddings]
           E --> F[k-NN Classifier]

           G[ESC-50 Negatives<br/>keyboard, mouse, cough] --> B
           B --> H[40x Negative<br/>Embeddings]
           H --> F
       end

       subgraph Recognition
           C --> I[k-NN: Find<br/>7 Nearest Neighbors]
           I --> J{Vote > 35%<br/>threshold?}
           J -->|Yes| K[Recognized:<br/>finger_snap]
           J -->|No| L[No match]
       end

       F -.Trained Model.-> I

       style B fill:#e8f5e9
       style F fill:#fff3e0
       style K fill:#e1f5fe

**YAMNet** transforms raw audio into a rich 1024-dimensional embedding that captures acoustic properties. These embeddings cluster similar sounds (all finger snaps close together) while separating dissimilar sounds (finger snaps far from keyboard typing).

**k-NN Classifier** (k=7, vote threshold=35%) performs few-shot learning: given 12 positive examples (user's sound) and 40 negative examples (ESC-50 environmental sounds), it classifies new audio by voting among the 7 nearest neighbors. If ≥35% vote for the custom sound, it's recognized.

Training Workflow
~~~~~~~~~~~~~~~~~~

The user trains a custom sound through a simple UI-driven flow:

.. mermaid::

   sequenceDiagram
       participant UI as Sound Control UI
       participant Service as SoundService
       participant Recognizer as SoundRecognizer
       participant YAMNet as YAMNet Model
       participant Storage as Storage Service

       UI->>Service: SoundTrainingRequestEvent<br/>("finger_snap", 12 samples)
       Service->>Service: Set training_active=True
       Service->>UI: SoundTrainingInitiatedEvent

       loop 12 times
           Note over UI,Service: User makes sound (e.g., snap)
           Service->>YAMNet: Generate embedding
           YAMNet->>Service: 1024-dim vector
           Service->>UI: SoundTrainingProgressEvent<br/>(sample 1/12, 2/12, ...)
       end

       Service->>Recognizer: train_sound("finger_snap", samples)
       Recognizer->>Recognizer: Combine with ESC-50 negatives
       Recognizer->>Recognizer: Train k-NN classifier
       Recognizer->>Storage: Save trained model
       Recognizer->>Service: Success
       Service->>UI: SoundTrainingCompleteEvent

The service collects 12 embeddings (default, configurable) from user demonstrations, combines them with 40 pre-loaded ESC-50 negative samples (keyboard typing, mouse clicks, breathing, coughing, wind, brushing teeth, drinking/sipping), trains the k-NN classifier, and persists the model.

Recognition Workflow
--------------------

During normal operation, unrecognized audio flows into sound recognition:

.. mermaid::

   flowchart TD
       A[Vosk STT returns empty] --> B[STT Service publishes<br/>ProcessAudioChunkForSoundRecognitionEvent]
       B --> C[SoundService receives audio]

       C --> D{training_active?}
       D -->|Yes| E[Collect embedding<br/>for training]
       D -->|No| F[SoundRecognizer:<br/>recognize_sound]

       F --> G[YAMNet: Generate<br/>embedding]
       G --> H[k-NN: Find 7<br/>nearest neighbors]
       H --> I{Vote threshold<br/>> 35%?}

       I -->|Yes| J[Get winning label]
       I -->|No| K[No recognition]

       J --> L{Is custom sound?<br/>not esc50_*}
       L -->|Yes| M[Look up<br/>command mapping]
       L -->|No ESC-50| K

       M --> N{Mapping<br/>exists?}
       N -->|Yes| O[Publish CustomSoundRecognizedEvent<br/>with command text]
       N -->|No| P[Publish event<br/>without command]

       O --> Q[Command Parser]
       Q --> R[Execute Command]

       style G fill:#e8f5e9
       style H fill:#fff3e0
       style O fill:#e1f5fe

ESC-50 negative examples ensure the classifier rejects common environmental sounds. If the k-NN vote indicates "esc50_keyboard_typing", the event is ignored. Only custom sounds with command mappings proceed to execution.

Sound-to-Command Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~

After training, users assign command phrases to sounds via the UI:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Sound Label
     - Command Mapping
     - Behavior
   * - ``finger_snap``
     - ``"click"``
     - Performs mouse click at cursor
   * - ``whistle``
     - ``"scroll down"``
     - Scrolls page down
   * - ``tongue_click``
     - ``"press enter"``
     - Presses Enter key
   * - ``double_snap``
     - ``"mark"``
     - Creates positional mark

Mappings are persisted to storage and loaded on startup. When ``SoundService`` recognizes a trained sound, it looks up the mapping and publishes ``CustomSoundRecognizedEvent(label="finger_snap", mapped_command="click")``. The command parser treats this identically to speech-recognized text, parsing and executing "click" as a normal command.

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

The sound recognition system exposes several tunable parameters in ``SoundRecognizerConfig``:

.. code-block:: python

   # Key tunable parameters
   confidence_threshold: 0.15        # Min similarity for recognition
   k_neighbors: 7                    # Number of neighbors for k-NN voting
   vote_threshold: 0.35              # Min vote percentage (2.45/7 = 35%)

   default_samples_per_sound: 12     # Training samples per custom sound
   max_total_esc50_samples: 40       # Negative examples from ESC-50

   energy_threshold: 0.001           # Min RMS energy to process audio

Lower thresholds increase sensitivity (more detections, more false positives). Higher thresholds increase specificity (fewer detections, fewer false positives). The defaults balance usability across environments.

From Sounds to Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sound recognition extends command input beyond speech, enabling discrete control. But even speech commands can be further accelerated through prediction. The next section explores how Vocalance achieves **zero-latency command execution** by predicting commands before STT completes.

Predictive Commands
===================

Even Vosk's fast recognition incurs ~50-100ms processing time plus 180ms silence timeout—totaling 230-280ms from speech start to execution. For repetitive workflows (scrolling through a document, navigating web pages), this latency compounds into noticeable lag. The ``MarkovCommandService`` eliminates this delay by **predicting and executing commands before STT completes**, achieving near-zero perceived latency.

The Zero-Latency Problem
-------------------------

Consider a user scrolling through a document with repeated "scroll down" commands:

.. mermaid::

   timeline
       title Traditional STT Latency (without prediction)

       section Command 1
           t=0ms : User starts saying "scroll down"
           t=60ms : Vosk recognizes (60ms)
           t=180ms : Silence timeout expires
           t=240ms : Command executes
           t=240ms : Total latency = 240ms

       section Command 2
           t=0ms : User says "scroll down" again
           t=60ms : Vosk recognizes
           t=180ms : Silence timeout
           t=240ms : Command executes
           t=480ms : Cumulative time for 2 commands = 480ms

Each command incurs ~240ms delay. For rapid sequences, this feels sluggish and breaks flow.

The Prediction Solution
------------------------

Markov prediction executes commands at **audio detection** (5-10ms) instead of recognition completion (240ms):

.. mermaid::

   timeline
       title With Markov Prediction

       section Command 1
           t=0ms : User starts saying "scroll down"
           t=10ms : Audio detected → Markov predicts "scroll down"
           t=10ms : Command executes IMMEDIATELY
           t=60ms : Vosk confirms (matches prediction, skip duplicate)
           t=10ms : Total latency = 10ms (24x faster)

       section Command 2
           t=0ms : User says "scroll down" again
           t=10ms : Markov predicts + executes
           t=10ms : Command executes
           t=20ms : Cumulative time for 2 commands = 20ms (24x faster)

Prediction Flow
----------------

The ``AudioRecorder`` publishes ``AudioDetectedEvent`` as soon as VAD energy exceeds threshold—before any audio is captured or recognized. This triggers the predictor:

.. mermaid::

   sequenceDiagram
       participant User
       participant Recorder as Command Recorder
       participant Markov as MarkovCommandService
       participant STT as STT Service
       participant Parser as Command Parser

       User->>Recorder: Starts saying "scroll down"
       Note over Recorder: Energy > threshold (5-10ms)
       Recorder->>Markov: AudioDetectedEvent

       Note over Markov: Check command history<br/>Last 3 commands: [scroll down, scroll down, scroll down]
       Markov->>Markov: 4th-order prediction: "scroll down" (confidence: 1.0)
       Markov->>Parser: MarkovPredictionEvent(predicted="scroll down")
       Parser->>Parser: Execute immediately

       Note over STT: Still processing... (230ms later)
       Recorder->>STT: CommandAudioSegmentReady (audio complete)
       STT->>STT: Vosk recognition
       STT->>Parser: CommandTextRecognized("scroll down")
       Parser->>Parser: Matches prediction - skip duplicate
       Parser->>Markov: MarkovPredictionFeedbackEvent(correct=True)

       Note over Markov: Update history, reset cooldown

**Key Observations**:

1. Prediction happens at ~10ms (audio detection), execution at ~10ms
2. STT confirmation arrives at ~240ms—already executed
3. Parser deduplicates: if STT matches prediction within 1-second window, skip re-execution
4. Feedback loop: correct predictions reinforce confidence, incorrect predictions trigger cooldown

Multi-Order Markov Chains with Backoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The service trains three Markov chain orders simultaneously, using **backoff** strategy for prediction:

.. mermaid::

   flowchart TD
       A[Audio Detected] --> B[Get Recent Command History]
       B --> C{Have 3+ commands<br/>in history?}

       C -->|Yes| D[Try 4th-order prediction<br/>P next | last 3 commands]
       C -->|No| E{Have 2+ commands<br/>in history?}

       D --> F{Confident match?<br/>Count >= 10<br/>Prob = 1.0}
       F -->|Yes| G[Predict with order=4]
       F -->|No| E

       E -->|Yes| H[Try 3rd-order prediction<br/>P next | last 2 commands]
       E -->|No| I{Have 1+ commands<br/>in history?}

       H --> J{Confident match?<br/>Count >= 10<br/>Prob = 1.0}
       J -->|Yes| K[Predict with order=3]
       J -->|No| I

       I -->|Yes| L[Try 2nd-order prediction<br/>P next | last 1 command]
       I -->|No| M[No prediction]

       L --> N{Confident match?<br/>Count >= 15<br/>Prob = 1.0}
       N -->|Yes| O[Predict with order=2]
       N -->|No| M

       G --> P[Execute Command]
       K --> P
       O --> P

       style D fill:#e3f2fd
       style H fill:#e1f5fe
       style L fill:#e0f7fa
       style P fill:#c8e6c9

**Order Selection Logic**:

- **4th-order** (context = last 3 commands): Most specific, requires 10+ occurrences, considers last 1500 commands or 60 days
- **3rd-order** (context = last 2 commands): Medium specificity, requires 10+ occurrences, considers last 1000 commands or 21 days
- **2nd-order** (context = last 1 command): Least specific, requires 15+ occurrences (higher bar), considers last 500 commands or 7 days

**Backoff rationale**: Higher-order chains capture complex patterns ("mark" → "click" → "scroll down" → **"scroll down"** repeats) but need more training data. Lower orders are more general ("scroll down" → **"scroll down"**) and work with less history.

Training Configuration
-----------------------

From ``MarkovPredictorConfig``:

.. code-block:: python

   confidence_threshold: 1.0  # Only predict at 100% confidence (deterministic pattern)

   training_window_commands: {
       2: 500,    # Last 500 commands for 2nd-order
       3: 1000,   # Last 1000 commands for 3rd-order
       4: 1500    # Last 1500 commands for 4th-order
   }

   training_window_days: {
       2: 7,      # Last 7 days for 2nd-order
       3: 21,     # Last 21 days for 3rd-order
       4: 60      # Last 60 days for 4th-order
   }

   min_command_frequency: {
       2: 15,     # 2nd-order requires 15+ occurrences (higher threshold for generality)
       3: 10,     # 3rd-order requires 10+ occurrences
       4: 10      # 4th-order requires 10+ occurrences
   }

The service retrains models on startup from persisted command history in storage.

Error Handling and Safeguards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Incorrect Prediction Cooldown**: When STT confirms a misprediction, the service enters cooldown mode:

.. code-block:: python

   # From MarkovCommandService._handle_prediction_feedback()
   if not was_correct:
       self._cooldown_remaining = 2  # Skip next 2 commands
       logger.warning(f"Markov INCORRECT: predicted '{predicted}', actual '{actual}' - cooldown")

This prevents cascading errors from distributional shifts (user changes workflow, website layout changes, etc.).

**Temporal Cooldown**: Prevents spam predictions within 50ms windows:

.. code-block:: python

   if time.time() - self._last_prediction_time < 0.05:
       return  # Skip prediction

**Dictation Mode Disabling**: Critical safety feature:

.. code-block:: python

   # From MarkovCommandService._handle_audio_detected_fast_track()
   if self._dictation_active:
       logger.debug("Skipping Markov prediction - dictation mode active")
       return

**Why?** Without this:

1. User says "dictate" → enters dictation mode
2. Markov sees "dictate" in history
3. User starts dictating → audio detected
4. Markov predicts "stop dictation" (common pattern after "dictate")
5. Dictation ends immediately before user speaks!

The service subscribes to ``DictationModeDisableOthersEvent`` to track dictation state and disable predictions during active dictation.

Real-World Performance
-----------------------

For repetitive workflows with established patterns:

- **Latency reduction**: 240ms → 10ms (24x faster)
- **Accuracy**: >95% for deterministic patterns (confidence=1.0)
- **False positive rate**: <5% with cooldown mechanisms
- **User experience**: Near-instant execution, feels like thought-to-action

The system shines in scenarios like:

- Scrolling through documents ("scroll down" × 20)
- Navigating browser tabs ("next tab" × 10)
- Sequential form filling ("tab", "click", "type", repeat)
- Grid navigation ("go", "select 5", "go", "select 8", repeat)

This completes the audio processing pipeline. From microphone input to command execution, Vocalance orchestrates VAD, dual-recorder coordination, multi-engine STT, sound recognition, and predictive execution—all optimized for sub-second response times and natural voice control.
