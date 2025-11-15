Speech & Sound Recognition
############################

This page explains how Vocalance transforms audio segments into recognized text and custom sound events using two parallel recognition pipelines: speech-to-text (STT) for voice commands and dictation, and sound recognition for non-speech sounds.

System Overview
================

After the audio listeners detect speech and sound segments (see :doc:`audio_capture_and_listeners`), those segments flow into two independent recognition pipelines that operate in parallel.

.. mermaid::

   flowchart TD
       A[CommandAudioSegmentReadyEvent] --> B[SpeechToTextService]
       C[DictationAudioSegmentReadyEvent] --> B
       D[ProcessAudioChunkForSoundRecognitionEvent] --> E[SoundService]

       B --> F{Mode?}
       F -->|Command| G[Vosk Engine<br/>Fast offline]
       F -->|Dictation| H[Whisper Engine<br/>Accurate model-based]

       G --> I[CommandTextRecognizedEvent]
       H --> J[DictationTextRecognizedEvent]

       E --> K{Training?}
       K -->|Yes| L[Collect Samples]
       K -->|No| M[YAMNet Recognition]
       M --> N[CustomSoundRecognizedEvent]

       style B fill:#e1f5ff
       style E fill:#fce4ec
       style G fill:#fff4e1
       style H fill:#e8f5e9

Understanding the Two Recognition Pipelines
=============================================

The diagram above illustrates how Vocalance processes audio through two independent, parallel recognition pipelines:

- **Speech-to-Text Pipeline**: Processes both command and dictation audio through the SpeechToTextService, which routes segments to either Vosk (fast command recognition) or Whisper (accurate dictation) depending on the current mode and segment type.

- **Sound Recognition Pipeline**: Processes sound audio (clicks, pops, whistles, etc.) through the SoundService, which either collects training samples (when training a custom sound) or performs k-NN based recognition using YAMNet embeddings (during normal operation).

Pipeline 1: Speech-to-Text Pipeline
====================================

Dual-Engine Strategy
--------------------

+----------+-------------------+----------------------+
|          | Vosk              | Whisper              |
+==========+===================+======================+
| Accuracy | Good for short    | Good for full        |
|          | phrases           | sentences            |
+----------+-------------------+----------------------+
| Speed    | Fast              | Medium               |
|          |                   |                      |
+----------+-------------------+----------------------+
| Memory   | 50MB              | 1GB                  |
|          |                   |                      |
+----------+-------------------+----------------------+

The ``SpeechToTextService`` manages two STT engines:

- Vosk: Fast command recognition for single or dual-word commands
- Whisper: Accurate dictation transcription for full sentences

Dictation Mode
---------------------------

**Command audio is processed differently depending on whether dictation mode is active**. When you enter dictation mode, Vocalance must allow dictation to proceed without command recognition interfering. However, stop triggers (like "stop dictation") must still be detected to exit dictation mode.

An outline of how Vocalance handles switching in and out of dictation mode is provided below:

.. mermaid::

   sequenceDiagram
       participant U as User
       participant Parser as CommandParser
       participant STT as SpeechToTextService
       participant Coord as DictationCoordinator

       U->>Parser: Say "start dictation"
       Parser->>Coord: DictationCommandParsedEvent
       Coord->>STT: DictationModeDisableOthersEvent(active=true)
       Note over STT: Switch to stop-detection-only mode

       U->>STT: Say "hello world" (command segment)
       STT->>STT: Check for stop trigger only
       Note over STT: No stop word found, discard

       U->>STT: Say "and this is great" (dictation segment)
       STT->>Coord: DictationTextRecognizedEvent(text="...")

       U->>STT: Say "stop dictation" (command segment)
       STT->>STT: Detect stop trigger!
       STT->>Parser: CommandTextRecognizedEvent(text="stop dictation")
       Parser->>Coord: DictationStopCommand
       Coord->>STT: DictationModeDisableOthersEvent(active=false)
       Note over STT: Resume full command recognition

Pipeline 2: Sound Recognition Pipeline
=======================================

The ``SoundService`` handles recognition of custom non-speech sounds (clicks, pops, whistles, etc.), allowing the user to execute commands without speaking. Sound recognition uses a two-phase workflow: training and recognition.

Training Custom Sounds
----------------------

Training builds a model by collecting audio samples of your custom sound and extracting acoustic features:

.. mermaid::

   flowchart TD
       A[Start Training] --> B[Collect Audio Samples]
       B --> C{Enough Samples?}
       C -->|No| D[Publish Progress]
       D --> B
       C -->|Yes| E[Extract YAMNet Embeddings]
       E --> F[Add ESC-50 Background Examples]
       F --> G[Fit StandardScaler]
       G --> H[Save Model Files]
       H --> I[SoundTrainingCompleteEvent]

       style A fill:#fff4e1
       style E fill:#e8f5e9
       style H fill:#e8f5e9
       style I fill:#e1f5ff

**Workflow**:

1. **Collection phase** – User initiates training (e.g., "Train tongue click"). System collects 12 audio samples during normal operation and preprocesses each: resampling to target rate, normalizing audio levels.

2. **Embedding extraction** – Once enough samples are collected, the system extracts acoustic features using YAMNet, a pre-trained audio event detection model. Each sample becomes a 1024-dimensional feature vector capturing its acoustic characteristics.

3. **Background robustness** – Environmental sounds from ESC-50 (birds, traffic, doors) are added as negative examples. This teaches the model to distinguish your custom sound from background noise.

4. **Normalization & storage** – A StandardScaler normalizes embeddings for consistent comparisons. All embeddings, labels, and the scaler are saved for recognition use.

Recognizing Custom Sounds
--------------------------

During recognition, incoming audio is matched against trained sounds using nearest-neighbor search with confidence filtering:

.. mermaid::

   flowchart TD
       A[Audio Chunk Arrives] --> B[Preprocess Audio]
       B --> C[Extract YAMNet Embedding]
       C --> D[Scale Embedding]
       D --> E[k-NN Search<br/>k=7, Cosine Similarity]
       E --> F{Best Match<br/>Confidence > 0.15?}
       F -->|No| G[Discard]
       F -->|Yes| H{Custom Sound<br/>Vote Ratio > 0.35?}
       H -->|No| I[Discard]
       H -->|Yes| J[CustomSoundRecognizedEvent]

       style A fill:#fff4e1
       style E fill:#e8f5e9
       style J fill:#e1f5ff

**Workflow**:

1. **Preprocessing** – Convert incoming audio to float32, resample to target rate, normalize audio levels.

2. **Feature extraction** – Extract a YAMNet embedding (1024-dimensional vector) from the audio, matching the training approach.

3. **Feature scaling** – Apply the saved StandardScaler to normalize the embedding for consistent similarity comparisons.

4. **Neighbor matching** – Find the 7 nearest neighbors (k=7) from trained embeddings using cosine similarity as the distance metric.

5. **Confidence filtering** – The best match must exceed 0.15 similarity (configurable). Matches below this threshold are discarded as false positives.

6. **Vote filtering** – Among the k=7 matches, check how many are actual trained sounds vs. background samples. Custom sounds must win at least 35% of the votes (configurable) to be recognized.

7. **Event emission** – If both confidence and vote checks pass, emit ``CustomSoundRecognizedEvent`` with the label and confidence score.

Recognition Output Events
==========================

All three recognition pipelines converge at output events that downstream services consume:

.. code-block:: python

   CommandTextRecognizedEvent(
       text="click",
       engine="vosk",
       mode="command",
       processing_time_ms=75.5
   )

Produced by Vosk when a command is recognized. Emitted every time a command segment is recognized in normal mode, or when a stop trigger is detected in dictation mode. The ``CentralizedCommandParser`` subscribes and parses the text into structured automation commands.

.. code-block:: python

   DictationTextRecognizedEvent(
       text="Hello world and welcome to the demo.",
       engine="whisper",
       mode="dictation",
       processing_time_ms=850.2
   )

Produced by Whisper when dictation text is recognized. Emitted only when dictation mode is active; Whisper processing is skipped otherwise to save computational resources. The ``DictationCoordinator`` subscribes and routes to the appropriate dictation handler (standard text insertion, visual preview, or smart context-aware mode).

.. code-block:: python

   CustomSoundRecognizedEvent(
       label="tongue_click",
       confidence=0.89,
       mapped_command="click"
   )

Produced by SoundService when a trained custom sound is recognized with sufficient confidence and vote agreement. ESC-50 background sounds are never emitted. The ``CentralizedCommandParser`` subscribes, optionally maps the sound label to a command phrase, and processes it as a command.

Event Routing Summary
=====================

.. list-table::
   :header-rows: 1

   * - Event
     - Source
     - Route
     - Purpose
   * - ``CommandTextRecognizedEvent``
     - Vosk
     - CentralizedCommandParser
     - Parse voice commands
   * - ``DictationTextRecognizedEvent``
     - Whisper
     - DictationCoordinator
     - Insert dictated text
   * - ``CustomSoundRecognizedEvent``
     - SoundService
     - CentralizedCommandParser
     - Execute sound-triggered commands

What Happens Next
==================

Recognized text and sound events now flow into the command parser, which determines what action to take:

- **Speech-to-text pipeline**:

    - ``CommandTextRecognizedEvent`` → ``CentralizedCommandParser`` → Execute click action
    - ``DictationTextRecognizedEvent`` → ``DictationCoordinator`` → Insert text

- **Sound recognition pipeline**:

    - ``CustomSoundRecognizedEvent`` → ``CentralizedCommandParser`` → Map to ``click`` → Execute

The command parsing pipeline, including Markov chain prediction and deduplication, is covered in :doc:`command_parsing`.
