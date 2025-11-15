Audio Capture & Listeners
###########################

System Overview
====================================

Audio capture handles continuous streaming of recorded chunks, while dedicated listeners buffer and publish the chunks according to custom VAD and silence timeout parameters:

.. mermaid::

   flowchart TD
       A[Microphone] --> B[AudioRecorder<br/>16kHz streaming]
       B --> C[AudioChunkEvent<br/>50ms chunks]
       C --> D[Command Listener<br/>150ms timeout]
       C --> E[Dictation Listener<br/>800ms timeout]
       C --> F[Sound Listener<br/>100ms timeout]
       D --> G[Command Segment]
       E --> H[Dictation Segment]
       F --> I[Sound Segment]

The Three Listeners
===================

AudioRecorder captures continuous 16kHz audio, publishing 50ms chunks as events. Three listeners subscribe simultaneously to these chunks, each applying Voice Activity Detection (VAD) to identify meaningful audio segments with parameters designed for their specific use case.

How Audio Listening Works
--------------------------

.. mermaid::

   flowchart LR
       A[Audio Chunks] --> B{VAD Activity<br/>Detected?}
       B -->|No| A
       B -->|Yes| C[Start Buffering<br/>+ Pre-roll]
       C --> D[Continue Buffering<br/>During Activity]
       D --> E{Silence Timeout<br/>Reached?}
       E -->|No| D
       E -->|Yes| F[Publish<br/>Buffered Segment]
       F --> A

The listeners constantly receive audio chunks but remain idle until VAD detects the start of speech or sound activity. When audio energy first exceeds the speech start threshold, buffering begins, capturing both the current chunk and any buffered pre-roll data to ensure word onsets aren't missed. Buffering continues as long as audio energy stays above the speech end threshold. When silence persists for the configured timeout period, the buffered segment is published as a ready event and buffering stops until the next VAD trigger.

Listener Comparison
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - Listener
     - Timeout
     - Purpose
     - Output Event
   * - CommandAudioListener
     - 150ms
     - Voice commands
     - ``CommandAudioSegmentReadyEvent``
   * - DictationAudioListener
     - 800ms
     - Speech transcription
     - ``DictationAudioSegmentReadyEvent``
   * - SoundAudioListener
     - 100ms
     - Non-speech sounds
     - ``ProcessAudioChunkForSoundRecognitionEvent``

- **CommandAudioListener**: Processes quick voice commands with a 150ms timeout for immediate execution. Looks for single or dual word commands, not sentences, to achieve low latency for fast command execution
- **DictationAudioListener**: Handles continuous speech transcription with an 800ms timeout to transcribe full sentences and account for natural pauses between words spoken at normal pace
- **SoundAudioListener**: Detects brief sounds (clicks, pops, whistles) with a near immediate silence timeout of 100ms. Dictation-aware and automatically disables during active dictation sessions to avoid interference

What Happens Next
==================

Audio segments flow into recognition services that transform them into actionable events:

- ``CommandAudioSegmentReadyEvent`` → Vosk engine → command text parsing
- ``DictationAudioSegmentReadyEvent`` → Whisper engine → dictation output
- ``ProcessAudioChunkForSoundRecognitionEvent`` → YAMNet → sound classification

These recognition services are covered in :doc:`speech_and_sound_recognition`.
