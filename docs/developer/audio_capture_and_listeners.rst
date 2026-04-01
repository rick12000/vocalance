Audio Capture & Listeners
###########################

System Overview
====================================

The recorder emits short PCM chunks; a VAD worker thread feeds **two** listeners (command and sound). Dictation does **not** use those listeners: raw PCM is forwarded through ``AudioService.set_dictation_chunk_callback`` into ``DictationCoordinator`` for Moonshine streaming.

.. mermaid::

   flowchart TD
       A[Microphone] --> B[AudioRecorder<br/>16 kHz ~30 ms chunks]
       B --> C[AudioService]
       C --> D[VAD worker thread]
       D --> E[CommandAudioListener]
       D --> F[SoundAudioListener]
       E --> G[CommandAudioSegmentReadyEvent]
       F --> H[ProcessAudioChunkForSoundRecognitionEvent]
       C --> I[dictation chunk callback]
       I --> J[DictationCoordinator<br/>Moonshine ingress]

The Two Listeners
===================

``CommandAudioListener`` and ``SoundAudioListener`` each run VAD on the same chunk stream with different thresholds and silence rules. When a segment ends, they publish to the event bus via ``asyncio.run_coroutine_threadsafe`` onto the GUI asyncio loop.

How Audio Listening Works
--------------------------

.. mermaid::

   flowchart LR
       A[PCM chunks] --> B{VAD onset?}
       B -->|No| A
       B -->|Yes| C[Buffer + pre-roll]
       C --> D[Buffer while active]
       D --> E{Silence timeout?}
       E -->|No| D
       E -->|Yes| F[Publish segment event]
       F --> A

Listener Comparison
-------------------

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Listener
     - Purpose
     - Output event
   * - ``CommandAudioListener``
     - Low-latency command and stop-word segments (configurable silence chunks)
     - ``CommandAudioSegmentReadyEvent``
   * - ``SoundAudioListener``
     - Short non-speech sounds; disabled while dictation is active
     - ``ProcessAudioChunkForSoundRecognitionEvent``

Dictation PCM
-------------

All dictation modes that use streaming STT share the same path: the coordinator registers a callback on ``AudioService``; the recorder invokes it on its thread with ``(pcm_bytes, sample_rate)``. The coordinator queues audio on a dedicated ingress thread and feeds Moonshine (see :doc:`dictation_system`).

What Happens Next
==================

- ``CommandAudioSegmentReadyEvent`` → ``SpeechToTextService`` (Vosk) → ``CommandTextRecognizedEvent``
- ``ProcessAudioChunkForSoundRecognitionEvent`` → ``SoundService`` → YAMNet / k-NN → ``CustomSoundRecognizedEvent``
- Dictation text → Moonshine partial/final handling in ``DictationCoordinator`` → ``PartialDictationTextEvent``, ``FinalDictationTextEvent``, ``DictationTextRecognizedEvent`` (mode-dependent)

See :doc:`speech_and_sound_recognition` for recognition details.
