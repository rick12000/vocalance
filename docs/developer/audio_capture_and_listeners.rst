Audio Capture & Listeners
###########################

System Overview
====================================

The recorder emits short PCM chunks; ``AudioService`` receives these chunks on the main asyncio loop and feeds them to **two** segmenters (command and sound). Dictation does **not** use those segmenters: raw PCM is forwarded directly into ``DictationCoordinator`` for Moonshine streaming.

.. mermaid::

   flowchart TD
       A[Microphone] --> B[AudioRecorder<br/>16 kHz ~30 ms chunks]
       B --> C[AudioService]
       C --> D[UtteranceSegmenter<br/>Command]
       C --> F[UtteranceSegmenter<br/>Sound]
       D --> G[CommandAudioSegmentReadyEvent]
       F --> H[ProcessAudioChunkForSoundRecognitionEvent]
       C --> I[DictationCoordinator<br/>Moonshine ingress]

The Two Segmenters
===================

``AudioService`` maintains two ``UtteranceSegmenter`` instances (one for commands and one for sounds) that each run VAD on the same chunk stream with different thresholds and silence rules. When a segment ends, they emit a ``Clip`` hit, and ``AudioService`` schedules a segment event to be published on the event bus via the main asyncio loop.

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
       E -->|Yes| F[Emit Clip hit]
       F --> G[Publish segment event]
       G --> A

Segmenter Comparison
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Segmenter
     - Purpose
     - Output event
   * - Command Segmenter
     - Low-latency command and stop-word segments (configurable silence chunks)
     - ``CommandAudioSegmentReadyEvent``
   * - Sound Segmenter
     - Short non-speech sounds; disabled while dictation is active
     - ``ProcessAudioChunkForSoundRecognitionEvent``

Dictation PCM
-------------

All dictation modes that use streaming STT share the same path: ``AudioService`` directly calls ``feed_moonshine_audio_chunk`` on the ``DictationCoordinator`` with the raw PCM bytes. The coordinator queues audio on a dedicated ingress thread and feeds Moonshine (see :doc:`dictation_system`).

What Happens Next
==================

- ``CommandAudioSegmentReadyEvent`` → ``SpeechToTextService`` (Vosk) → ``CommandTextRecognizedEvent``
- ``ProcessAudioChunkForSoundRecognitionEvent`` → ``SoundService`` → YAMNet / k-NN → ``CustomSoundRecognizedEvent``
- Dictation text → Moonshine partial/final handling in ``DictationCoordinator`` → ``PartialDictationTextEvent``, ``FinalDictationTextEvent``, ``DictationTextRecognizedEvent`` (mode-dependent)

See :doc:`speech_and_sound_recognition` for recognition details.
