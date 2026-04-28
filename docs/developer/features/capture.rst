Capture
#######

The capture layer is the entry point of the pipeline. One service
owns the microphone; for every buffer the device delivers, it
publishes one event on the bus. Every consumer that needs raw audio
— the two segmenters, the dictation coordinator, the popup
wave-meter — subscribes to that event the same way it subscribes
to anything else.

This chapter describes capture as a dataflow problem. Threading
questions are deferred to :doc:`../foundations/concurrency`.

Layer at a glance
=================

One publisher, four subscribers, one event type. Each arrow on the
diagram is a real call site.

.. mermaid::

   flowchart LR
       Mic[Microphone] --> Cap[AudioCaptureService]
       Cap -->|AudioChunkCapturedEvent| Bus((Event bus))
       Bus --> Cmd[CommandSegmenterService]
       Bus --> Snd[SoundSegmenterService]
       Bus --> Dic[DictationCoordinator]
       Bus --> UI[QtDictationPopupController<br/><i>wave meter</i>]

Each subscriber treats the chunk differently:

================================  ========================================
Subscriber                        What it does with each chunk
================================  ========================================
``CommandSegmenterService``       VAD-segments speech utterances
                                  (:doc:`command_flow`).
``SoundSegmenterService``         VAD-segments short transients
                                  (:doc:`command_flow`).
``DictationCoordinator``          Forwards to the streaming engine while
                                  a session is active
                                  (:doc:`dictation_flow`).
``QtDictationPopupController``    Computes RMS to drive the wave meter
                                  (:doc:`user_interface`).
================================  ========================================

None of them holds a reference to ``AudioCaptureService``. Wiring
is the bus event, nothing else.

The unit of audio
=================

The bus event is ``AudioChunkCapturedEvent``
(``vocalance/app/events/core_events.py``):

.. code-block:: python

   class AudioChunkCapturedEvent(BaseEvent):
       pcm_bytes: bytes
       timestamp: float
       sample_rate: int

One event represents one mono PCM buffer delivered by the audio
device — typically about 30 milliseconds, so roughly thirty events
per second while capture is running. ``timestamp`` is wall-clock
time at delivery; ``sample_rate`` is the rate the bytes were
captured at.

The capture service
===================

``AudioCaptureService``
(``vocalance/app/services/capture/audio_capture_service.py``) is
the only component that talks to ``sounddevice``. Its public
surface is three methods:

.. code-block:: python

   class AudioCaptureService(Service):
       def start(self) -> None: ...
       def stop(self) -> None: ...
       async def shutdown(self) -> None: ...

``start`` opens an input stream and arms the PortAudio callback.
``stop`` closes the stream. ``shutdown`` is what the lifecycle
calls during teardown; it is ``stop`` plus the standard
``Service.shutdown`` cleanup.

Inside the service, the work splits in two by thread of execution.

.. mermaid::

   flowchart LR
       PA[PortAudio thread] -->|raw PCM buffer| CB[_portaudio_callback]
       CB -->|copy bytes,<br/>schedule| Loop[asyncio loop]
       Loop --> Pub[_publish_chunk]
       Pub -->|AudioChunkCapturedEvent| Bus((Event bus))

The callback's only job is to copy and schedule. Anything heavier
on the audio thread — running a model, taking a lock, allocating
a large object — would risk dropping audio.

.. code-block:: python

   def _portaudio_callback(self, indata, frames, time_info, status):
       pcm_bytes = indata.tobytes()
       timestamp = time.time()
       self.loop.call_soon_threadsafe(self._publish_chunk, pcm_bytes, timestamp)

   def _publish_chunk(self, pcm_bytes, timestamp):
       asyncio.create_task(
           self.event_bus.publish(
               AudioChunkCapturedEvent(
                   pcm_bytes=pcm_bytes,
                   timestamp=timestamp,
                   sample_rate=self.sample_rate,
               )
           )
       )

The hop between callback and publish is a single
``loop.call_soon_threadsafe`` call.
:doc:`../foundations/concurrency` explains the primitive from
first principles. From the rest of the application's perspective
the hop is invisible: events arrive on the bus, in order, like
any other event.

Device errors take the same two-step path. If the stream fails to
open, the service publishes one ``AudioDeviceErrorEvent`` and
stops; subsequent failures are suppressed so the user is not
flooded with dialogs.

What leaves the layer
=====================

For every buffer the device delivers, the capture layer:

#. Hops from the audio thread to the asyncio loop.
#. Publishes one ``AudioChunkCapturedEvent``.
#. (On failure) publishes one ``AudioDeviceErrorEvent``.

Segmentation, streaming dictation, and wave-meter rendering live
in the chapters that follow.

Where to read next
==================

Two flows fan out from this layer:

- :doc:`command_flow` — segmenters, recognizers, parser,
  executors.
- :doc:`dictation_flow` — the streaming session.
