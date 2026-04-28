Capture
#######

The capture layer is the entry point of the pipeline. A single service
owns the microphone, and for every buffer the device delivers it
publishes one event on the bus. Every consumer that needs raw audio —
the two segmenters, the dictation coordinator, the UI wave-meter —
subscribes to that event the same way it subscribes to anything else.
Nothing else in the application touches the audio device, and there
are no callbacks shared between services.

This chapter describes capture as a dataflow problem, in functional
terms. Questions about *which thread* runs each step are deferred to
:doc:`../foundations/concurrency`.

The shape of the layer
======================

.. mermaid::

   flowchart LR
       Mic[Microphone] --> Cap[AudioCaptureService]
       Cap --> Bus[Event bus]
       Bus --> Cmd[CommandSegmenterService]
       Bus --> Snd[SoundSegmenterService]
       Bus --> Dic[DictationCoordinator]
       Bus --> UI[Dictation popup<br/>wave meter]

One service publishes; everyone else subscribes. The diagram is
literally the architecture: each arrow is a real call site, and there
is no fan-out structure hidden behind it.

The unit of audio
=================

The bus event is ``AudioChunkCapturedEvent``
(``vocalance/app/events/core_events.py``):

.. code-block:: python

   class AudioChunkCapturedEvent(BaseEvent):
       pcm_bytes: bytes
       timestamp: float
       sample_rate: int

One event represents one mono PCM buffer delivered by the audio device
— typically about 30 milliseconds of audio, so roughly thirty events
per second while capture is running. ``timestamp`` is wall-clock time
at delivery; ``sample_rate`` is the rate at which the bytes were
captured.

The capture service
===================

``AudioCaptureService``
(``vocalance/app/services/audio/audio_capture_service.py``) is the
only component that talks to ``sounddevice``. Its public surface is
three methods:

.. code-block:: python

   class AudioCaptureService(Service):
       def start(self) -> None: ...
       def stop(self) -> None: ...
       async def shutdown(self) -> None: ...

``start`` opens an input stream and arms the PortAudio callback.
``stop`` closes the stream. ``shutdown`` is what the lifecycle calls
during teardown; it is just ``stop`` plus the standard
``Service.shutdown`` cleanup.

Inside the service the work splits in two by thread of execution:

- ``_portaudio_callback`` runs on PortAudio's native audio thread. It
  copies the buffer's bytes, takes a timestamp, and asks the asyncio
  loop to publish them. It does no real work itself, because audio
  threads must not block.
- ``_publish_chunk`` runs on the asyncio loop. It builds an
  ``AudioChunkCapturedEvent`` and hands it to the bus.

The hop between those two methods is a single
``loop.call_soon_threadsafe`` call, which the
:doc:`../foundations/concurrency` chapter explains from first
principles. From the perspective of the rest of the application the
hop is invisible: events arrive on the bus on the main asyncio loop,
in order, just like every other event.

Device errors take a similar two-step path. If the stream fails to
open, the service publishes a single ``AudioDeviceErrorEvent`` and
stops; subsequent failures are suppressed so the user does not get
flooded with dialogs.

What the consumers do with the event
====================================

Every chunk subscriber is an ordinary bus subscriber, declared in the
service's ``__init__``:

.. code-block:: python

   class CommandSegmenterService(Service):
       def __init__(self, event_bus, config):
           super().__init__(event_bus)
           ...
           self.subscribe(AudioChunkCapturedEvent, self._handle_audio_chunk)

The four subscribers in the running application are:

================================  =====================================
Subscriber                        What it does with each event
================================  =====================================
``CommandSegmenterService``       Feeds a voice-activity-detection
                                  segmenter tuned for short speech
                                  utterances.
``SoundSegmenterService``         Feeds a second segmenter tuned for
                                  short transients (claps, snaps).
``DictationCoordinator``          Forwards the chunk to the streaming
                                  dictation engine while a dictation
                                  session is active.
``QtDictationPopupController``    Computes an RMS level and updates
                                  the popup wave-meter while it is
                                  visible.
================================  =====================================

The first two are the subject of :doc:`command_flow`; the third belongs to
:doc:`dictation`; the popup belongs to :doc:`user_interface`. None of
them know anything about the audio device. None of them holds a
reference to the capture service. They are wired in only by the bus
event they subscribe to, which is exactly the contract the
:doc:`../overview/architecture` chapter promised.

What leaves the layer
=====================

For every buffer the device delivers, the capture layer:

1. Hops from the audio thread to the asyncio loop.
2. Publishes one ``AudioChunkCapturedEvent`` on the bus.
3. (On failure) publishes one ``AudioDeviceErrorEvent``.

That is the entire output of the layer. Segmentation, streaming
dictation, and wave-meter rendering all live in the chapters that
follow.
