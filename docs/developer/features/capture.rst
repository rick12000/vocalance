Capture
#######

.. sectnum::

The capture layer is the microphone end of the pipeline. One service owns
the audio device; every other component that needs audio subscribes to receive
it from the bus.

The Capture Layer
==================

The single entry point for all audio is ``AudioCaptureService``
(``vocalance/app/services/capture/audio_capture_service.py``). It opens the
default input device, listens continuously, and publishes each buffer of audio
as an event on the bus. There are no callbacks between the capture service and
its consumers — every downstream component simply subscribes to the event type.

.. mermaid::

   flowchart LR
       Mic[Microphone] --> Cap[AudioCaptureService]
       Cap -->|AudioChunkCapturedEvent| Bus((Event bus))
       Bus --> Cmd[CommandSegmenterService]
       Bus --> Snd[SoundSegmenterService]
       Bus --> Dic[DictationCoordinator]
       Bus --> UI[Dictation popup<br/>wave meter]

Each subscriber reacts to the same event independently. The capture service does
not know they exist, and they do not know about each other.

AudioChunkCapturedEvent
------------------------

The event published for every buffer is ``AudioChunkCapturedEvent``
(``vocalance/app/events/core_events.py``):

.. code-block:: python

   class AudioChunkCapturedEvent(BaseEvent):
       pcm_bytes: bytes
       timestamp: float
       sample_rate: int

``pcm_bytes`` is a block of 16-bit mono PCM samples — the raw amplitude values
the microphone's analogue-to-digital converter produced, uncompressed and
unencoded. At a 44 100 Hz sample rate with a block size of 1 323 frames, one
block covers exactly 30 milliseconds. The application receives roughly thirty
events per second while capture is running.

Audio Capture Service
----------------------

``AudioCaptureService`` has a minimal public surface:

.. code-block:: python

   class AudioCaptureService(Service):
       def start(self) -> None: ...
       def stop(self) -> None: ...
       async def shutdown(self) -> None: ...

``start`` opens the input stream and registers the PortAudio callback. ``stop``
closes the stream. ``shutdown`` calls ``stop`` and releases bus subscriptions
through the standard ``Service`` teardown.

The rest of the service is two private methods: the PortAudio callback and the
publish method. They live on separate threads, which is why they are separate.

The Thread Crossing
~~~~~~~~~~~~~~~~~~~~

The audio hardware driver operates on a dedicated OS thread it controls — not
the main application thread. Its constraint is strict: the callback it invokes
must copy the buffer and return within a few hundred microseconds, or the
driver's internal ring buffer overflows and audio frames are dropped. Any work
that takes longer — touching a lock, publishing an event, calling any model —
cannot happen inside the callback.

The callback's only permitted work is:

1. **Copy** the raw bytes. The buffer pointer is only valid for the duration of
   the callback; the driver may overwrite that memory as soon as it returns.
2. **Record the timestamp.** A single ``time.time()`` call.
3. **Schedule publish** on the main thread via ``loop.call_soon_threadsafe``
   and return immediately.

``call_soon_threadsafe`` places a callable into a thread-safe internal queue
that the main thread's event loop reads on its next tick, waking the loop if
it is currently idle. The callable runs on the main thread — in the same
single-thread context as all services and bus dispatch — with no threading
hazards.

.. mermaid::

   flowchart LR
       subgraph Foreign["Driver Thread (PortAudio)"]
           PA[PortAudio driver] -->|PCM buffer| CB[_portaudio_callback<br/>1. copy bytes<br/>2. record timestamp<br/>3. schedule publish]
       end
       subgraph Main["Main Thread"]
           Pub[_publish_chunk] -->|AudioChunkCapturedEvent| Bus((Event bus))
       end
       CB -.->|call_soon_threadsafe| Pub

.. code-block:: python

   def _portaudio_callback(self, indata, frames, time_info, status):
       pcm_bytes = indata.tobytes()
       timestamp = time.time()
       self.loop.call_soon_threadsafe(
           self._publish_chunk, pcm_bytes, timestamp
       )

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

The full mechanics of ``call_soon_threadsafe`` and the asyncio event loop are
covered in :doc:`../foundations/concurrency`.

Error Handling
~~~~~~~~~~~~~~~

If the device stream fails to open or encounters a fatal error, the service
publishes one ``AudioDeviceErrorEvent`` and halts. The UI system controller
displays a modal dialog on this event.

With audio chunks flowing on the bus, two independent pipelines take over from
here: the command flow, which segments and classifies each clip as speech or
sound, and the dictation flow, which streams audio directly to a transcription
engine. Both are described in the chapters that follow.
