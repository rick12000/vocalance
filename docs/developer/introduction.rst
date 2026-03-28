Vocalance at a Glance
###########################

Introduction
============

Vocalance is a voice-controlled automation application that transforms spoken commands into keyboard and mouse actions.
This page provides a brief introduction to how the application works, but to actually understand it you'll have to read the rest of the documentation (each page will focus on one aspect of Vocalance's functionality).

System Overview
================

Omitting as much detail as possible, the diagram below shows how Vocalance goes from microphone input to computer action.

.. mermaid::

   flowchart TD
       A[Microphone Input] --> B[AudioRecorder<br/>~30 ms PCM]
       B --> C[AudioService]
       C --> D[VAD worker thread]
       D --> E[CommandAudioListener]
       D --> F[SoundAudioListener]
       E --> G[CommandAudioSegmentReadyEvent]
       F --> H[ProcessAudioChunkForSoundRecognitionEvent]
       G --> J[SpeechToTextService<br/>Vosk]
       J --> N[CommandTextRecognizedEvent]
       H --> P[SoundService]
       P --> Q{Training Active?}
       Q -->|Yes| R[Collect Training Sample]
       Q -->|No| S[Sound Recognition]
       S --> T[CustomSoundRecognizedEvent]
       C --> K[Dictation chunk callback]
       K --> L[DictationCoordinator]
       L --> M[Moonshine STT]
       M --> O[Partial / final dictation events]
       N --> U[CentralizedCommandParser]
       O --> L
       T --> U
       U --> W{Command Type?}
       W -->|Automation| X[AutomationService]
       W -->|Mark| Y[MarkService]
       W -->|Grid| Z[GridService]
       W -->|Dictation| L
       X --> AA[pyautogui execution]
       Y --> AB[Mouse jump]
       Z --> AC[Grid UI display]
       L --> AD[Text output]

The general pattern is:

- The recorder streams ~30 ms PCM frames; ``AudioService`` enqueues them for a VAD worker and optionally forwards raw bytes to dictation
- ``CommandAudioListener`` and ``SoundAudioListener`` apply VAD on the worker thread and publish segment events to the bus (via the main asyncio loop)
- ``SpeechToTextService`` runs Vosk on command segments (full commands, or stop-word-only while dictation is active); it loads Moonshine for use by ``DictationCoordinator``, not for command segments
- Dictation text comes from Moonshine streams in the coordinator (all streaming dictation modes), not from a third audio listener
- ``SoundAudioListener`` publishes sound segments; ``SoundService`` classifies them without going through Vosk/Moonshine
- ``CentralizedCommandParser`` combines command text and sound-derived commands; services execute the parsed result

Event-Driven Architecture
===========================

Vocalance uses an event driven architecture. Services don't call each other directly, instead, they communicate by publishing and subscribing to **events** through a central ``EventBus``.

How It Works: A Concrete Example
----------------------------------

Let's trace through exactly what happens when you say "click" into the microphone. We'll follow the event flow shown in the sequence diagram below, connecting each step to the specific events and services.

**1. Audio capture and routing**

``AudioService`` starts the recorder (16 kHz, ~30 ms frames). Each frame is copied to a dictation callback if registered, and enqueued for the VAD worker. The worker calls ``process_audio_chunk`` on ``CommandAudioListener`` and ``SoundAudioListener``—no per-chunk events on the event bus.

**2. Segment detection**

Listeners buffer normalized float chunks until adaptive VAD sees enough silence (command timing is configurable; sound uses short fixed windows). Completed segments are published with ``asyncio.run_coroutine_threadsafe(event_bus.publish(...), main_loop)`` so native-thread audio work stays off the asyncio loop.

**3. Speech-to-text**

``SpeechToTextService`` subscribes only to ``CommandAudioSegmentReadyEvent`` (and dictation mode toggles). Vosk runs full command recognition outside dictation, or stop-word detection inside dictation. Moonshine is owned by the service as ``moonshine_engine`` but streaming dictation is driven from ``DictationCoordinator`` (``open_dictation_stream``, ``feed_moonshine_audio_chunk``).

**4. Sound Recognition Processing**

The SoundAudioListener publishes ``ProcessAudioChunkForSoundRecognitionEvent`` directly to the
SoundService, bypassing STT entirely. The SoundService recognizes trained sounds or collects
training samples without involvement from the speech-to-text pipeline:

.. code-block:: python

   # SoundAudioListener publishes audio chunks for sound recognition
   event = ProcessAudioChunkForSoundRecognitionEvent(
       audio_chunk=audio_bytes,
       sample_rate=16000
   )
   await event_bus.publish(event)

   # SoundService processes independently
   @event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent)
   async def _handle_audio_chunk(self, event):
       if training_active:
           await self._collect_training_sample(event.audio_chunk)
       else:
           result = await self.recognizer.recognize_sound(event.audio_chunk)
           if result:
               await event_bus.publish(CustomSoundRecognizedEvent(...))

**5. Command Parsing and Execution**

The CentralizedCommandParser receives recognized text from both ``CommandTextRecognizedEvent`` and
``CustomSoundRecognizedEvent``. For sound events, it looks up any mapped command phrase, then
processes all text through the unified command hierarchy:

.. code-block:: python

   # CentralizedCommandParser handles text from both STT and sound recognition
   @event_bus.subscribe(CommandTextRecognizedEvent)
   async def _handle_command_text_recognized(self, event):
       command = await self._parse_text(event.text)
       await event_bus.publish(AutomationCommandParsedEvent(command=command))

   @event_bus.subscribe(CustomSoundRecognizedEvent)
   async def _handle_custom_sound_recognized(self, event):
       # Get mapped command for this sound
       command_text = self._sound_to_command_mapping.get(event.label)
       if command_text:
           command = await self._parse_text(command_text)
           await event_bus.publish(AutomationCommandParsedEvent(command=command))

**6. Command Execution**

Finally, the AutomationService receives the ``AutomationCommandParsedEvent`` and executes
the command using pyautogui:

.. code-block:: python

   # AutomationService executes the parsed command
   @event_bus.subscribe(AutomationCommandParsedEvent)
   async def _handle_automation_command(self, event):
       command = event.command
       if command.action_type == ActionType.CLICK:
           success = await self._execute_command(
               ActionType.CLICK,
               command.action_value
           )

       # Publish execution status
       status_event = CommandExecutedStatusEvent(
           command=command.__dict__,
           success=success
       )
       await event_bus.publish(status_event)

The mouse click happens here, completing the event flow from microphone to computer action.

The Event Flow
---------------

Notice the pattern: each service does its job and publishes an event when done. Other services that care about that event will react to it. No service knows about the others directly.

.. mermaid::

   sequenceDiagram
       participant Recorder as AudioRecorder
       participant AS as AudioService
       participant VAD as VAD worker
       participant CmdListener as CommandAudioListener
       participant SoundListener as SoundAudioListener
       participant Bus as EventBus
       participant STT as SpeechToTextService
       participant SoundSvc as SoundService
       participant Parser as CentralizedCommandParser
       participant Automation as AutomationService

       Note over Recorder,Automation: User says "click"

       loop Each ~30 ms frame
           Recorder->>AS: PCM bytes + timestamp
           AS->>VAD: enqueue chunk
           VAD->>CmdListener: process_audio_chunk
           VAD->>SoundListener: process_audio_chunk
       end

       CmdListener->>Bus: CommandAudioSegmentReadyEvent
       SoundListener->>Bus: ProcessAudioChunkForSoundRecognitionEvent

       Bus->>STT: CommandAudioSegmentReadyEvent
       Bus->>SoundSvc: ProcessAudioChunkForSoundRecognitionEvent

       STT->>Bus: CommandTextRecognizedEvent (engine=vosk)
       SoundSvc->>Bus: (no match)

       Bus->>Parser: CommandTextRecognizedEvent
       Parser->>Bus: AutomationCommandParsedEvent

       Bus->>Automation: AutomationCommandParsedEvent
       Automation->>Automation: pyautogui.click()

This architecture makes the system:

- **Flexible**: Add new services without modifying existing ones
- **Testable**: Test each service in isolation

Next Steps
======================

Now that you know the basics, you can dive into the detailed documentation:

- :doc:`Audio Capture & Listeners <audio_capture_and_listeners>`: How audio is captured and buffered
- :doc:`Speech and Sound Recognition <speech_and_sound_recognition>`: How audio becomes recognized text
- :doc:`Command Parsing & Prediction <command_parsing>`: How text becomes structured commands
- :doc:`Command Execution Services <command_execution_services>`: How commands trigger actions
- :doc:`Dictation System <dictation_system>`: How text becomes dictation outputs (all modes, streaming, LLM, including amend)
- :doc:`User Interface Architecture <user_interface>`: How the UI coordinates with services
- :doc:`Event Bus and Infrastructure <event_bus_and_infrastructure>`: How services communicate
