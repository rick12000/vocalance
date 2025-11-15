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
       A[Microphone Input] --> B[AudioRecorder]
       B --> C[AudioChunkEvent<br/>50ms chunks]

       C --> D[CommandAudioListener]
       C --> E[DictationAudioListener]
       C --> F[SoundAudioListener]

       D --> G[CommandAudioSegmentReadyEvent]
       E --> H[DictationAudioSegmentReadyEvent]
       F --> I[ProcessAudioChunkForSoundRecognitionEvent]

       G --> J[SpeechToTextService]
       H --> J

       J --> K{Engine Selection}
       K -->|Command Audio| L[Vosk Engine]
       K -->|Dictation Audio| M[Whisper Engine]

       L --> N[CommandTextRecognizedEvent]
       M --> O[DictationTextRecognizedEvent]

       I --> P[SoundService]
       P --> Q{Training Active?}
       Q -->|Yes| R[Collect Training Sample]
       Q -->|No| S[Sound Recognition]
       S --> T[CustomSoundRecognizedEvent]

       N --> U[CentralizedCommandParser]
       O --> V[DictationCoordinator]
       T --> U

       U --> W{Command Type?}
       W -->|Automation| X[AutomationService]
       W -->|Mark| Y[MarkService]
       W -->|Grid| Z[GridService]
       W -->|Dictation| V

       X --> AA[pyautogui execution]
       Y --> AB[Mouse jump]
       Z --> AC[Grid UI display]
       V --> AD[Text output]

The general pattern is:

- The AudioRecorder continuously captures raw audio at 16kHz and publishes 50ms chunks
- Three independent listeners subscribe to these chunks and apply their own Voice Activity Detection (VAD)
- Each listener buffers chunks according to its own parameters (command mode: ~150ms silence timeout, dictation mode: ~800ms, sound: ~100ms)
- When silence timeout is reached, each listener publishes a segment-ready event
- SpeechToTextService processes command and dictation segments with appropriate engines (Vosk/Whisper)
- SoundAudioListener publishes directly to SoundService, bypassing STT
- CentralizedCommandParser receives text from both STT and sound recognition, maps sounds to commands, and publishes parsed commands
- AutomationService executes commands, MarkService handles marks, GridService displays grids, and DictationCoordinator outputs dictated text

Event-Driven Architecture
===========================

Vocalance uses an event driven architecture. Services don't call each other directly, instead, they communicate by publishing and subscribing to **events** through a central ``EventBus``.

How It Works: A Concrete Example
----------------------------------

Let's trace through exactly what happens when you say "click" into the microphone. We'll follow the event flow shown in the sequence diagram below, connecting each step to the specific events and services.

**1. Audio Capture and Continuous Streaming**

The AudioService starts the AudioRecorder, which continuously captures audio at 16kHz sample rate
and publishes an ``AudioChunkEvent`` for every 50ms of audio. The recorder has no VAD logic—it
simply streams raw audio chunks for downstream listeners to process:

.. code-block:: python

   # AudioRecorder captures and publishes 50ms chunks continuously
   while recording:
       audio_chunk = stream.read(800_samples)  # 50ms at 16kHz
       event = AudioChunkEvent(
           audio_chunk=audio_chunk.tobytes(),
           sample_rate=16000,
           timestamp=time.time()
       )
       await event_bus.publish(event)

**2. Parallel Audio Segment Detection**

Three independent listeners subscribe to ``AudioChunkEvent`` and process the same audio stream
in parallel, each with their own VAD parameters and silence timeouts:

- **CommandAudioListener**: Low-latency (~150ms silence timeout) for responsive command detection
- **DictationAudioListener**: Longer timeout (~800ms) tolerant of natural pauses in speech
- **SoundAudioListener**: Quick detection (~100ms) for brief sound recognition

Each listener maintains its own buffer and publishes a segment-ready event when speech/sound is
detected followed by the configured silence timeout:

.. code-block:: python

   # CommandAudioListener processes AudioChunkEvent with command-specific VAD
   @event_bus.subscribe(AudioChunkEvent)
   async def _handle_audio_chunk(self, event):
       # Calculate energy and apply VAD logic
       chunk = np.frombuffer(event.audio_chunk, dtype=np.int16)
       energy = self._calculate_energy(chunk)

       # Buffer chunks until silence timeout
       if energy > threshold and not self._is_recording:
           self._is_recording = True
           self._audio_buffer.append(chunk)
       elif energy < silence_threshold:
           self._consecutive_silent_chunks += 1

           # Publish when silence timeout reached
           if self._consecutive_silent_chunks >= 3:  # 150ms
               event = CommandAudioSegmentReadyEvent(
                   audio_bytes=concatenated_audio.tobytes(),
                   sample_rate=16000
               )
               await event_bus.publish(event)

**3. Speech-to-Text Processing**

The SpeechToTextService subscribes to ``CommandAudioSegmentReadyEvent`` and ``DictationAudioSegmentReadyEvent``.
For command segments, it uses the fast Vosk engine; for dictation segments, it uses the accurate Whisper engine:

.. code-block:: python

   # SpeechToTextService receives audio segments and uses appropriate engine
   @event_bus.subscribe(CommandAudioSegmentReadyEvent)
   async def _handle_command_audio_segment(self, event):
       # In command mode: run full Vosk recognition
       # In dictation mode: check only for stop trigger words
       text = await self.vosk_engine.recognize(event.audio_bytes, event.sample_rate)

       text_event = CommandTextRecognizedEvent(
           text=text,
           engine="vosk",
           mode="command"
       )
       await self.event_bus.publish(text_event)

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
       participant Bus as EventBus
       participant CmdListener as CommandAudioListener
       participant SoundListener as SoundAudioListener
       participant STT as SpeechToTextService
       participant SoundSvc as SoundService
       participant Parser as CentralizedCommandParser
       participant Automation as AutomationService

       Note over Recorder,Automation: User says "click"

       Recorder->>Bus: AudioChunkEvent 50ms chunk #1
       Bus->>CmdListener: (delivers event)
       Bus->>SoundListener: (delivers event)

       Recorder->>Bus: AudioChunkEvent 50ms chunk #2
       Bus->>CmdListener: (delivers event)
       Bus->>SoundListener: (delivers event)

       Recorder->>Bus: AudioChunkEvent 50ms chunk #3<br/>(silence detected)
       Bus->>CmdListener: (delivers event)
       Bus->>SoundListener: (delivers event)

       CmdListener->>Bus: CommandAudioSegmentReadyEvent<br/>audio_bytes, sample_rate=16000
       SoundListener->>Bus: ProcessAudioChunkForSoundRecognitionEvent<br/>audio_bytes, sample_rate=16000

       Bus->>STT: (delivers CommandAudioSegmentReadyEvent)
       Bus->>SoundSvc: (delivers ProcessAudioChunkForSoundRecognitionEvent)

       STT->>Bus: CommandTextRecognizedEvent<br/>text="click", engine="vosk"
       SoundSvc->>Bus: (no match for sound recognition)

       Bus->>Parser: (delivers CommandTextRecognizedEvent)
       Parser->>Bus: AutomationCommandParsedEvent<br/>command=ExactMatchCommand("click")

       Bus->>Automation: (delivers AutomationCommandParsedEvent)
       Automation->>Automation: pyautogui.click()<br/>🖱️ Mouse click executed!

This architecture makes the system:

- **Flexible**: Add new services without modifying existing ones
- **Testable**: Test each service in isolation

Next Steps
======================

Now that you know the basics, you can dive into the detailed documentation:

- :doc:`Audio Processing <audio_processing>`: How audio is captured and converted to text
- :doc:`Command Parsing <command_parsing>`: How text becomes structured commands
- :doc:`Command Execution <command_execution>`: How commands trigger actions
- :doc:`Dictation System <dictation_system>`: How text becomes dictation outputs
- :doc:`Infrastructure <infrastructure>`: How the different services communicate
