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
       A[Microphone Input] --> B[AudioRecorder VAD]
       B --> C[Speech Detection]
       C --> D[AudioService]

       D --> E{Current Mode?}
       E -->|Command Mode| F[Command Recorder Active]
       E -->|Dictation Mode| G[Both Recorders Active]

       F --> H[CommandAudioSegmentReady Event]
       G --> I[DictationAudioSegmentReady Event]
       G --> J[CommandAudioSegmentReady Event<br/>for stop words]

       H --> K[SpeechToTextService]
       I --> K
       J --> K

       K --> L{Engine Selection}
       L -->|Command Audio| M[Vosk Engine]
       L -->|Dictation Audio| N[Whisper Engine]
       L -->|Empty Result| O[SoundService]

       M --> P[CommandTextRecognized Event]
       N --> Q[DictationTextRecognized Event]
       O --> R[CustomSoundRecognized Event]

       P --> S[CentralizedCommandParser]
       Q --> T[DictationCoordinator]
       R --> S

       S --> U{Command Type?}
       U -->|Automation| V[AutomationService]
       U -->|Mark| W[MarkService]
       U -->|Grid| X[GridService]
       U -->|Dictation| T

       V --> Y[pyautogui execution]
       W --> Z[Mouse jump]
       X --> AA[Grid UI display]
       T --> AB[Text output]

The general pattern is:

- The user speaks into the microphone
- Vocalance captures the audio and converts it to text
- The text is parsed into a command
- The command is executed to carry out some action on the computer

Event-Driven Architecture
===========================

Vocalance uses an event driven architecture. Services don't call each other directly, instead, they communicate by publishing and subscribing to **events** through a central ``EventBus``.

How It Works: A Concrete Example
----------------------------------

Let's trace through exactly what happens when you say "click" into the microphone. We'll follow the event flow shown in the sequence diagram below, connecting each step to the specific events and services.

**1. Audio Capture and Speech Detection**

When you speak "click", the AudioService is already listening through its command recorder. The AudioRecorder uses Voice Activity Detection (VAD) to detect when you start speaking, captures the audio segment, and immediately publishes a ``CommandAudioSegmentReadyEvent`` to the EventBus:

.. code-block:: python

   # AudioService publishes the captured audio
   event = CommandAudioSegmentReadyEvent(
       audio_bytes=audio_bytes,
       sample_rate=16000
   )
   await event_bus.publish(event)

This corresponds to the "CommandAudioSegmentReady Event" shown in the diagram flowing from AudioService to SpeechToTextService.

**2. Speech-to-Text Recognition**

The SpeechToTextService is subscribed to ``CommandAudioSegmentReadyEvent``. It receives the audio and uses the Vosk engine (optimized for fast, offline command recognition) to convert the speech to text:

.. code-block:: python

   # SpeechToTextService receives the audio event
   @event_bus.subscribe(CommandAudioSegmentReadyEvent)
   async def _handle_command_audio_segment(self, event):
       # Use Vosk for fast command recognition
       text = await self.vosk_engine.recognize(event.audio_bytes)

       # Publish the recognized text
       text_event = CommandTextRecognizedEvent(
           text=text,  # "click"
           engine="vosk",
           mode="command"
       )
       await self.event_bus.publish(text_event)

The text "click" is now published as a ``CommandTextRecognizedEvent``, which matches the event flowing from SpeechToTextService to CentralizedCommandParser in the diagram.

**3. Command Parsing and Classification**

The CentralizedCommandParser receives the ``CommandTextRecognizedEvent`` and parses the text through its hierarchical command system. It checks if "click" matches any known automation commands by looking up the action map:

.. code-block:: python

   # CentralizedCommandParser handles text recognition
   @event_bus.subscribe(CommandTextRecognizedEvent)
   async def _handle_command_text_recognized(self, event):
       # Parse "click" through the command hierarchy
       command = await self._parse_text(event.text)

       # For "click", this returns an ExactMatchCommand
       # with action_type="click" and action_value=""

       # Publish the parsed command
       parsed_event = AutomationCommandParsedEvent(
           command=command,
           source="stt"
       )
       await self.event_bus.publish(parsed_event)

This creates an ``AutomationCommandParsedEvent`` that flows to the AutomationService, as shown in the diagram.

**4. Command Execution**

Finally, the AutomationService receives the ``AutomationCommandParsedEvent`` and executes the command using pyautogui:

.. code-block:: python

   # AutomationService executes the parsed command
   @event_bus.subscribe(AutomationCommandParsedEvent)
   async def _handle_automation_command(self, event):
       command = event.command

       # For a "click" command, execute mouse click
       if command.action_type == ActionType.CLICK:
           success = await self._execute_command(
               ActionType.CLICK,
               command.action_value,
               count=getattr(command, 'count', 1)
           )

       # Publish execution status
       status_event = CommandExecutedStatusEvent(
           command=command.__dict__,
           success=success,
           source=event.source
       )
       await event_bus.publish(status_event)

The mouse click happens here! This completes the full event flow from microphone input to computer action.

The Event Flow
---------------

Notice the pattern: each service does its job and publishes an event when done. Other services that care about that event will react to it. No service knows about the others directly.

.. mermaid::

   sequenceDiagram
       participant Audio as AudioService
       participant Bus as EventBus
       participant STT as SpeechToTextService
       participant Parser as CentralizedCommandParser
       participant Automation as AutomationService

       Note over Audio,Automation: User says "click"
       Audio->>Bus: CommandAudioSegmentReadyEvent<br/>audio_bytes, sample_rate=16000
       Bus->>STT: (delivers event)
       STT->>Bus: CommandTextRecognizedEvent<br/>text="click", engine="vosk"
       Bus->>Parser: (delivers event)
       Parser->>Bus: AutomationCommandParsedEvent<br/>command=ExactMatchCommand("click")
       Bus->>Automation: (delivers event)
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
