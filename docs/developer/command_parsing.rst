Command Parsing
================

After the audio pipeline captures speech and produces text via STT, sound recognition, or Markov prediction, that text must be converted into structured commands. The ``CentralizedCommandParser`` acts as the routing hub: it receives text from multiple sources, identifies the command type, handles deduplication, and publishes strongly-typed command events for execution services to consume.

Parser Architecture
-------------------

The ``CentralizedCommandParser`` sits at the intersection of recognition and execution. It subscribes to events from the audio processing layer and publishes events that trigger command execution.

.. mermaid::

   flowchart LR
       subgraph Input Sources
           A[Vosk STT]
           B[Sound Recognition]
           C[Markov Predictor]
       end

       subgraph CentralizedCommandParser
           D[Event Handler]
           E[Deduplication Layer]
           F[Hierarchical Parser]
           G[Command History]
       end

       subgraph Output Events
           H[DictationCommandParsedEvent]
           I[AutomationCommandParsedEvent]
           J[MarkCommandParsedEvent]
           K[GridCommandParsedEvent]
           L[SoundCommandParsedEvent]
       end

       A -->|CommandTextRecognizedEvent| D
       B -->|CustomSoundRecognizedEvent| D
       C -->|MarkovPredictionEvent| D

       D --> E
       E --> F
       F --> G
       G --> H
       G --> I
       G --> J
       G --> K
       G --> L

**Event Subscriptions**

The parser subscribes to five event types:

- ``CommandTextRecognizedEvent``: Complete utterances from Vosk STT
- ``CustomSoundRecognizedEvent``: User-trained sounds mapped to command text
- ``MarkovPredictionEvent``: Predicted commands from pattern analysis
- ``CommandMappingsUpdatedEvent``: Dynamic command map updates
- ``ProcessCommandPhraseEvent``: Explicit parsing requests from other services

When any of these events fire, the parser routes them through deduplication and hierarchical parsing to produce a structured command object.

Intelligent Deduplication
-------------------------

Since commands can arrive from multiple sources simultaneously (especially Markov predictions racing against STT), the parser implements two deduplication strategies to prevent double-execution.

Time-Based Text Deduplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identical text from any source within a 1-second window is suppressed:

.. code-block:: python

   # Prevents "press enter" → "press enter" within 1 second
   if text == self._last_text and current_time - self._last_text_time < 1.0:
       return  # Skip duplicate

This handles rapid repeated utterances and prevents acoustic echo from triggering the same command twice.

Markov-STT Coordination
~~~~~~~~~~~~~~~~~~~~~~~~

The Markov predictor fires immediately when speech is detected (see Audio Processing documentation), often 200-400ms before STT completes. The parser tracks these predictions in ``_recent_predictions`` and checks incoming STT results against them:

.. mermaid::

   sequenceDiagram
       participant User
       participant Markov
       participant Parser
       participant STT
       participant Executor

       User->>Markov: Speech detected (AudioDetectedEvent)
       Markov->>Parser: MarkovPredictionEvent("press enter")
       Parser->>Parser: Store in _recent_predictions
       Parser->>Executor: Execute predicted command

       Note over STT: 200-400ms later...
       STT->>Parser: CommandTextRecognizedEvent("press enter")
       Parser->>Parser: Check _recent_predictions
       alt Prediction matches STT
           Parser->>Markov: Send positive feedback
           Parser->>Parser: Skip execution (already done)
       else Prediction differs
           Parser->>Markov: Send negative feedback + cooldown
           Parser->>Executor: Execute STT command
       end

When STT matches a recent prediction, execution is skipped and positive feedback trains the Markov model. When they differ, both commands execute but the predictor enters cooldown mode to reduce false positives.

Hierarchical Command Parsing
-----------------------------

Text flows through a strict priority cascade where each parser attempts to match, returning immediately on success. This ensures specific command types (dictation triggers, system commands) take precedence over the general automation command space.

**Parsing Priority**

1. **Dictation Commands**: Fixed triggers ("start dictation", "stop dictation", "smart dictation", "type dictation")
2. **Mark Commands**: Visual navigation ("mark [label]", "go [label]", "show marks", "hide marks", etc.)
3. **Grid Commands**: Numerical UI selection ("show grid", "show grid five", "[number]" when grid active)
4. **Automation Commands**: User-defined and default commands from action map
5. **Mark Execute Fallback**: Single-word utterances treated as mark labels

Command Type Details
~~~~~~~~~~~~~~~~~~~~

**Dictation Commands**

Simple exact-match triggers that control dictation mode. Example matches: ``"start dictation"`` → ``DictationStartCommand()``

**Mark Commands**

Pattern-based parsing for visual navigation. The parser extracts labels from phrases like "mark button" → ``MarkCreateCommand(label="button")``

**Grid Commands**

Numerical grid overlays for mouse-free UI interaction. Handles "show grid" and grid cell selection.

**Automation Commands**

The most complex category, automation commands use the ``CommandActionMapProvider`` to merge default commands with user customizations:

.. code-block:: python

   # Exact match: "press enter" → ExactMatchCommand
   if normalized_text in action_map:
       return ExactMatchCommand(
           command_key=normalized_text,
           action_type="key",
           action_value="enter"
       )

   # Parameterized: "scroll down three" → ParameterizedCommand(count=3)
   # Parser walks backwards through words to find longest command prefix
   for i in range(len(words) - 1, 0, -1):
       potential_command = " ".join(words[:i])
       if potential_command in action_map:
           remaining_words = words[i:]
           count = parse_number(remaining_words[0])
           if count:
               return ParameterizedCommand(
                   command_key=potential_command,
                   count=count,
                   ...
               )

This handles commands like "scroll down three" by finding "scroll down" in the action map and parsing "three" as a repeat count.

**Mark Execute Fallback**

Single-word utterances that don't match any other category are interpreted as mark labels: ``"button"`` → ``MarkExecuteCommand(label="button")``. This provides ultra-fast navigation: create marks with "mark button", then jump back anytime by saying just "button".

From Parse to Execution
------------------------

Once text is successfully parsed into a command object, the parser publishes a domain-specific event and records the command to history (for Markov training).

**Event Publishing**

Each command type routes to its corresponding event:

- ``DictationStartCommand`` → ``DictationCommandParsedEvent``
- ``ExactMatchCommand`` → ``AutomationCommandParsedEvent``
- ``MarkCreateCommand`` → ``MarkCommandParsedEvent``
- ``GridShowCommand`` → ``GridCommandParsedEvent``
- ``SoundTrainCommand`` → ``SoundCommandParsedEvent``

Execution services subscribe to these events (see Command Execution documentation) and perform the actual automation actions.

**Command History & Markov Training**

Only successfully parsed commands are recorded to the ``CommandHistoryManager``:

.. code-block:: python

   if isinstance(parse_result, BaseCommand):
       # Valid command - record for Markov training
       await self._history_manager.record_command(command=text, source=source)
       await self._publish_command_event(parse_result, source)

   elif isinstance(parse_result, NoMatchResult):
       # No match - publish event but don't record
       await self._event_bus.publish(CommandNoMatchEvent(...))
