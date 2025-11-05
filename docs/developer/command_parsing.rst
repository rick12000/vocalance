Command Parsing & Prediction
##############################

This page explains how Vocalance transforms recognized text and sound events into structured commands, with predictive execution and duplicate prevention.

System Overview
================

After recognition services produce text and sound events (see :doc:`speech_and_sound_recognition`), those events flow into the ``CentralizedCommandParser``, which determines what action to take.

.. mermaid::

   flowchart TD
       A[CommandTextRecognizedEvent<br/>from Vosk] --> B[CentralizedCommandParser]
       C[CustomSoundRecognizedEvent<br/>from SoundService] --> B

       B --> E{Sound Mapping?}
       E -->|Yes| F[Map to Command Text]
       E -->|No| G[Use Text Directly]

       F --> H[Parse Text]
       G --> H

       H --> I{Match Found?}
       I -->|Dictation| J[DictationCommandParsedEvent]
       I -->|Mark| K[MarkCommandParsedEvent]
       I -->|Grid| L[GridCommandParsedEvent]
       I -->|Automation| M[AutomationCommandParsedEvent]
       I -->|No Match| N[CommandNoMatchEvent]

       style B fill:#e1f5ff
       style H fill:#fff4e1

Events from voice recognition and sound detection flow to the parser. If a sound has been mapped to a command phrase, that mapping is applied. The text is then parsed to identify which command type it matches. When a match is found, a typed command event is published for downstream services.


The Parsing Flow
================

Text enters the parser normalized: lowercase and whitespace-trimmed. The parser runs a series of pattern-matching functions to determine what command type the text represents. It stops at the first successful match:

.. code-block:: python

   async def _parse_text(self, text: str) -> ParseResultType:
       normalized_text = text.lower().strip()

       parsers = [
           self._parse_dictation_commands,
           self._parse_mark_commands,
           self._parse_grid_commands,
           self._parse_automation_commands,
           self._parse_mark_execute_fallback,
       ]

       for parser in parsers:
           result = await parser(normalized_text)
           if not isinstance(result, NoMatchResult):
               return result

       return NoMatchResult()

Each parser checks specific patterns. If a parser returns a command object, parsing is complete. If all parsers return no match, a ``CommandNoMatchEvent`` is published. If a parser succeeds, the resulting command object is published as a typed event for execution.

Command Types
=============

Dictation Commands
------------------

Dictation commands enter text-capture mode. Once activated, the system stops interpreting voice as commands and instead transcribes everything you say:

.. code-block:: python

   def _parse_dictation_commands(self, normalized_text: str) -> ParseResultType:
       if normalized_text == "start dictation":
           return DictationStartCommand()

       if normalized_text == "stop dictation":
           return DictationStopCommand()

       if normalized_text == "dictate type":
           return DictationTypeCommand()

       if normalized_text == "dictate smart":
           return DictationSmartStartCommand()

       if normalized_text == "dictate visual":
           return DictationVisualStartCommand()

       return NoMatchResult()

Each trigger is a simple text match. Saying one of these phrases switches your input mode entirely.

Mark Commands
-------------

Marks let you save and recall screen positions. A mark is a named position you can jump to later:

.. code-block:: python

   def _parse_mark_commands(self, normalized_text: str) -> ParseResultType:
       words = normalized_text.split()

       # "mark button" → create a mark labeled "button"
       if words[0] == "mark" and len(words) == 2:
           label = words[1]
           x, y = pyautogui.position()
           return MarkCreateCommand(label=label, x=float(x), y=float(y))

       # "delete mark button" → remove the "button" mark
       if normalized_text.startswith("delete mark "):
           label = normalized_text[len("delete mark "):].strip()
           return MarkDeleteCommand(label=label)

       # "show marks" → display all saved marks
       if normalized_text in self._mark_visualize_phrases:
           return MarkVisualizeCommand()

       # "reset marks" → clear all marks
       if normalized_text in self._mark_reset_phrases:
           return MarkResetCommand()

       return NoMatchResult()

The mark parser recognizes four operations: creating a mark at your current position, deleting a specific mark, visualizing all marks, and clearing all marks.

Grid Commands
-------------

The grid system displays a clickable overlay divided into numbered cells. Saying a number selects that cell:

.. code-block:: python

   async def _parse_grid_commands(self, normalized_text: str) -> ParseResultType:
       # "show grid" → display the grid
       if normalized_text == "show grid":
           return GridShowCommand(num_rects=None)

       # "show grid 20" → display grid with 20 cells
       if normalized_text.startswith("show grid "):
           after_trigger = normalized_text[len("show grid "):].strip()
           parsed_num = parse_number(text=after_trigger)
           if parsed_num is not None and parsed_num > 0:
               return GridShowCommand(num_rects=parsed_num)

       # "5" → select cell 5 (if not part of an automation command)
       if not is_automation_prefix(normalized_text):
           parsed_num = parse_number(text=normalized_text)
           if parsed_num is not None and parsed_num > 0:
               return GridSelectCommand(selected_number=parsed_num)

       return NoMatchResult()

The grid parser checks for display commands and cell selection. A bare number like "5" could be ambiguous—it might be the start of "5 press right" (press right 5 times), so the parser checks whether the full text is an automation command before treating it as a grid selection.

Automation Commands
-------------------

Automation commands perform keyboard and mouse actions. They come in two forms:

**Exact Match**

A complete command phrase that maps to a fixed action:

.. code-block:: python

   async def _parse_automation_commands(self, normalized_text: str) -> ParseResultType:
       action_map = await self._action_map_provider.get_action_map()

       # Try exact match
       if normalized_text in action_map:
           command_data = action_map[normalized_text]
           return ExactMatchCommand(
               command_key=normalized_text,
               action_type=command_data.action_type,
               action_value=command_data.action_value,
               is_custom=command_data.is_custom,
               short_description=command_data.short_description,
               long_description=command_data.long_description,
           )

The action map is a dictionary of commands loaded from storage. If the full text matches an entry, it returns an exact match command.

**Parameterized**

A command that accepts a parameter—typically a repeat count:

.. code-block:: python

   # Try parameterized: command + number
   words = normalized_text.split()
   for i in range(len(words) - 1, 0, -1):  # Try longest match first
       potential_command = " ".join(words[:i])

       if potential_command in action_map:
           remaining_words = words[i:]
           if len(remaining_words) == 1:
               count = parse_number(text=remaining_words[0])
               if count is not None and count > 0:
                   command_data = action_map[potential_command]
                   return ParameterizedCommand(
                       command_key=potential_command,
                       action_type=command_data.action_type,
                       action_value=command_data.action_value,
                       count=count,
                       is_custom=command_data.is_custom,
                   )
           break

       return NoMatchResult()

This searches backwards through word boundaries. For "3 press right": it tries "3 press right" (no), then "3 press" (no), then checks "press right" as the command with "3" as the count. The backwards approach prioritizes longer matches, so if both "press" and "press right" are commands, "press right" wins.

Mark Execute Fallback
---------------------

Single-word inputs that don't match other patterns are treated as mark names. This enables saying a mark name to jump to it:

.. code-block:: python

   def _parse_mark_execute_fallback(self, normalized_text: str) -> ParseResultType:
       words = normalized_text.split()

       if len(words) == 1:
           return MarkExecuteCommand(label=normalized_text)

       return NoMatchResult()

This runs last, after all other parsers. Single words that didn't match any explicit pattern become mark lookups.

From Command to Event
=====================

Once a command is successfully parsed, it's published as a typed event:

.. code-block:: python

   command_type_map = {
       DictationStartCommand: DictationCommandParsedEvent,
       ExactMatchCommand: AutomationCommandParsedEvent,
       ParameterizedCommand: AutomationCommandParsedEvent,
       MarkCreateCommand: MarkCommandParsedEvent,
       MarkExecuteCommand: MarkCommandParsedEvent,
       GridShowCommand: GridCommandParsedEvent,
       GridSelectCommand: GridCommandParsedEvent,
       # ... more mappings
   }

Each command type has a corresponding event. Downstream services listen for these events and execute the actions. The parser's role is purely to recognize patterns and create command objects—execution happens elsewhere.

The Markov Prediction System
=============================

The ``MarkovCommandService`` predicts your next command based on patterns in your recent command history. By recognizing what you're likely to do next, it can execute commands before you finish speaking.

The system has two distinct phases: training, which learns patterns from history, and inference, which uses those patterns to make real-time predictions.

Training Phase: Building the Prediction Model
----------------------------------------------

At startup, the system analyzes your command history to build a statistical model. This model captures patterns in what commands typically follow each other.

**Multi-order analysis**: The system maintains three separate Markov chains, each looking at different time windows:

- **4th-order chains**: 14 days, up to 200 commands (long-term patterns)
- **3rd-order chains**: 7 days, up to 150 commands (medium-term patterns)
- **2nd-order chains**: 3 days, up to 100 commands (recent patterns)

Longer windows reveal stable behavioral trends. Shorter windows adapt quickly if your workflow changes temporarily.

**Building the statistics**: The model extracts command sequences from your history and counts transitions:

.. mermaid::

   graph LR
       A[Command History] --> B[show grid<br/>c5<br/>show grid<br/>c7<br/>show grid<br/>c5]
       B --> C[Extract Sequences]
       C --> D[show grid → c5: 2<br/>show grid → c7: 1]
       D --> E[Probabilities:<br/>c5: 67%<br/>c7: 33%]

       style E fill:#e8f5e9

For 2nd-order chains: "After command A, how often does B follow?" For 3rd and 4th-order: "After this sequence of commands, what comes next?" This builds a probabilistic model that knows your patterns.

Inference Phase: Making Real-Time Predictions
----------------------------------------------

Once trained, the model is ready to predict what you'll do next and execute commands before you finish speaking.

**The prediction flow**: When you start speaking, the predictor analyzes your recent commands and makes a guess:

.. mermaid::

   sequenceDiagram
       participant U as User
       participant Audio as AudioListener
       participant Markov as MarkovPredictor
       participant Parser as CommandParser
       participant Exec as ExecutionService

       Note over U,Exec: Established pattern: show grid → c5

       U->>Audio: Starts speaking
       Audio->>Markov: Audio detected
       Markov->>Markov: Check history:<br/>Last: show grid
       Markov->>Markov: Predict: c5 (67% confident)
       Markov->>Parser: Predicted command
       Parser->>Exec: Execute c5

       Note over U,Exec: Cell selected before speech finishes!

       U->>Parser: c5 (recognized text)
       Parser->>Parser: Already executed

The predictor executes immediately. When speech recognition finishes, the system compares the actual command to the prediction. If they match, the command runs once (not twice). If they differ, the actual command is executed.

**Pattern specificity**: The system uses the most specific pattern available:

.. code-block:: python

   async def _predict_next_command(self):
       for order in [4, 3, 2]:
           context = tuple(self._command_history[-(order-1):])

           if context in self._transition_counts[order]:
               counts = self._transition_counts[order][context]
               total = sum(counts.values())

               next_cmd, count = counts.most_common(1)[0]
               confidence = count / total

               if confidence >= self._min_confidence:
                   return next_cmd, confidence

       return None, 0.0

The algorithm tries 4-command patterns first, then 3-command, then 2-command. It stops at the first pattern with sufficient confidence. This prioritizes longer, more specific patterns.

Adapting Through Feedback
---------------------------

When predictions are wrong, the system learns and adjusts:

.. code-block:: python

   async def _handle_prediction_feedback(self, event):
       if not event.correct:
           self._cooldown_remaining += 3  # Skip next 3 predictions
           logger.info(f"Prediction incorrect")

A wrong prediction increases a cooldown counter. While counting down, predictions are skipped entirely. This prevents a series of bad guesses from degrading the experience. As correct predictions accumulate, the cooldown decrements, and predictions resume.

Critical Exception: Dictation Mode
-----------------------------------

Prediction is automatically disabled when dictation mode is active. This requires special handling because of how Markov chains work.

**The problem**: Dictation mode works by setting a flag that tells the parser to treat all input as text to transcribe, not as commands. However, to exit dictation, you must say "stop dictation"—the only command that works during dictation. This creates a deterministic transition in the Markov model.

After enough sessions of activating and deactivating dictation, the model learns:

.. code-block:: text

   "start dictation" → "stop dictation": 100% probability

Why? Because the only way to get back to command mode is to say "stop dictation". The model sees this pattern repeatedly and assigns it maximum confidence.

**The consequence**: If predictions were enabled during dictation, this is what would happen:

1. You say "start dictation" (activates text transcription mode)
2. The Markov predictor, with 100% confidence, immediately predicts "stop dictation"
3. The predictor executes "stop dictation" before you speak any text
4. Dictation mode is instantly terminated
5. You never get to dictate anything

**The solution**: Prediction is disabled the moment dictation starts:

.. code-block:: python

   async def _handle_audio_detected(self, event):
       if self._dictation_active:
           return  # Skip prediction during dictation

       # ... normal prediction logic

This ensures that whatever you say during dictation is transcribed as text, not interpreted as the predictable "stop dictation" command. Once you exit dictation manually, predictions resume.

What Happens Next
==================

Parsed commands are published as events and routed to specialized services:

- **AutomationCommandParsedEvent** → AutomationService
- **MarkCommandParsedEvent** → MarkService
- **GridCommandParsedEvent** → GridService
- **DictationCommandParsedEvent** → DictationCoordinator

These execution services are covered in :doc:`command_execution_services`.
