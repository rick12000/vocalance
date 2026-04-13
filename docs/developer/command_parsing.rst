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
       I -->|No Match| N[NoMatchResult]

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

Each parser checks specific patterns. If a parser returns a command object, parsing is complete. If all parsers return no match, a ``NoMatchResult`` is returned. If a parser succeeds, the resulting command object is published as a typed event for execution.

Command Types
=============

Dictation Commands
------------------

Dictation commands enter text-capture mode. Once activated, the system stops interpreting voice as commands and instead transcribes everything you say (except the configured stop phrase, which ends the session).

Phrases are **exact matches** on lowercased, trimmed text. They come from ``DictationConfig`` and are cached on the parser at startup—for example ``start_trigger`` (default ``green``), ``stop_trigger`` (``amber``), ``type_trigger``, ``smart_start_trigger`` (``smart green``), ``visual_start_trigger``, ``hidden_start_trigger``, and ``amend_start_trigger`` (``amend``).

.. code-block:: python

   def _parse_dictation_commands(self, normalized_text: str) -> ParseResultType:
       if normalized_text == self._dictation_start_trigger:
           return DictationStartCommand()
       if normalized_text == self._dictation_stop_trigger:
           return DictationStopCommand()
       if normalized_text == self._dictation_type_trigger:
           return DictationTypeCommand()
       if normalized_text == self._dictation_smart_trigger:
           return DictationSmartStartCommand()
       if normalized_text == self._dictation_visual_trigger:
           return DictationVisualStartCommand()
       if normalized_text == self._dictation_hidden_trigger:
           return DictationHiddenStartCommand()
       if normalized_text == self._dictation_amend_trigger:
           return DictationAmendStartCommand()
       return NoMatchResult()

Defaults match ``vocalance.app.config.app_config.DictationConfig``; the parser stores lowercased trigger strings when it initializes (or when its config cache is rebuilt).

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

The grid system shows a full-screen overlay of numbered cells. Configured phrases (from ``GridConfig``) open the grid in **click**, **hover**, or **drag** mode; a bare number then selects a cell. The parser tries each show phrase in order via a shared helper:

.. code-block:: python

   from typing import Union

   def _parse_grid_show_for_phrase(
       self, normalized_text: str, phrase: str, click_mode: str
   ) -> Union[GridShowCommand, ErrorResult, None]:
       if not normalized_text.startswith(phrase):
           return None
       if normalized_text == phrase:
           return GridShowCommand(num_rects=None, click_mode=click_mode)
       after_trigger = normalized_text[len(phrase) :].strip()
       if not after_trigger:
           return None
       parsed_num = parse_number(text=after_trigger)
       if parsed_num is not None and parsed_num > 0:
           return GridShowCommand(num_rects=parsed_num, click_mode=click_mode)
       return ErrorResult(error_message=f"Invalid number of rectangles: '{after_trigger}'")

   # In _parse_grid_commands: for (phrase, mode) in ("go", "click"), ("hover", "hover"), ("move", "drag"): ...

The default phrases are ``go``, ``hover``, and ``move`` (the latter opens **drag** mode), each optionally followed by a cell count (e.g. ``go 100``). Grid parsing runs **before** automation parsing, so an exact show phrase is always a grid command, not an automation match.

For cell selection, a bare number becomes ``GridSelectCommand`` only when the full normalized text is **not** an automation prefix—otherwise ``5 press right`` would be parsed as automation rather than grid cell ``5``.

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

The ``MarkovCommandService`` predicts your next command based on patterns in your recent command history. By recognizing what you're likely to do next, it can execute commands before you finish speaking. This provides ultra-low latency execution (~30-50ms) compared to waiting for STT (~400-600ms).

The system has two distinct phases: training, which learns patterns from history, and inference, which uses those patterns to make real-time predictions.

Training Phase: Building the Prediction Model
----------------------------------------------

At startup, the system analyzes your command history to build a statistical model. This model captures patterns in what commands typically follow each other.

**Multi-order analysis**: The system maintains three separate Markov chains (orders 2, 3, 4), each with configurable training windows:

- **4th-order chains**: Up to 1500 commands, up to 60 days (captures long-term patterns)
- **3rd-order chains**: Up to 1000 commands, up to 21 days (medium-term patterns)
- **2nd-order chains**: Up to 500 commands, up to 7 days (recent patterns)

Each order requires a minimum transition frequency before a prediction is considered valid (configurable: order 2→2, order 3→5, order 4→10). Longer windows reveal stable behavioral trends. Shorter windows adapt quickly if your workflow changes temporarily.

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
       participant Dedup as Deduplicator
       participant Exec as ExecutionService

       Note over U,Exec: Pattern: show grid → c5 (85% confidence)

       U->>Audio: Starts speaking
       Audio->>Markov: AudioDetectedEvent
       Markov->>Markov: Check: enabled & history >= 2
       Markov->>Markov: Predict: c5 (85% confident)
       Markov->>Parser: MarkovPredictionEvent

       par Parallel Processing
           Parser->>Dedup: Record: c5 (markov source)
           Dedup->>Exec: Execute c5 (MARKOV)
       end

       Note over U,Exec: Command executes immediately!

       rect rgb(200, 220, 255)
           Note over U,Exec: ~30-50ms: Markov execution
           Note over U,Exec: ~400-600ms: STT processing
       end

       U->>Parser: c5 (STT recognized)
       Parser->>Dedup: Check: should_deduplicate(c5, stt)?
       Dedup-->>Parser: YES - already executed by Markov
       Parser->>Markov: Feedback: prediction CORRECT
       Markov->>Markov: Confirm prediction

If prediction matches actual command, deduplication prevents duplicate execution. If they differ, the actual command executes (prediction was wrong). Feedback is always sent to update command history and manage cooldown.

**Pattern specificity and backoff**: The system uses the most specific pattern available through a backoff strategy:

.. code-block:: python

   def _predict_next_command(self) -> Optional[Tuple[str, float, int]]:
       for order in range(max_order, min_order - 1, -1):  # 4 → 3 → 2
           if len(self._command_history) < order:
               continue

           context = tuple(list(self._command_history)[-order:])

           if context not in self._transition_counts[order]:
               continue

           transitions = self._transition_counts[order][context]
           total_count = sum(transitions.values())

           min_freq = self._markov_config.min_command_frequency.get(order, 2)
           valid_transitions = {
               cmd: count for cmd, count in transitions.items()
               if count >= min_freq
           }

           if not valid_transitions:
               continue

           most_common_cmd, count = max(valid_transitions.items(), key=lambda x: x[1])
           confidence = count / total_count

           if confidence >= confidence_threshold:
               return (most_common_cmd, confidence, order)

       return None

The algorithm tries 4-command patterns first, then 3-command, then 2-command. It stops at the first pattern with sufficient confidence **and** transition frequency. This prioritizes longer, more specific patterns while ensuring transitions are statistically significant.

Adapting Through Feedback
---------------------------

When predictions are wrong, the system learns and adjusts. All commands (correct predictions, incorrect predictions, and STT/sound commands) update the in-memory history:

.. code-block:: python

   async def _handle_prediction_feedback(self, event):
       actual_command = event.actual_command
       was_correct = event.was_correct

       # Handle incorrect predictions by entering cooldown
       if event.predicted_command != actual_command and was_correct is False:
           self._cooldown_remaining = self._config.incorrect_prediction_cooldown
           logger.warning(f"Markov prediction incorrect...")
       elif event.predicted_command == actual_command and was_correct:
           logger.info(f"Markov prediction correct...")

       # Decrement cooldown on every command execution
       if self._cooldown_remaining > 0:
           self._cooldown_remaining -= 1

       # Always update command history with actual command
       self._command_history.append(actual_command)

An incorrect prediction enters a configurable cooldown (default: skip 1 command). While in cooldown, predictions are skipped entirely. This prevents a series of bad guesses from degrading the experience. The cooldown decrements on every executed command, and predictions resume when the counter reaches zero. Crucially, only **actual** commands (from STT/sound) update the training history, not predictions.


Critical Exception: Dictation Mode
-----------------------------------

Prediction is automatically disabled when dictation mode is active. This requires special handling because of how Markov chains work.

**The problem**: Dictation mode works by setting a flag that tells the parser to treat all input as text to transcribe, not as commands. To exit, you must say your configured **stop** phrase—the only command path that still applies—so the Markov model sees a near-deterministic start → stop pair.

**The consequence**: If predictions stayed enabled during dictation, the model could immediately predict the stop phrase right after the start phrase, execute it, and end dictation before you speak any content.

**The solution**: Prediction is disabled the moment dictation starts:

.. code-block:: python

   async def _handle_audio_detected_fast_track(self, event):
       if self._dictation_active:
           return  # Skip prediction during dictation

       # ... normal prediction logic

This ensures that whatever you say during dictation is transcribed as text, not replaced by a predicted stop phrase. Once you exit dictation manually, predictions resume.

What Happens Next
==================

Parsed commands are published as events and routed to specialized services:

- **AutomationCommandParsedEvent** → AutomationService
- **MarkCommandParsedEvent** → MarkService
- **GridCommandParsedEvent** → GridService
- **DictationCommandParsedEvent** → DictationCoordinator

These execution services are covered in :doc:`command_execution_services`.
