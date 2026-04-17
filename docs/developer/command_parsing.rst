Command Parsing
#################

This page explains how Vocalance transforms recognized text and sound events into structured commands, with duplicate prevention across STT and sound sources.

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

       F --> H[parse_full_command_text]
       G --> H

       H --> I{Match Found?}
       I -->|System Control| P[SystemControlCommandParsedEvent]
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

Text enters the parser normalized: lowercase and whitespace-trimmed. The ``CentralizedCommandParser`` applies rate limiting (minimum command interval) and pause rules. If the system is paused, only the ``ResumeCommand`` is permitted.

The actual parsing is delegated to a pure function pipeline in ``parse_full_command_text``, which runs a series of pattern-matching functions to determine what command type the text represents. It stops at the first successful match:

.. code-block:: python

   def parse_full_command_text(
       normalized_text: str, triggers: CommandParserTriggers, action_map: Dict[str, AutomationCommand]
   ) -> ParseResultType:
       if not normalized_text:
           return NoMatchResult()

       steps: List[ParseResultType] = [
           parse_system_control(normalized_text),
           parse_dictation(normalized_text, triggers),
           parse_mark_commands(normalized_text, triggers),
           parse_grid_commands(normalized_text, triggers, action_map),
           parse_automation_commands(normalized_text, action_map),
           parse_mark_execute_fallback(normalized_text),
       ]
       for result in steps:
           if not isinstance(result, NoMatchResult):
               return result
       return NoMatchResult()

Each parser checks specific patterns. If a parser returns a command object, parsing is complete. If all parsers return no match, a ``NoMatchResult`` is returned. If a parser succeeds, the resulting command object is published as a typed event for execution.

Command Types
=============

System Control Commands
-----------------------

System control commands handle global state like pausing and resuming the application:

.. code-block:: python

   def parse_system_control(normalized_text: str) -> ParseResultType:
       if normalized_text == "pause":
           return PauseCommand()
       if normalized_text == "resume":
           return ResumeCommand()
       return NoMatchResult()

Dictation Commands
------------------

Dictation commands enter text-capture mode. Once activated, the system stops interpreting voice as commands and instead transcribes everything you say (except the configured stop phrase, which ends the session).

Phrases are **exact matches** on lowercased, trimmed text. They come from ``DictationConfig`` and are cached in a ``CommandParserTriggers`` object at startup—for example ``start_trigger`` (default ``green``), ``stop_trigger`` (``amber``), ``type_trigger``, ``smart_start_trigger`` (``smart green``), ``visual_start_trigger``, ``hidden_start_trigger``, and ``amend_start_trigger`` (``amend``).

.. code-block:: python

   def parse_dictation(normalized_text: str, triggers: CommandParserTriggers) -> ParseResultType:
       if normalized_text == triggers.dictation_start_trigger:
           return DictationStartCommand()
       if normalized_text == triggers.dictation_stop_trigger:
           return DictationStopCommand()
       if normalized_text == triggers.dictation_type_trigger:
           return DictationTypeCommand()
       if normalized_text == triggers.dictation_smart_trigger:
           return DictationSmartStartCommand()
       if normalized_text == triggers.dictation_visual_trigger:
           return DictationVisualStartCommand()
       if normalized_text == triggers.dictation_hidden_trigger:
           return DictationHiddenStartCommand()
       if normalized_text == triggers.dictation_amend_trigger:
           return DictationAmendStartCommand()
       return NoMatchResult()

Mark Commands
-------------

Marks let you save and recall screen positions. A mark is a named position you can jump to later:

.. code-block:: python

   def parse_mark_commands(normalized_text: str, triggers: CommandParserTriggers) -> ParseResultType:
       words = normalized_text.split()
       if not words:
           return NoMatchResult()

       if words[0] == triggers.mark_create_prefix and len(words) == 2:
           label = words[1]
           if not label:
               return ErrorResult(error_message="Mark label cannot be empty")
           x, y = pyautogui.position()
           return MarkCreateCommand(label=label, x=float(x), y=float(y))

       if normalized_text.startswith(f"{triggers.mark_delete_prefix} "):
           label_part = normalized_text[len(triggers.mark_delete_prefix) :].strip()
           if label_part and len(label_part.split()) == 1:
               return MarkDeleteCommand(label=label_part)
           return ErrorResult(error_message="Mark delete requires a single word label")

       if normalized_text in triggers.mark_visualize_phrases:
           return MarkVisualizeCommand()
       if normalized_text in triggers.mark_reset_phrases:
           return MarkResetCommand()
       if normalized_text in triggers.mark_cancel_visualize_phrases:
           return MarkVisualizeCancelCommand()

       return NoMatchResult()

The mark parser recognizes four operations: creating a mark at your current position, deleting a specific mark, visualizing all marks, and clearing all marks.

Grid Commands
-------------

The grid system shows a full-screen overlay of numbered cells. Configured phrases (from ``GridConfig``) open the grid in **click**, **hover**, or **drag** mode; a bare number then selects a cell. The parser tries each show phrase in order via a shared helper:

.. code-block:: python

   from typing import Union

   def grid_show_from_phrase(normalized_text: str, phrase: str, click_mode: str) -> Union[GridShowCommand, ErrorResult, None]:
       if not normalized_text.startswith(phrase):
           return None
       if normalized_text == phrase:
           return GridShowCommand(num_rects=None, click_mode=click_mode)
       rest = normalized_text[len(phrase) :].strip()
       if not rest:
           return None
       n = parse_number(text=rest)
       if n is not None and n > 0:
           return GridShowCommand(num_rects=n, click_mode=click_mode)
       return ErrorResult(error_message=f"Invalid number of rectangles: '{rest}'")

   # In parse_grid_commands: for (phrase, mode) in ("go", "click"), ("hover", "hover"), ("move", "drag"): ...

The default phrases are ``go``, ``hover``, and ``move`` (the latter opens **drag** mode), each optionally followed by a cell count (e.g. ``go 100``). Grid parsing runs **before** automation parsing, so an exact show phrase is always a grid command, not an automation match.

For cell selection, a bare number becomes ``GridSelectCommand`` only when the full normalized text is **not** an automation prefix—otherwise ``5 press right`` would be parsed as automation rather than grid cell ``5``.

Automation Commands
-------------------

Automation commands perform keyboard and mouse actions. They come in two forms:

**Exact Match**

A complete command phrase that maps to a fixed action:

.. code-block:: python

   def parse_automation_commands(normalized_text: str, action_map: Dict[str, AutomationCommand]) -> ParseResultType:
       words = normalized_text.split()
       if not words:
           return NoMatchResult()

       if normalized_text in action_map:
           spec = action_map[normalized_text]
           return ExactMatchCommand(
               command_key=normalized_text,
               action_type=spec.action_type,
               action_value=spec.action_value,
               is_custom=spec.is_custom,
               short_description=spec.short_description,
               long_description=spec.long_description,
           )

The action map is a dictionary of commands loaded from storage. If the full text matches an entry, it returns an exact match command.

**Parameterized**

A command that accepts a parameter—typically a repeat count:

.. code-block:: python

       for i in range(len(words) - 1, 0, -1):
           prefix = " ".join(words[:i])
           if prefix not in action_map:
               continue
           tail = words[i:]
           if len(tail) != 1:
               break
           count = parse_number(text=tail[0])
           if count is None or count <= 0:
               break
           spec = action_map[prefix]
           return ParameterizedCommand(
               command_key=prefix,
               action_type=spec.action_type,
               action_value=spec.action_value,
               count=count,
               is_custom=spec.is_custom,
               short_description=spec.short_description,
               long_description=spec.long_description,
           )

       return NoMatchResult()

This searches backwards through word boundaries. For "3 press right": it tries "3 press right" (no), then "3 press" (no), then checks "press right" as the command with "3" as the count. The backwards approach prioritizes longer matches, so if both "press" and "press right" are commands, "press right" wins.

Mark Execute Fallback
---------------------

Single-word inputs that don't match other patterns are treated as mark names. This enables saying a mark name to jump to it:

.. code-block:: python

   def parse_mark_execute_fallback(normalized_text: str) -> ParseResultType:
       words = normalized_text.split()
       if len(words) == 1:
           return MarkExecuteCommand(label=normalized_text)
       return NoMatchResult()

This runs last, after all other parsers. Single words that didn't match any explicit pattern become mark lookups.

From Command to Event
=====================

Once a command is successfully parsed, it's published as a typed event:

.. code-block:: python

   PARSED_EVENT_BY_COMMAND: Dict[Type[BaseCommand], Type[BaseEvent]] = {
       DictationStartCommand: DictationCommandParsedEvent,
       ExactMatchCommand: AutomationCommandParsedEvent,
       ParameterizedCommand: AutomationCommandParsedEvent,
       MarkCreateCommand: MarkCommandParsedEvent,
       MarkExecuteCommand: MarkCommandParsedEvent,
       GridShowCommand: GridCommandParsedEvent,
       GridSelectCommand: GridCommandParsedEvent,
       PauseCommand: SystemControlCommandParsedEvent,
       ResumeCommand: SystemControlCommandParsedEvent,
       # ... more mappings
   }

Each command type has a corresponding event. Downstream services listen for these events and execute the actions. The parser's role is purely to recognize patterns and create command objects—execution happens elsewhere.

What Happens Next
==================

Parsed commands are published as events and routed to specialized services:

- **AutomationCommandParsedEvent** → AutomationService
- **MarkCommandParsedEvent** → MarkService
- **GridCommandParsedEvent** → GridService
- **DictationCommandParsedEvent** → DictationCoordinator
- **SystemControlCommandParsedEvent** → PauseStateManager

These execution services are covered in :doc:`command_execution_services`.
