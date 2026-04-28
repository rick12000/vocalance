Command flow
############

A *command* is any voice or sound input that produces a single
observable change: a click, a keypress, the appearance of a
numbered grid, the execution of a saved mark. Two input types
produce one — spoken words or a trained non-speech sound — but
both travel through the same pipeline from end to end.

This chapter picks up where :doc:`capture` left off (audio chunks
arriving on the bus) and follows the path to the OS action.

Stages at a glance
==================

The pipeline has four stages.

.. mermaid::

   flowchart LR
       Chunks[AudioChunkCapturedEvent] --> S1[1. Segment]
       S1 --> S2[2. Recognize]
       S2 --> S3[3. Parse]
       S3 --> S4[4. Execute]
       S4 --> OS[OS input]

The speech and sound paths run in parallel through stages 1 and
2, converge at stage 3, and stay together for stage 4.

Zooming in one level shows that each stage is a pair of services
in the speech-and-sound stages, and a single service in the
parse-and-execute stages.

.. mermaid::

   flowchart LR
       subgraph S1["1. Segment"]
           CSeg[CommandSegmenterService]
           SSeg[SoundSegmenterService]
       end
       subgraph S2["2. Recognize"]
           CSpeech[CommandSpeechService<br/><i>Vosk</i>]
           SRec[SoundService<br/><i>YAMNet + k-NN</i>]
       end
       subgraph S3["3. Parse"]
           Parser[CentralizedCommandParser]
       end
       subgraph S4["4. Execute"]
           Auto[AutomationService]
           Mark[MarkService]
           Grid[GridService]
           Pause[PauseStateManager]
       end
       Bus((Event bus))
       CSeg -->|CommandAudioSegmentReadyEvent| Bus
       SSeg -->|ProcessAudioChunkForSoundRecognitionEvent| Bus
       Bus --> CSpeech
       Bus --> SRec
       CSpeech -->|CommandTextRecognizedEvent| Bus
       SRec -->|CustomSoundRecognizedEvent| Bus
       Bus --> Parser
       Parser -->|*CommandParsedEvent| Bus
       Bus --> Auto
       Bus --> Mark
       Bus --> Grid
       Bus --> Pause

Every arrow is a publish or a delivery. The remainder of the
chapter walks through one stage at a time.

----

Stage 1 — Segmentation
======================

Both Vosk and YAMNet need a *complete clip*, not a stream. The
continuous flow of ``AudioChunkCapturedEvent`` therefore has to
be cut into clips first. Two segmenters do this in parallel, one
per input type.

Shared state machine
--------------------

Both segmenters implement the same VAD-driven state machine,
differing only in the parameters they tune it with.

.. mermaid::

   flowchart LR
       Idle[<b>Idle</b><br/>holding pre-roll buffer] -->|chunk energy<br/>&gt; threshold| Cap[<b>Capturing</b><br/>append chunks]
       Cap -->|silence streak<br/>&ge; min| Done[Emit clip]
       Cap -->|duration<br/>&ge; cap| Done
       Done -->|reset| Idle

Three things to note:

- The **pre-roll buffer** is prepended on entry to ``Capturing``
  so the first consonant of a phrase is not clipped.
- The **silence streak** is the configured number of consecutive
  sub-threshold chunks that ends a clip naturally.
- The **duration cap** is a hard upper bound; reaching it ends
  the clip even if the user has not paused.

The energy threshold is *adaptive*. It tracks a rolling estimate
of the room's noise floor and applies a configurable multiplier,
so the same settings work in a quiet room and a noisy café
without recalibration.

The two segmenters
------------------

The two services use the same machine with different tunings.

==================================================================  =================================  =================================
Parameter                                                           ``CommandSegmenterService``        ``SoundSegmenterService``
==================================================================  =================================  =================================
Tuned for                                                           Spoken utterances                  Short transients
Silence streak                                                      ~500 ms                            ~150 ms
Max clip duration                                                   several seconds                    ~1 second
Extra rejection                                                     —                                  Peak-amplitude / baseline ratio
Disabled while dictation is active                                  no                                 yes
Output event                                                        ``CommandAudioSegmentReadyEvent``  ``ProcessAudioChunkForSoundRecognitionEvent``
==================================================================  =================================  =================================

``SoundSegmenterService``
(``vocalance/app/services/command_flow/segmenting/sound_segmenter_service.py``)
needs the extra peak-amplitude gate because sustained background
noise can drift above the energy threshold without actually
spiking. It also subscribes to
``DictationModeDisableOthersEvent`` so that every spoken word
during dictation does not produce a false-positive sound clip.
``CommandSegmenterService``
(``vocalance/app/services/command_flow/segmenting/command_segmenter_service.py``)
has neither of those rules.

----

Stage 2 — Recognition
=====================

The two clip events produced by Stage 1 are now consumed by two
independent recognizers. They share no state.

Speech: Vosk
------------

``CommandSpeechService``
(``vocalance/app/services/command_flow/speech_recognition/command_speech_service.py``)
wraps an offline Vosk model (~50 MB, bundled). It subscribes to
``CommandAudioSegmentReadyEvent``, runs Vosk on the PCM, and
publishes the result.

.. mermaid::

   sequenceDiagram
       participant Seg as CommandSegmenterService
       participant Bus as Event bus
       participant Speech as CommandSpeechService
       participant Vosk as VoskEngine

       Seg->>Bus: CommandAudioSegmentReadyEvent
       Bus->>Speech: deliver
       Speech->>Vosk: recognize(pcm)
       Note over Speech,Vosk: runs on a daemon thread<br/>via run_blocking
       Vosk-->>Speech: lower-case text
       Speech->>Bus: CommandTextRecognizedEvent

Recognition is synchronous and blocks for hundreds of
milliseconds. It runs on a background thread via ``run_blocking``
so it does not stall the main loop
(:doc:`../foundations/concurrency`). Output is plain lower-case
text — no punctuation, no confidence score.

Vosk also plays a secondary role during dictation: it watches
for the stop trigger and modifier phrases. That role is covered
in :doc:`dictation_flow`; it is independent of the command path
described here.

Sound: YAMNet + k-NN
--------------------

``SoundService``
(``vocalance/app/services/command_flow/sound_recognition/sound_service.py``)
subscribes to ``ProcessAudioChunkForSoundRecognitionEvent`` and
runs a two-step pipeline.

.. mermaid::

   flowchart LR
       Clip[Sound clip] --> Pre[Resample &amp; normalize]
       Pre --> Emb[YAMNet<br/>embedding<br/><i>5,120-D</i>]
       Emb --> KNN[k-NN vote<br/>over user samples<br/>+ ESC-50 negatives]
       KNN --> Gate{Winning label<br/>is user-trained?}
       Gate -->|yes| Pub[CustomSoundRecognizedEvent]
       Gate -->|no, ESC-50 wins| Drop[Drop silently]

**Embedding.** YAMNet is a pre-trained classifier; the recognizer
ignores its class labels and extracts a 5,120-D vector from a
hidden layer. That vector encodes the acoustic character of the
sound in a way that generalizes across recording conditions.

**k-NN vote.** The user trains the recognizer by saying ``train
<label>`` and producing the sound a few times — no model
fine-tuning is required.

k-NN has no built-in "neither" category. To prevent every door
slam from being assigned to a user label, the recognizer keeps a
background set of samples drawn from **ESC-50** (a standard
library of environmental noises) under internal ``esc50_*``
labels. They participate in the vote like any other; if an
ESC-50 label wins, the result is silently dropped.

The published ``CustomSoundRecognizedEvent(label, confidence,
mapped_command)`` carries the command phrase the user has mapped
to the trained label in the Sounds tab.

----

Stage 3 — Parsing
=================

``CentralizedCommandParser``
(``vocalance/app/services/command_flow/parsing/parser.py``) is
the convergence point. It subscribes to *both*
``CommandTextRecognizedEvent`` and ``CustomSoundRecognizedEvent``
and runs them through the same pipeline.

For sounds, the mapped phrase is substituted before parsing, so
from this point on the parser cannot tell whether the input came
from speech or sound.

.. code-block:: python

   async def handle_custom_sound_recognized(self, sound_recognized):
       phrase = sound_recognized.mapped_command
       if not phrase:
           return
       await self.process_text_input(text=phrase, source="sound")

Two gates and a cascade
-----------------------

Every input passes two gates and then a cascade.

.. mermaid::

   flowchart LR
       In[Text from Vosk<br/>or sound mapping] --> G1{Within<br/>rate-limit<br/>window?}
       G1 -->|yes| D1[Drop]
       G1 -->|no| Cas[Family cascade<br/>first match wins]
       Cas --> M{Matched?}
       M -->|no| D2[Drop]
       M -->|yes| G2{Paused<br/>and not Resume?}
       G2 -->|yes| D3[Drop]
       G2 -->|no| Out[Publish typed event]

The **rate-limit gate** drops input that arrives within a few
hundred milliseconds of the previous successful parse, because
both Vosk and the sound recognizer can double-fire on the same
utterance.

The **pause gate** runs *after* the cascade so that "resume" can
still produce a typed command while paused; the pause check
filters the result, not the input.

The cascade tries the input against six families in a fixed
order:

#. **System** — ``pause`` / ``resume``. First so neither phrase
   can be hijacked by a user-defined automation.
#. **Dictation triggers** — start / stop / type / smart / visual /
   hidden / amend. Before marks so a single-word trigger like
   "type" is not consumed as a mark.
#. **Marks** — ``mark <label>``, ``visualize``, ``reset``, …
   Before grid so a mark named "five" still works while a grid
   is on screen.
#. **Grid** — ``grid show``, ``grid hover``, ``grid drag``, plus
   numeric selections.
#. **Automation** — every user-configured action. Before the
   fallback so a single-word automation can claim the word.
#. **Single-word mark fallback** — every otherwise-unmatched
   single word is treated as a mark execute. This is what makes
   mark navigation feel frictionless.

Inputs that match nothing are dropped silently. Voice input is
noisy; surfacing a parse error on every misheard syllable would
be more disruptive than helpful.

What the parser publishes
-------------------------

A successful parse produces one typed event, chosen by the
matched family.

============================  ===========================================
Match                         Published event
============================  ===========================================
System                        ``SystemControlCommandParsedEvent``
Dictation                     ``DictationCommandParsedEvent``
Mark                          ``MarkCommandParsedEvent``
Grid                          ``GridCommandParsedEvent``
Automation                    ``AutomationCommandParsedEvent``
============================  ===========================================

Each event carries the parsed command as a Pydantic value object
(e.g. ``MarkCreateCommand(label="home", x=540.0, y=720.0)``)
plus the source (``"stt"`` or ``"sound"``). The original text or
label is gone.

----

Stage 4 — Execution
===================

Five subscribers consume the parsed events. The dictation event
goes to ``DictationCoordinator`` and is covered in
:doc:`dictation_flow`; the other four are below.

Every executor that touches ``pyautogui`` routes through
``KeyboardInputService``
(``vocalance/app/services/keyboard_input_service.py``), which
serializes OS input via an ``asyncio.Lock``. A sequence of
"click, click, scroll up" reaches the OS in that order even if
three different services made the calls. The mechanism lives in
:doc:`../foundations/concurrency`.

Automation
----------

``AutomationService``
(``vocalance/app/services/command_flow/execution/automation_service.py``)
handles user-configured actions: hotkeys, key sequences, single
and multi-clicks, scrolls. It subscribes to
``AutomationCommandParsedEvent`` and dispatches by
``action_type``.

Two command shapes:

- ``ExactMatchCommand`` — execute the action once.
- ``ParameterizedCommand`` — execute it ``count`` times
  (``"scroll down five"``).

Two runtime rules:

- A per-key **cooldown** (default ~0.5 s) prevents the same
  command double-firing from rapid-fire recognitions.
- **Stepped scrolls** split large scrolls into a loop of small
  partial-scrolls with inter-step sleeps; most applications drop
  scroll deltas that arrive faster than a real mouse wheel
  produces them.

Marks
-----

``MarkService``
(``vocalance/app/services/command_flow/execution/mark_service.py``)
maps a short label to a screen position and clicks it on
request.

==================================  ===============================================
Command                             Effect
==================================  ===============================================
``MarkCreateCommand``               Persist the label at the current cursor
                                    position.
``MarkExecuteCommand``              Click the stored position for this label.
``MarkDeleteCommand``               Remove a single label.
``MarkResetCommand``                Clear all labels.
``MarkVisualizeCommand``            Show the on-screen mark overlay.
``MarkVisualizeCancelCommand``      Hide the overlay.
==================================  ===============================================

The cursor position for ``MarkCreateCommand`` is captured at
*parse time* (before the event is published), not at execution
time:

.. code-block:: python

   label = words[1]
   x, y = pyautogui.position()
   return MarkCreateCommand(label=label, x=float(x), y=float(y))

If the cursor moves between the end of the phrase and the run of
the executor, the saved coordinate is still the one current when
the phrase ended.

Grid
----

``GridService``
(``vocalance/app/services/command_flow/execution/grid/grid_service.py``)
implements a "show, then pick" interaction.

==========================  =====================================================================
Command                     Effect
==========================  =====================================================================
``GridShowCommand``         Compute rows × cols, publish ``GridStateEvent("visible")``.
``GridSelectCommand``       If visible, publish ``GridStateEvent("interaction_request")``.
==========================  =====================================================================

The grid service owns back-end state: visibility, click mode
(``"click"`` / ``"hover"`` / ``"drag"``), and cell ranking. The
overlay window that renders the grid lives in the UI layer
(:doc:`user_interface`).

A naive grid would label cells in row-major order, but the cell
the user actually wants is rarely the one labeled ``1``.
Vocalance re-orders labels so the most-clicked regions get the
lowest numbers next time. The bookkeeping lives in
``ClickTrackerService``
(``vocalance/app/services/command_flow/execution/grid/click_tracker_service.py``).

.. mermaid::

   sequenceDiagram
       participant User
       participant Grid as GridService
       participant Tracker as ClickTrackerService
       participant Disk

       User->>Grid: "grid show 16"
       Grid->>Tracker: read current ranking
       Tracker-->>Grid: ordered cell labels
       User->>Grid: "5"
       Grid->>Grid: click cell 5
       Grid->>Tracker: record click
       Tracker->>Disk: persist (debounced)

Two debouncers smooth the system: a UI re-rank publish that
batches a flurry of clicks into one snapshot, and a disk write
that batches a streak of clicks into one file write.

Pause / resume
--------------

``PauseStateManager``
(``vocalance/app/services/command_flow/pause_state_manager.py``)
owns the single shared paused flag. It subscribes to
``SystemControlCommandParsedEvent`` and toggles on
``PauseCommand`` / ``ResumeCommand``. The parser's pause gate and
every executor read this flag.

Where to read next
==================

The command flow ends at the OS boundary. The parser's other
output — ``DictationCommandParsedEvent`` — opens a long-running
session with its own state machine. That story is in
:doc:`dictation_flow`.
