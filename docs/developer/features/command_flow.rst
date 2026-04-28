Command Flow
############

A *command* in Vocalance is any instruction that produces a single
observable change: a click, a keypress, the appearance of a numbered
grid, or the execution of a saved mark. Two kinds of input can produce
one — spoken words or a trained non-speech sound — but both travel
through the same pipeline from end to end.

This chapter picks up where :doc:`capture` left off (audio chunks
arriving on the bus) and follows the command path all the way to the
OS action. The chapter is arranged as a journey with four stages.

.. mermaid::

   flowchart LR
       Chunks[AudioChunkCapturedEvent] --> Seg[1. Segment]
       Seg --> Rec[2. Recognize]
       Rec --> Par[3. Parse]
       Par --> Exec[4. Execute]
       Exec --> OS[OS input]

Each stage transforms the data: the segmenters cut the stream into
clips, the recognizers turn clips into text or a label, the parser
turns text or a label into a typed command object, and the executors
turn typed commands into OS calls. The speech and sound paths run in
parallel through stages 1 and 2, converge at stage 3, and stay
together for stage 4.

----

Stage 1 — Segmentation
=======================

Commands are *clip-based*: both Vosk (speech) and YAMNet (sound) take
a complete audio clip and return a single answer. The continuous stream
of ``AudioChunkCapturedEvent`` must therefore be cut into clips first.
Two services do this in parallel, one per input type.

The shared segmentation model
------------------------------

Both segmenters implement the same state machine, differing only in
the parameters used to tune it.

.. mermaid::

   flowchart LR
       Idle[Idle<br/>pre-roll buffer only] -->|energy &gt; threshold| Cap[Capturing<br/>append chunks]
       Cap -->|silence streak| Done[Clip ready]
       Cap -->|max duration| Done
       Done -->|publish| Idle

The segmenter starts idle, holding a short *pre-roll buffer* of recent
chunks. When a chunk's energy crosses an adaptive threshold it
transitions to capturing: it prepends the pre-roll (so the first
consonant is not clipped) and starts appending every incoming chunk.
Capture ends when either a configurable streak of sub-threshold chunks
(the sound stopped) or a hard duration cap (something sustained tricked
the energy gate) is reached. The finished buffer is emitted as a bus
event and the segmenter returns to idle.

The threshold is adaptive. It tracks a rolling estimate of the room's
noise floor and applies a configurable multiplier, so the same settings
work in a quiet room and a noisy café without recalibration.

1.1 Speech segmentation (``CommandSegmenterService``)
------------------------------------------------------

``CommandSegmenterService``
(``vocalance/app/services/command_flow/segmenting/command_segmenter_service.py``)
is tuned for spoken utterances. Speech has natural mid-utterance
pauses — a brief silence does not mean the speaker has finished. The
segmenter therefore uses a longer silence streak (~half a second) and
a longer maximum duration (several seconds).

When the clip is ready the service publishes
``CommandAudioSegmentReadyEvent(audio_bytes, sample_rate)``. That event
is the handoff to Stage 2 speech recognition.

1.2 Sound segmentation (``SoundSegmenterService``)
---------------------------------------------------

``SoundSegmenterService``
(``vocalance/app/services/command_flow/segmenting/sound_segmenter_service.py``)
is tuned for short, transient sounds: claps, snaps, lip-pops. These
events are tightly bounded in time, so the segmenter uses a much
shorter silence streak (~150 ms) and a much shorter maximum duration
(roughly 1 second).

An extra quality gate applies: the clip's peak amplitude must be some
minimum ratio above the baseline energy. Sustained background noise
can drift above the energy threshold without actually spiking; this
gate rejects those false triggers.

There is one additional rule: while a dictation session is active,
every spoken word would produce a false-positive sound clip. The
service subscribes to ``DictationModeDisableOthersEvent`` and mutes
itself for the duration of any dictation session. The capture layer
and the command segmenter are entirely unaware of this; it is a
self-contained rule inside the sound segmenter.

When the clip is ready the service publishes
``ProcessAudioChunkForSoundRecognitionEvent(audio_chunk, sample_rate)``.
That event is the handoff to Stage 2 sound recognition.

----

Stage 2 — Recognition
======================

The two clip events produced by Stage 1 are now processed by two
completely independent recognizers. They share no state and have no
awareness of each other. Their outputs converge only at Stage 3.

2.1 Speech recognition: Vosk
-----------------------------

``CommandSpeechService``
(``vocalance/app/services/command_flow/speech_recognition/command_speech_service.py``) wraps an
offline Vosk model. It subscribes to ``CommandAudioSegmentReadyEvent``,
feeds the PCM to Vosk, and publishes the result.

.. mermaid::

   sequenceDiagram
       participant Seg as CommandSegmenterService
       participant Bus
       participant STT as CommandSpeechService
       participant Vosk

       Seg->>Bus: CommandAudioSegmentReadyEvent
       Bus->>STT: deliver
       STT->>Vosk: recognize(pcm)
       Vosk-->>STT: plain lower-case text
       STT->>Bus: CommandTextRecognizedEvent

Vosk runs offline on a bundled model (~50 MB). Recognition is
synchronous and blocks for a few hundred milliseconds; it runs on a
background thread via ``run_blocking`` so it does not stall the main
event loop (see :doc:`../foundations/concurrency`). The output is
plain lower-case text with no punctuation or confidence score.

Vosk also plays a secondary role during dictation: it watches for
the stop trigger and modifier phrases. That role is covered in
:doc:`dictation`; it does not affect the command path described here.

2.2 Sound recognition: YAMNet + k-NN
--------------------------------------

``SoundService``
(``vocalance/app/services/command_flow/sound_recognition/sound_service.py``)
subscribes to ``ProcessAudioChunkForSoundRecognitionEvent`` and runs
a two-step recognition pipeline.

.. mermaid::

   flowchart LR
       Clip[Sound clip] --> Pre[Resample &amp; normalize]
       Pre --> Emb[YAMNet embedding<br/>5120-D vector]
       Emb --> KNN[k-NN vote<br/>over user samples]
       KNN --> Gate{User label<br/>wins?}
       Gate -->|yes| Pub[CustomSoundRecognizedEvent]
       Gate -->|ESC-50 wins| Drop[drop silently]

**Step 1 — embedding.** YAMNet is a pre-trained audio-classification
model. The recognizer does not use its class labels; it extracts a
5,120-dimensional embedding from a hidden layer. That vector encodes
the acoustic character of the sound in a way that generalises across
recording conditions.

**Step 2 — k-NN vote.** A k-nearest-neighbours lookup compares the
embedding against the user's own trained samples. The user trains the
recognizer by saying "train <label>" and making the sound three to five
times; no model fine-tuning is required.

k-NN has no built-in "neither" category. Without help it would assign
every door slam or keystroke to one of the user's labels. The
recognizer therefore keeps a background set of samples drawn from
**ESC-50**, a standard library of environmental noises, stored under
internal ``esc50_*`` labels. Those samples participate in the vote like
any other; if the winner is an ESC-50 label the result is silently
dropped. This creates a practical "neither" basket without requiring
the user to train it explicitly.

When a user label wins, the service publishes a
``CustomSoundRecognizedEvent(label, confidence, mapped_command)``.
The ``mapped_command`` field carries the command phrase the user has
mapped to that label in the Sounds tab.

----

Stage 3 — Parsing
=================

``CentralizedCommandParser``
(``vocalance/app/services/command_flow/parsing/parser.py``) is the convergence
point. It subscribes to both ``CommandTextRecognizedEvent`` (from Vosk)
and ``CustomSoundRecognizedEvent`` (from the sound recognizer) and runs
them through the same pipeline.

For sounds, the mapped command phrase is substituted before parsing, so
from here onward the parser cannot tell whether the input came from
speech or sound.

.. code-block:: python

   async def handle_custom_sound_recognized(self, sound_recognized):
       phrase = sound_recognized.mapped_command
       if not phrase:
           return
       await self.process_text_input(text=phrase, source="sound")

The two gates
-------------

Every input passes two checks before parsing.

A **rate-limit gate** drops input that arrives within a short
configurable window (a few hundred milliseconds) of the previous
successful parse. Both Vosk and the sound recognizer can double-fire
on the same utterance; this gate prevents the duplicate reaching the
executor.

A **pause gate** drops everything except ``ResumeCommand`` while the
system is paused. The parse still runs so "resume" can produce a typed
command; the gate acts on the parsed result.

The cascade
-----------

After the gates, the parser tries the input against five families in
a fixed order. The first match wins.

.. mermaid::

   flowchart TD
       In[Normalized text] --> S1[1. System: pause / resume]
       S1 --> S2[2. Dictation triggers]
       S2 --> S3[3. Marks]
       S3 --> S4[4. Grid]
       S4 --> S5[5. Automation]
       S5 --> S6[6. Single-word mark fallback]
       S6 --> Out[First match wins]

The order is deliberate. ``pause`` must always be safe to say; it
cannot be hijacked by a user-defined automation. Dictation triggers
come next so they cannot be consumed by a single-word mark. Marks come
before grid so a mark named "five" still works while a grid is on
screen. Automation comes before the fallback so a user-defined
automation can claim a single word. The fallback catches every
remaining single word and treats it as a mark-execute attempt, which
is what makes mark navigation feel frictionless.

Inputs that match nothing are silently discarded. Voice input is noisy,
and surfacing a parse error on every misheard syllable would be more
disruptive than helpful.

What the parser publishes
-------------------------

A successful parse produces a typed event, one per family:

.. mermaid::

   flowchart LR
       P[Parser] --> SCE[SystemControlCommandParsedEvent]
       P --> DCE[DictationCommandParsedEvent]
       P --> MCE[MarkCommandParsedEvent]
       P --> GCE[GridCommandParsedEvent]
       P --> ACE[AutomationCommandParsedEvent]

Each event carries the parsed command as a Pydantic value object (e.g.
``MarkCreateCommand(label="home", x=540.0, y=720.0)``) plus the source
("stt" or "sound"). The original text or label is gone.

----

Stage 4 — Execution
====================

Four services subscribe to the parsed-event types: **automation**,
**mark**, **grid**, and **pause-state manager**. The
``DictationCommandParsedEvent`` goes to the dictation coordinator and
is covered in :doc:`dictation`.

All OS-touching executors share one rule: every call to ``pyautogui``
is routed through ``KeyboardInputService``
(``vocalance/app/services/keyboard_input_service.py``),
which serialises OS input via an ``asyncio.Lock``. A sequence of
"click, click, scroll up" arrives at the OS in that order regardless
of which service made each call. The mechanism is explained in
:doc:`../foundations/concurrency`.

4.1 Automation
--------------

``AutomationService``
(``vocalance/app/services/command_flow/execution/automation_service.py``)
handles the user's configured actions: hotkeys, key sequences, single
and multi-clicks, and scrolls. It subscribes to
``AutomationCommandParsedEvent`` and dispatches based on
``action_type`` / ``action_value``.

Two command shapes exist:

- ``ExactMatchCommand`` executes the action once.
- ``ParameterizedCommand`` executes it ``count`` times ("scroll down five").

Two runtime rules apply:

- A per-key **cooldown** (default ~0.5 s, configurable) prevents the
  same command double-firing from two rapid-fire recognitions.
- **Stepped scrolls** split large scrolls into a loop of small
  partial-scrolls with inter-step sleeps, because most applications
  drop scroll deltas that arrive faster than a real mouse wheel produces
  them.

4.2 Marks
---------

``MarkService``
(``vocalance/app/services/command_flow/execution/mark_service.py``)
maps a short label to a screen position and clicks it on request.

==================================  ================================================
Command                             Effect
==================================  ================================================
``MarkCreateCommand``               Persist the label at the current cursor position.
``MarkExecuteCommand``              Click the stored position for this label.
``MarkDeleteCommand``               Remove a single label.
``MarkResetCommand``                Clear all labels.
``MarkVisualizeCommand``            Show the on-screen mark overlay.
``MarkVisualizeCancelCommand``      Hide the overlay.
==================================  ================================================

One detail is worth pointing out about mark creation. The parser
captures the cursor position at *parse time* (before the event is
published), not at execution time:

.. code-block:: python

   label = words[1]
   x, y = pyautogui.position()
   return MarkCreateCommand(label=label, x=float(x), y=float(y))

If the cursor moves between when the phrase ends and when the executor
runs, the saved coordinate is still the one that was current when the
phrase ended.

4.3 Grid
--------

``GridService``
(``vocalance/app/services/command_flow/execution/grid/grid_service.py``)
implements a two-step "show, then pick" interaction.

=======================  =====================================================================
Command                  Effect
=======================  =====================================================================
``GridShowCommand``      Compute rows × cols, publish ``GridStateEvent("visible")``.
``GridSelectCommand``    If visible, publish ``GridStateEvent("interaction_request")``.
=======================  =====================================================================

The grid service owns the back-end state: visibility, click mode
(``"click"`` / ``"hover"`` / ``"drag"``), and cell ranking. The
overlay window that renders the grid lives in the UI layer
(:doc:`user_interface`).

A naive grid would label cells in row-major order. The cell the user
actually wants is rarely the one labelled ``1``. Vocalance re-orders
the labels so the most-clicked regions get the lowest numbers on the
next invocation. The bookkeeping lives in
``ClickTrackerService``
(``vocalance/app/services/command_flow/execution/grid/click_tracker_service.py``).

.. mermaid::

   sequenceDiagram
       participant User
       participant Grid
       participant Tracker as ClickTracker
       participant Disk

       User->>Grid: "grid show 16"
       Grid->>Tracker: ask for current ranking
       Tracker-->>Grid: cell label order
       User->>Grid: "5"
       Grid->>Grid: click cell "5"
       Grid->>Tracker: click event
       Tracker->>Disk: persist (debounced)

Two debouncers smooth the system: a UI re-rank publish that batches a
flurry of clicks into one snapshot, and a disk write that batches a
streak of clicks into one file write.

4.4 System (pause / resume)
----------------------------

``PauseStateManager``
(``vocalance/app/services/command_flow/pause_state_manager.py``)
owns the single shared paused flag. It subscribes to
``SystemControlCommandParsedEvent`` and toggles on ``PauseCommand`` /
``ResumeCommand``. The parser and all executors check this flag before
acting.

----

Where to read next
==================

The command flow ends at the OS boundary. The other output of the
parser — ``DictationCommandParsedEvent`` — starts a long-running
session with its own state machine. That story is in :doc:`dictation`.
