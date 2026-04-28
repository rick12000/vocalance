Commands
########

A *command* in Vocalance is a short, atomic instruction that produces
a single observable change on the operating system: a click, a
keystroke, a jump to a saved screen position, the appearance of a
numbered grid. Saying "click", saying "home" (a saved mark), or
making a trained sound all produce commands. They differ in how the
phrase is recognized and which executor handles the result, but they
share the same shape end to end.

This chapter is the end-to-end story for that whole feature. It
picks up where :doc:`capture` left off — raw audio chunks arriving
at the segmenter services — and walks all the way to the OS-side
action the user sees.

The journey of a command
========================

A command travels through four stages:

.. mermaid::

   flowchart LR
       Cap[Capture] --> Seg[Segment]
       Seg --> Rec[Recognize]
       Rec --> Par[Parse]
       Par --> Exec[Execute]
       Exec --> OS[OS input]

Each stage transforms the data: capture produces audio chunks, the
segmenters emit short clips, the recognizers turn those clips into
text or a label, the parser turns text or labels into typed commands,
the executors turn typed commands into OS calls.

The rest of this chapter walks through the four stages in order, then
covers each executor family in detail.

Stage 1: segmenting the stream
==============================

The recognizers used for commands are *clip-based*: they take a
complete utterance (or a complete sound) and return a single answer.
The capture layer delivers a continuous stream of audio chunks, so
the first job of the command path is to cut that stream into clips.

Two services do this in parallel:

- ``CommandSegmenterService`` produces clips of human speech.
- ``SoundSegmenterService`` produces clips of short transients
  (snaps, claps, taps).

Both are first-class services in
``vocalance/app/services/audio/segmenting/``. Each one subscribes to
``AudioChunkCapturedEvent`` on the bus, runs each chunk through a
small state machine, and emits its own clip event on the bus when an
utterance ends.

The state machine is shared; the parameters are not.

The segmenter as a state machine
--------------------------------

A segmenter is in one of two states: idle (listening, retaining a
short pre-roll buffer) or capturing (appending each incoming chunk
to a running buffer).

.. mermaid::

   flowchart LR
       Idle[Idle<br/>buffer pre-roll only] -->|energy &gt; threshold| Cap[Capturing<br/>append chunks]
       Cap -->|silence streak| Done[Finalize clip]
       Cap -->|max duration| Done
       Done -->|emit clip| Idle

The transition from idle to capturing fires when the chunk's energy
crosses an *adaptive* speech threshold. The threshold is recomputed
every chunk against a rolling estimate of the room's noise floor, so
the same configuration works in a quiet office and a noisy café
without recalibration. The pre-roll buffer is dumped into the clip
on the way in, so the leading consonant of the first word is not
clipped.

Capture ends in one of two ways: a configurable streak of
sub-threshold chunks (the speaker stopped talking), or a hard
duration cap (a sustained noise fooled the energy gate).

Two kinds of segmenter
----------------------

The two services use the same state machine with different
``SegmentConfig`` values, tuned to the inputs they expect:

=======================  =================================  ==========================================
Aspect                   Command segmenter                  Sound segmenter
=======================  =================================  ==========================================
Silence streak to end    ~half a second                     Around 150 ms
Maximum clip duration    Several seconds                    Roughly 1 second
Quality gates            Minimum duration                   Minimum duration plus a peak-energy ratio
Output event             ``CommandAudioSegmentReadyEvent``  ``ProcessAudioChunkForSoundRecognitionEvent``
=======================  =================================  ==========================================

The command segmenter is patient — natural speech has mid-utterance
pauses, and being too eager to end a clip would chop commands in
half. The sound segmenter is impatient — claps and snaps are
transient, and the clip needs to be tight around the event. The
extra peak-ratio gate on the sound segmenter rejects clips whose
loudest sample is barely above the noise floor, which catches
background noise that drifts above the threshold without being a
real "tap".

A small extra rule lives in the sound segmenter: while a dictation
session is active, the dictated speech would otherwise produce a
constant stream of false-positive sound clips. The service therefore
subscribes to the dictation mode event on the bus and mutes itself
while dictation is on. The mute is a flag inside the service, not
something the capture layer is aware of.

Stage 2: recognizing speech and sounds
======================================

Two recognizers run in parallel. They never share state; they meet
again only at the parser.

Speech: Vosk
------------

Speech goes through ``SpeechToTextService``
(``vocalance/app/services/audio/stt/stt_service.py``), which wraps an
offline Vosk model. Vosk takes a complete clip of PCM and returns
plain lower-case text, with no streaming and no confidence score. It
fits the clip-based output of the command segmenter exactly:

.. mermaid::

   sequenceDiagram
       participant Seg as CommandSegmenter
       participant Bus
       participant STT as SpeechToTextService
       participant Vosk

       Seg->>Bus: CommandAudioSegmentReadyEvent
       Bus->>STT: deliver
       STT->>Vosk: recognize(pcm)
       Vosk-->>STT: text
       STT->>Bus: CommandTextRecognizedEvent

Vosk also has a side-channel role during dictation (recognizing the
stop trigger and modifier phrases). That role is covered in
:doc:`dictation`; it does not affect the command path.

Sound: YAMNet + k-NN
--------------------

Sound clips go through ``SoundService``
(``vocalance/app/services/audio/sound_recognizer/streamlined_sound_service.py``)
which wraps a recognizer pipeline of four steps:

.. mermaid::

   flowchart LR
       Clip[Sound clip] --> Pre[Resample &amp; normalize]
       Pre --> Yam[YAMNet embedding]
       Yam --> KNN[k-NN vote]
       KNN --> Out{User label?}
       Out -->|yes| Pub[CustomSoundRecognizedEvent]
       Out -->|no, ESC-50| Drop[drop]

YAMNet is a pre-trained sound-classification model shipped with the
application. The recognizer does not use YAMNet's labels; it pulls a
5,120-dimensional embedding from a hidden layer and uses that vector
as a feature for a k-nearest-neighbors lookup over the user's own
trained samples. With three to five samples per label, the user
trains a small personal classifier without ever touching a model.

k-NN has no notion of "neither". Without help it would assign every
door slam, keystroke, and cough to one of the user's labels. The
recognizer is therefore bootstrapped with the **ESC-50** dataset of
environmental noises stored under ``esc50_*`` labels. Those samples
participate in the vote like any other; if the winner is one of
them, the result is silently dropped before the bus sees it. The
practical effect is a "neither" basket built out of ordinary
background noise.

When a user-trained label wins, the service publishes a
``CustomSoundRecognizedEvent`` carrying the label and the *command
phrase* the user has mapped that label to in the Sounds tab.

Stage 3: parsing
================

The parser is ``CentralizedCommandParser``
(``vocalance/app/services/commands/parser.py``). It is the point
where the speech path and the sound path converge: it subscribes to
both ``CommandTextRecognizedEvent`` and ``CustomSoundRecognizedEvent``
and routes them through the same pipeline.

For sounds, one extra step runs first: the recognized label is
swapped for its mapped command phrase. After that the rest of the
pipeline cannot tell whether a command came from speech or sound.

.. code-block:: python

   async def handle_custom_sound_recognized(self, sound_recognized):
       phrase = sound_recognized.mapped_command or self.sound_to_command_mapping.get(sound_recognized.label)
       if not phrase:
           return
       await self.process_text_input(text=phrase, source="sound")

The two gates
-------------

Every input passes through two checks before parsing.

A **rate limit** drops any input that arrives within a short window
(a few hundred milliseconds, configurable) of the previous successful
parse. Vosk and the sound recognizer occasionally double-fire on the
same utterance; without this gate, "click" would sometimes click
twice.

A **pause gate** drops every command except ``Resume`` while the
shared paused flag is set. The user toggles it by saying "pause" or
"resume". The parse runs *first* so that "resume" still produces a
command and reaches the bus; the gate is checked on the parsed
result.

The cascade
-----------

Once the gates have passed, the parser tries to match the input
against five families of commands in a fixed order. The first family
that matches wins; nothing else is tried.

.. mermaid::

   flowchart TD
       In[Normalized text] --> S1[1. System: pause / resume]
       S1 --> S2[2. Dictation triggers]
       S2 --> S3[3. Marks]
       S3 --> S4[4. Grid]
       S4 --> S5[5. Automation]
       S5 --> S6[6. Single-word mark fallback]
       S6 --> Out[First match wins]

The implementation is a linear pipeline in
``parse_full_command_text``
(``vocalance/app/services/commands/utilities/text_command_parse.py``):

.. code-block:: python

   steps = [
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

The order is deliberate. ``pause`` must always be safe to say and
must not be hijacked by a user-defined automation. Dictation triggers
come next so they cannot be hijacked by a single-word mark. Marks
come before grid because a mark named "five" must still work when a
grid is on screen — a grid pass would otherwise consume the bare
number. Automation comes before the mark fallback so that a
user-defined automation can claim a single word. The fallback
catches every remaining single word and treats it as a mark
execution attempt; this is what makes mark navigation feel
frictionless.

Inputs that match nothing are silently discarded. Voice input is
noisy, and surfacing parse errors on every misheard syllable would
be more disruptive than helpful.

What the parser publishes
-------------------------

A successful parse produces a typed *parsed-event* on the bus, one
per family:

.. mermaid::

   flowchart LR
       P[Parser] --> SCE[SystemControlCommandParsedEvent]
       P --> DCE[DictationCommandParsedEvent]
       P --> MCE[MarkCommandParsedEvent]
       P --> GCE[GridCommandParsedEvent]
       P --> ACE[AutomationCommandParsedEvent]

Each event carries the parsed command (a Pydantic value object such
as ``MarkCreateCommand(label="home", x=540.0, y=720.0)``) along with
the source the input arrived from ("stt" or "sound"). From this
point on, only typed events flow downstream; the original text or
label is gone.

Stage 4: executing
==================

Four services subscribe to the parsed-event types and turn typed
commands into OS-side actions: **automation**, **mark**, **grid**
(plus its companion **click tracker**), and a small **pause-state
manager** that owns the system pause flag. The dictation parsed
event goes to the dictation coordinator and is the subject of
:doc:`dictation`.

The OS input boundary
---------------------

All three OS-touching executors call into ``pyautogui``. Every such
call is routed through a single shared ``KeyboardInputService``
(``vocalance/app/services/commands/utilities/input_executor.py``).
For the purposes of this chapter, the only fact that matters is that
this service guarantees **strict FIFO ordering** of OS input across
all callers: a sequence of "click, click, scroll up" arrives at the
OS in that order even though three different executors made the
calls. The mechanism that achieves this is detailed in
:doc:`../foundations/concurrency`.

Automation
----------

``AutomationService`` (``vocalance/app/services/automation_service.py``)
runs the user's configured actions: hotkeys, key presses, key
sequences, single clicks, double / triple clicks, and scrolls. It
subscribes to ``AutomationCommandParsedEvent``, builds an action
function from the typed command's ``action_type`` / ``action_value``
pair, and runs it on the input service.

Two automation command shapes exist:

- ``ExactMatchCommand`` runs the action once.
- ``ParameterizedCommand`` runs the action ``count`` times — the
  output of "scroll down five".

Two practical rules sit on top:

- A **per-key cooldown** (default half a second, configurable)
  prevents the same command from firing twice in tight succession.
  Without it, "click click click" risks producing more clicks than
  the user intended.
- **Stepped scrolls** decompose a directional scroll into a small
  loop of partial scrolls with sleeps in between. Most applications
  drop scroll deltas that arrive faster than a real wheel can
  produce them; the stepping makes the scroll feel native.

Marks
-----

``MarkService`` (``vocalance/app/services/mark_service.py``) maps a
short label to a screen position and clicks the position when asked.
Six command types share a single ``MarkCommandParsedEvent``:

==================================  ================================================
Command                             Effect
==================================  ================================================
``MarkCreateCommand``               Persist the label at ``(x, y)``.
``MarkExecuteCommand``              Click the stored ``(x, y)`` for the label.
``MarkDeleteCommand``               Remove a single label.
``MarkResetCommand``                Clear all labels.
``MarkVisualizeCommand``            Show the on-screen overlay of every mark.
``MarkVisualizeCancelCommand``      Hide the overlay.
==================================  ================================================

Mark creation has a subtle rule worth pointing out. The parser's
grammar for "create mark home" produces a ``MarkCreateCommand`` with
the label *and* the cursor position at parse time:

.. code-block:: python

   if words[0] == triggers.mark_create_prefix and len(words) == 2:
       label = words[1]
       x, y = pyautogui.position()
       return MarkCreateCommand(label=label, x=float(x), y=float(y))

The cursor snapshot happens before the event is published. If the
user moves the cursor between saying the phrase and the executor
running, the saved coordinate is still the one current when the
phrase ended.

Grid
----

``GridService`` (``vocalance/app/services/grid/grid_service.py``)
implements a two-step "show, then pick" interaction. Two commands
share a single ``GridCommandParsedEvent``:

=======================  =====================================================================
Command                  Effect
=======================  =====================================================================
``GridShowCommand``      Compute rows × cols, publish ``GridStateEvent("visible")``.
``GridSelectCommand``    If the grid is visible, publish ``GridStateEvent("interaction_request")``.
=======================  =====================================================================

The grid service is the back-end half: it tracks visibility, the
configured click mode (``"click"`` / ``"hover"`` / ``"drag"``), and
publishes ``GridStateEvent`` for the overlay controller to render.
The overlay window itself lives in the UI layer
(:doc:`user_interface`).

A naive numbered grid labels cells in row-major order. The cell the
user actually wants is rarely the one labelled ``1``. Vocalance
re-orders the labels so the most-clicked regions get the lowest
numbers next time. The bookkeeping lives in
``ClickTrackerService``
(``vocalance/app/services/grid/click_tracker_service.py``).

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
       Grid->>Grid: click cell at "5"
       Grid->>Tracker: click happened here
       Tracker->>Tracker: append to history
       Tracker->>Disk: persist (debounced)

Two debouncers smooth the system out: a UI re-rank publish that
batches a flurry of clicks into one snapshot, and a disk write that
batches a streak of clicks into one file write.

System
------

``PauseStateManager`` (``vocalance/app/services/pause_state_manager.py``)
owns the single shared paused flag. It subscribes to
``SystemControlCommandParsedEvent`` and toggles the flag in response
to ``PauseCommand`` / ``ResumeCommand``. Every other executor — and
the parser itself — consults the flag before acting.

Where to read next
==================

The fifth family the parser can emit, ``DictationCommandParsedEvent``,
goes to the dictation coordinator. From there the behaviour is no
longer a single transaction; it is a long-running session with its
own state machine. That story is in :doc:`dictation`.
