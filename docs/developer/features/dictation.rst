Dictation
#########

Dictation is the one feature in Vocalance that does not fit the
"phrase in, action out" shape of every other command. Saying
"dictate" does not produce a single OS event. It opens a *session*
that streams audio to a streaming speech model, watches for a stop
phrase, optionally rewrites the result with a local large language
model, and finally types the resulting text into whatever
application is focused. Sessions can last seconds or minutes; the
application stays responsive throughout.

This chapter is the end-to-end story for that feature. It picks up
where :doc:`command_flow` ended — the parser has just emitted a
``DictationCommandParsedEvent`` — and follows the session all the
way to the keystrokes the user sees.

The sections below are layered on top of each other. The first three
describe what a session *is*: the modes, the state machine, and the
audio paths. The next three describe how a session *behaves*:
modifiers, aliases, and the LLM handshake. The last describes how
the popup mirrors the back-end through the bus.

The six modes
=============

A dictation session has a *mode*, fixed at the moment it starts. All
modes share the same audio path and the same modifier system; they
differ in what they do with the finalized text.

==========  ============================================================
Mode        What it does on stop
==========  ============================================================
Standard    Type each finalized utterance immediately, as it comes in.
Type        Like Standard, plus auto-stop after a configurable silence.
Visual      Hold streaming text in a popup, type the whole block on stop.
Hidden      Like Visual, but never show the streaming text to the user.
Smart       Like Visual, then rewrite via the LLM before typing.
Amend       Like Smart, but rewrite the user's pre-existing selection.
==========  ============================================================

Two of the six are *LLM modes* (Smart and Amend); they enter an
extra "processing" stage after recording ends. The other four go
straight back to idle.

The mode is chosen by which start command the parser emitted —
``DictationStartCommand`` for Standard, ``DictationSmartStartCommand``
for Smart, and so on. There is no in-session mode switch; the user
ends one session and starts another with a different trigger phrase.

The state machine
=================

A session is always in one of three states. The transition table is
enforced inside ``DictationCoordinator``:

.. mermaid::

   flowchart LR
       Idle[Idle] -->|start trigger| Rec[Recording]
       Rec -->|stop, simple modes| Idle
       Rec -->|stop, LLM modes| Proc[Processing LLM]
       Proc -->|LLM finished| Idle

In **Idle** the coordinator is alive but discarding audio. In
**Recording** the streaming model is consuming audio and the popup
is showing what it has heard so far. In **Processing LLM** the audio
has been finalized and an LLM is rewriting the accumulated text;
only Smart and Amend ever enter this state.

The coordinator validates every state change against an explicit
table:

.. code-block:: python

   VALID_DICTATION_STATE_TRANSITIONS = {
       DictationState.IDLE: frozenset({DictationState.RECORDING, DictationState.SHUTTING_DOWN}),
       DictationState.RECORDING: frozenset({DictationState.PROCESSING_LLM, DictationState.IDLE, DictationState.SHUTTING_DOWN}),
       DictationState.PROCESSING_LLM: frozenset({DictationState.IDLE, DictationState.SHUTTING_DOWN}),
       DictationState.SHUTTING_DOWN: frozenset(),
   }

Anything outside the table raises immediately. That guarantee is
what lets the rest of the coordinator's code assume invariants
("there is at most one active session", "no LLM run starts before
recording ends") without further locking.

The audio path during a session
===============================

The coordinator does not own an audio stream. While a session is
active, two recognizers work on the *same* audio in parallel.

.. mermaid::

   flowchart LR
       Cap[Capture] --> Dict[DictationCoordinator]
       Cap --> Cmd[CommandSegmenterService]
       Dict --> MS[Moonshine streaming]
       Cmd --> Vosk[Vosk side channel]
       MS --> Stream[Partial / final lines]
       Vosk --> Stop[Stop trigger]
       Vosk --> Mod[Modifier phrases]

The two recognizers play complementary roles:

- **Moonshine** is the streaming engine. It accepts PCM as it
  arrives and emits *partial* transcripts (subject to change) and
  *final* transcripts (committed). Moonshine segments speech
  internally using acoustic features, which is why dictation does
  not run audio through the command segmenter on the way in.
- **Vosk** stays live as a side channel. The same command segmenter
  used by command mode keeps producing clips during dictation; the
  speech-to-text service interprets each clip differently while a
  session is active, looking specifically for the stop trigger and
  for modifier phrases.

The two recognizers do not share state. Their outputs converge later
inside the coordinator, never in the recognizers themselves.

Routing chunks to Moonshine
---------------------------

The coordinator subscribes to ``AudioChunkCapturedEvent`` exactly like
the segmenters do. While idle, the handler is a no-op; while a
session is active, every chunk is appended to the current Moonshine
stream.

.. code-block:: python

   self.subscribe(AudioChunkCapturedEvent, self._handle_audio_chunk)

   def _handle_audio_chunk(self, event: AudioChunkCapturedEvent) -> None:
       self.feed_moonshine_audio_chunk(event.pcm_bytes, event.sample_rate)

Moonshine returns transcripts through *line callbacks*: a "line" is
its unit of finalization, roughly a phrase between natural pauses.
The coordinator's internal ``DictationMoonshineController`` registers
two:

- ``on_partial`` fires repeatedly while Moonshine refines its guess
  for the current line. The text is what the popup renders during
  Visual / Smart / Amend / Hidden modes.
- ``on_final`` fires once when the line is committed. The text is
  what gets appended to the session transcript.

In Standard and Type modes only the final callback matters: the
coordinator publishes a ``DictationTextRecognizedEvent`` and types
the line immediately. In the streaming modes, partials drive the
popup as ``PartialDictationTextEvent`` and finals are accumulated
into a buffer that becomes the input to either direct typing
(Visual / Hidden) or the LLM (Smart / Amend).

Vosk as a side channel
----------------------

While a dictation session is active, the speech-to-text service
inspects every Vosk recognition and reacts in one of three ways:

- If the result contains the configured stop trigger, the service
  publishes a ``CommandTextRecognizedEvent`` carrying the trigger
  text. The parser parses it as a ``DictationStopCommand``, which
  reaches the coordinator's command handler exactly like any other
  parsed command. The stop trigger is just a command, in other
  words; nothing special.
- If the result contains a configured modifier phrase ("snake
  case", "all caps", "spelling", and so on), the service publishes
  a ``DictationModifierPhraseEvent`` directly. The parser is
  bypassed; the modifier system reacts.
- Otherwise the result is dropped. Vosk's normal "I heard you say
  this" output is irrelevant during dictation — Moonshine is
  responsible for the prose.

Stopping a session
==================

A session ends in one of three ways:

1. **The user says the stop trigger.** Vosk recognizes it, the path
   above produces a ``DictationStopCommand``, the coordinator's
   handler runs ``stop_session``.
2. **Type-mode silence.** A background timer started at session
   creation watches the timestamp of the last finalized line. If the
   gap exceeds the configured threshold, the timer triggers
   ``stop_session`` itself.
3. **Application shutdown.** When the application asks for shutdown,
   the coordinator drives any in-flight session to completion as
   part of its own teardown.

Whichever path triggered it, the coordinator branches on the mode at
stop time:

.. mermaid::

   flowchart TD
       Stop[Stop requested] --> Halt[Halt Moonshine, collect text]
       Halt --> Mode{Mode?}
       Mode -->|Standard / Type| IdleA[Set IDLE, exit popup]
       Mode -->|Visual| TypeV[Type final text, exit popup]
       Mode -->|Hidden| TypeH[Type final text, no popup]
       Mode -->|Smart / Amend| Proc[Set PROCESSING_LLM,<br/>publish LLMProcessingStartedEvent]

Smart and Amend do *not* run the LLM inline at this point. They only
publish the started event and wait. The reason is the popup
handshake described in a later section.

The modifier system
===================

Modifiers transform the *formatting* of dictated text without
changing its content. Saying "snake case" puts subsequent words in
``snake_case``; "all caps" upper-cases them; "spelling" treats each
word as a sequence of letters; and so on. A modifier stays active
until the user toggles it off (by saying the same phrase) or until
another modifier in the same group replaces it.

Mutual exclusion
----------------

Modifiers are organized into two independent groups:

================  =====================================================
Group             Members
================  =====================================================
Casing            ``upper``, ``capitals``, ``camel``, ``snake``, ``kebab``, ``diminish``
Punctuation       ``spelling``, ``strip``
================  =====================================================

Within a group, only one modifier can be active at a time:
activating a casing modifier removes any other casing modifier;
activating a punctuation modifier removes any other punctuation
modifier. The two groups are independent — a session can be in
``snake`` casing and ``spelling`` punctuation simultaneously.

The Moonshine suppression window
--------------------------------

When the user says a modifier phrase, Vosk recognizes it and the
service publishes ``DictationModifierPhraseEvent`` immediately. But
the same audio is also flowing into Moonshine, which will eventually
produce a final line containing the modifier phrase as plain prose.
Without intervention, the modifier phrase would leak into the
dictated text.

The coordinator solves this with a short suppression window. When a
modifier phrase fires, the controller records a "suppress until"
timestamp half a second in the future:

.. code-block:: python

   def output_suppressed(self) -> bool:
       return time.monotonic() < self.moonshine_suppress_until

While the window is open, every Moonshine partial and final is
discarded. The window is long enough to swallow the modifier phrase
and short enough to keep the next phrase intact.

This is the only place where the two recognizers' outputs interact
with each other. Everything else they do is independent.

Aliases
=======

Aliases are user-configured shorthand expansions: saying "insert
address" emits a stored block of text, "insert signature" emits
another, and so on. The substitution is handled by
``DictationAliasService`` and is applied inside the coordinator's
segment pipeline.

The substitution is two-pass to interact correctly with modifiers:

1. The alias trigger is replaced with a placeholder.
2. The modifier-aware post-processing runs on the placeholder
   version. Snake-casing, for example, will not snake-case the
   placeholder itself.
3. The placeholder is replaced with the alias body.

The upshot: a snake-cased session does not snake-case the contents
of an alias, even though both the alias trigger and the alias body
went through the same pipeline.

The LLM handshake
=================

For Smart and Amend modes, the coordinator's job after recording
stops is to hand the accumulated text to the LLM and replace the
popup's contents with streaming LLM tokens. The popup needs to
finish its UI swap before the first token arrives, otherwise the
first few tokens would render in the wrong widget.

The coordination is a small handshake on the bus:

.. mermaid::

   sequenceDiagram
       participant Coord as Coordinator
       participant Bus
       participant Popup
       participant LLM as LLMService

       Coord->>Bus: LLMProcessingStartedEvent
       Bus->>Popup: deliver
       Popup->>Popup: swap to streaming UI
       Popup->>Bus: LLMProcessingReadyEvent
       Bus->>Coord: deliver
       Coord->>LLM: run (streaming)
       loop Each token
           LLM->>Bus: partial token event
           Bus->>Popup: deliver
       end
       LLM->>Bus: LLMProcessingCompletedEvent
       Bus->>Coord: deliver
       Coord->>Coord: type final text into focused app

Three properties of this design follow from it:

- The coordinator never drives the popup directly. It publishes one
  event and waits for the popup's reply on the bus. The back-end /
  front-end contract from the architecture chapter is preserved.
- Tokens stream straight from the LLM service to the popup, without
  going through the coordinator. The coordinator only sees the
  final completed text, which it types as a single block.
- The Amend mode adds one extra step *before* recording starts: the
  coordinator captures the user's current selection via copy, stashes
  it, and passes it to the LLM as the text to rewrite. The spoken
  text becomes the *instructions*, not the input to be transformed.

The popup as a back-end mirror
==============================

The dictation popup is the most visible piece of front-end in the
application, but it owns no state of its own. Every change in the
coordinator's state is published on the bus, and the popup's
controllers subscribe to those events and re-render:

================================================  ============================================
Event                                             Popup reaction
================================================  ============================================
``DictationStatusChangedEvent``                   Show / hide the popup, set the active mode.
``DictationModifierStateChangedEvent``            Update the active-modifier label.
``PartialDictationTextEvent``                     Update the streaming text widget.
``FinalDictationTextEvent``                       Append a finalized line.
``DictationSessionEvent``                         Drive the visual / smart / amend transitions.
``LLMProcessingStartedEvent``                     Begin the LLM-mode UI swap.
``LLMProcessingCompletedEvent``                   Hide.
================================================  ============================================

There are no method calls between the coordinator and the popup in
either direction — only events. The popup could be replaced with a
CLI display, or removed entirely, and the coordinator would not
notice.

Where to read next
==================

The user interface — the popup, the main window, the overlays — is
covered in :doc:`user_interface`. After that, the *foundations*
chapters explain how the audio chunks, the LLM tokens, and the OS
input calls actually move between threads and through the queue
without blocking each other.
