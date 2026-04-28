Dictation flow
##############

Dictation is the one feature that does not fit the "phrase in,
action out" shape of every other command. Saying ``dictate`` does
not produce a single OS event; it opens a *session* that streams
audio to a streaming speech model, watches for a stop phrase,
optionally rewrites the result with a local large language model,
and finally types the resulting text into whatever application is
focused. Sessions can last seconds or minutes.

This chapter picks up where :doc:`command_flow` ended — the
parser has just emitted a ``DictationCommandParsedEvent`` — and
follows the session to the keystrokes the user sees.

Flow at a glance
================

A session has three states. Audio flows through *two* recognizers
in parallel; their outputs converge inside the coordinator.

.. mermaid::

   flowchart LR
       Cmd[DictationCommandParsedEvent] --> Coord[DictationCoordinator]
       Chunks[AudioChunkCapturedEvent] --> Coord
       Chunks --> CSpeech[CommandSpeechService<br/><i>Vosk side channel</i>]
       Coord --> Moon[MoonshineEngine<br/><i>streaming dictation</i>]
       CSpeech -->|stop trigger or<br/>modifier phrase| Coord
       Moon -->|partials and finals| Coord
       Coord --> Type[Typed text]
       Coord --> LLM[LLMService<br/><i>Smart / Amend only</i>]
       LLM --> Type

The coordinator orchestrates. Moonshine produces the prose. The
Vosk side channel produces the control phrases. The LLM, if any,
rewrites the accumulated text before typing.

The rest of the chapter unpacks each piece.

The six modes
=============

A session has a *mode* fixed at the moment it starts. All modes
share the same audio path and the same modifier system; they
differ in what they do with the finalized text.

==========  =================================================================
Mode        What it does on stop
==========  =================================================================
Standard    Type each finalized utterance immediately, as it comes in.
Type        Like Standard, plus auto-stop after a configurable silence.
Visual      Hold streaming text in a popup; type the whole block on stop.
Hidden      Like Visual, but never show the streaming text.
Smart       Like Visual, then rewrite via the LLM before typing.
Amend       Like Smart, but rewrite the user's pre-existing selection.
==========  =================================================================

Two of the six (Smart and Amend) enter an extra LLM stage on
stop. The other four go straight back to idle.

The mode is chosen by the *start command* the parser emitted:
``DictationStartCommand`` for Standard,
``DictationSmartStartCommand`` for Smart, and so on. There is no
in-session mode switch.

The state machine
=================

Three states are reachable during a session, plus a one-way
``SHUTTING_DOWN`` for application teardown.

.. mermaid::

   flowchart LR
       Idle[<b>IDLE</b><br/><i>discarding audio</i>] -->|start trigger,<br/>any mode| Rec[<b>RECORDING</b><br/><i>Moonshine streaming</i>]
       Rec -->|stop,<br/>Standard / Type / Visual / Hidden| Idle
       Rec -->|stop,<br/>Smart / Amend| Proc[<b>PROCESSING_LLM</b><br/><i>LLM rewriting</i>]
       Proc -->|LLM completed| Idle

Transitions are validated against an explicit table inside
``DictationCoordinator``
(``vocalance/app/services/dictation_flow/dictation_coordinator.py``):

.. code-block:: python

   VALID_DICTATION_STATE_TRANSITIONS = {
       DictationState.IDLE: frozenset({DictationState.RECORDING, DictationState.SHUTTING_DOWN}),
       DictationState.RECORDING: frozenset({DictationState.PROCESSING_LLM, DictationState.IDLE, DictationState.SHUTTING_DOWN}),
       DictationState.PROCESSING_LLM: frozenset({DictationState.IDLE, DictationState.SHUTTING_DOWN}),
       DictationState.SHUTTING_DOWN: frozenset(),
   }

Anything outside the table raises immediately. The rest of the
coordinator can therefore assume invariants ("at most one active
session", "no LLM run starts before recording ends") without
explicit locking.

----

Audio routing
=============

The coordinator does not own an audio stream. It subscribes to
``AudioChunkCapturedEvent`` like everyone else and forwards each
chunk to Moonshine while a session is active.

.. code-block:: python

   self.subscribe(AudioChunkCapturedEvent, self._handle_audio_chunk)

   def _handle_audio_chunk(self, event):
       self.feed_moonshine_audio_chunk(event.pcm_bytes, event.sample_rate)

The forward is unconditional; the inner moonshine controller
checks state and drops chunks when no recording session is
active.

Moonshine returns transcripts through *line callbacks*. A "line"
is its unit of finalization, roughly a phrase between natural
pauses. The coordinator registers two callbacks per stream.

================  ==================================================================
Callback          Fires when                                  Used for
================  ==================================================================
``on_partial``    Moonshine refines the current line          Popup live text (Visual, Smart, Amend, Hidden)
``on_final``      Moonshine commits a line                    Append to the session transcript
================  ==================================================================

In Standard and Type, only finals matter: each finalized line
publishes a ``DictationTextRecognizedEvent`` and is typed
immediately. In the streaming modes, partials drive the popup
through ``PartialDictationTextEvent`` and finals are accumulated
into a buffer that becomes the input to either direct typing
(Visual / Hidden) or the LLM (Smart / Amend).

Vosk as a side channel
======================

While a session is active, ``CommandSpeechService`` interprets
every Vosk recognition differently. The same command segmenter
keeps cutting clips and Vosk keeps recognizing them, but the
service ignores everything except two things: the configured
*stop trigger* and the configured *modifier phrases*.

.. mermaid::

   flowchart LR
       Vosk[Vosk recognition] --> Q1{Contains<br/>stop trigger?}
       Q1 -->|yes| Stop[Publish<br/>CommandTextRecognizedEvent<br/>→ parser → DictationStopCommand]
       Q1 -->|no| Q2{Contains<br/>modifier phrase?}
       Q2 -->|yes| Mod[Publish<br/>DictationModifierPhraseEvent<br/><i>parser bypassed</i>]
       Q2 -->|no| Drop[Drop]

The stop trigger goes through the parser like any other command,
which is why it can also be activated by a sound mapped to that
phrase. The modifier phrases bypass the parser because they are
not commands; they belong to the dictation system specifically.

----

Stopping a session
==================

A session ends in one of three ways.

#. **The user says the stop trigger.** Vosk → parser →
   ``DictationStopCommand`` → coordinator's command handler.
#. **Type-mode silence.** A background timer started at session
   creation watches the timestamp of the last finalized line. If
   the gap exceeds the configured threshold, the timer triggers
   ``stop_session`` itself.
#. **Application shutdown.** ``DictationCoordinator.shutdown``
   drives any in-flight session to completion.

Whichever path triggered it, the coordinator branches on the
mode at stop time.

.. mermaid::

   flowchart TD
       Stop[stop_session] --> Halt[Halt Moonshine,<br/>collect text]
       Halt --> Mode{Mode?}
       Mode -->|Standard / Type| End1[Set IDLE,<br/>exit popup]
       Mode -->|Visual| End2[Type final text,<br/>exit popup]
       Mode -->|Hidden| End3[Type final text,<br/>no popup]
       Mode -->|Smart / Amend| Proc[Set PROCESSING_LLM,<br/>publish<br/>LLMProcessingStartedEvent]

Smart and Amend do not run the LLM inline at this point. They
publish the started event and wait. The reason is the popup
handshake described below.

----

The modifier system
===================

Modifiers transform the *formatting* of dictated text without
changing its content. ``snake case`` puts subsequent words in
``snake_case``; ``all caps`` upper-cases them; ``spelling``
treats each word as a sequence of letters. A modifier stays
active until the user toggles it off (saying the same phrase) or
until another modifier in the same group replaces it.

Mutual exclusion
----------------

Modifiers belong to two independent groups.

================  ============================================================
Group             Members
================  ============================================================
Casing            ``upper``, ``capitals``, ``camel``, ``snake``, ``kebab``,
                  ``diminish``
Punctuation       ``spelling``, ``strip``
================  ============================================================

Within a group, only one modifier can be active at a time;
activating one removes any other in the same group. The two
groups are independent — a session can be in ``snake`` casing
and ``spelling`` punctuation simultaneously.

The Moonshine suppression window
--------------------------------

When the user says a modifier phrase, two things happen:
``CommandSpeechService`` publishes
``DictationModifierPhraseEvent`` (so the modifier system
reacts), but the same audio is also flowing into Moonshine,
which will eventually emit a final line containing the phrase as
plain prose. Without intervention, the phrase would leak into
the dictated text.

The coordinator solves this with a short suppression window. On
``DictationModifierPhraseEvent``, the controller records a
"suppress until" timestamp ~500 ms in the future:

.. code-block:: python

   def output_suppressed(self) -> bool:
       return time.monotonic() < self.moonshine_suppress_until

While the window is open, every Moonshine partial and final is
discarded. The window is long enough to swallow the phrase and
short enough to keep the next phrase intact. This is the *only*
place where the two recognizers' outputs interact with each
other.

----

Aliases
=======

Aliases are user-configured shorthand expansions: ``insert
address`` emits a stored block, ``insert signature`` emits
another. The substitution is handled by
``DictationAliasService``
(``vocalance/app/services/dictation_flow/dictation_alias_service.py``)
inside the coordinator's segment pipeline.

The substitution is two-pass to interact correctly with
modifiers:

#. The alias trigger is replaced with a placeholder.
#. Modifier-aware post-processing runs on the placeholder
   version, so snake-casing does not snake-case the placeholder.
#. The placeholder is replaced with the alias body.

A snake-cased session does not snake-case the contents of an
alias, even though both went through the same pipeline.

----

The LLM handshake (Smart and Amend)
===================================

Smart and Amend hand the accumulated text to the LLM and replace
the popup's contents with streaming LLM tokens. The popup must
finish its UI swap before the first token arrives, or the first
few tokens render in the wrong widget.

The coordination is a small handshake on the bus.

.. mermaid::

   sequenceDiagram
       participant Coord as DictationCoordinator
       participant Bus as Event bus
       participant Popup as Dictation popup
       participant LLM as LLMService

       Coord->>Bus: LLMProcessingStartedEvent
       Bus->>Popup: deliver
       Popup->>Popup: swap to streaming UI
       Popup->>Bus: LLMProcessingReadyEvent
       Bus->>Coord: deliver
       Coord->>LLM: run streaming
       loop each token
           LLM->>Bus: partial token event
           Bus->>Popup: deliver
       end
       LLM->>Bus: LLMProcessingCompletedEvent
       Bus->>Coord: deliver
       Coord->>Coord: type final text into focused app

Three properties follow:

- The coordinator never drives the popup directly. It publishes
  one event and waits for the popup's reply on the bus.
- Tokens stream straight from the LLM service to the popup,
  without going through the coordinator. The coordinator only
  sees the completed text, which it types as a single block.
- Amend mode adds one extra step *before* recording starts: the
  coordinator copies the user's current selection, stashes it,
  and passes it to the LLM as the text to rewrite. The spoken
  text becomes the *instructions*.

----

Popup as a back-end mirror
==========================

The dictation popup is the most visible piece of front end, but
it owns no state of its own. Every change in the coordinator's
state is published on the bus; the popup's controllers subscribe
and re-render.

================================================  ===============================================
Event                                             Popup reaction
================================================  ===============================================
``DictationStatusChangedEvent``                   Show / hide the popup, set the active mode.
``DictationModifierStateChangedEvent``            Update the active-modifier label.
``PartialDictationTextEvent``                     Update the streaming text widget.
``FinalDictationTextEvent``                       Append a finalized line.
``DictationSessionEvent``                         Drive the visual / smart / amend transitions.
``LLMProcessingStartedEvent``                     Begin the LLM-mode UI swap.
``LLMProcessingCompletedEvent``                   Hide.
================================================  ===============================================

There are no method calls between the coordinator and the popup
in either direction — only events. The popup could be replaced
with a CLI display, or removed entirely, and the coordinator
would not notice.

Where to read next
==================

The user interface — the popup, the main window, the overlays —
is covered in :doc:`user_interface`. After that, the *foundations*
chapters explain how the audio chunks, the LLM tokens, and the
OS input calls actually move between threads and through the
queue without blocking each other.
