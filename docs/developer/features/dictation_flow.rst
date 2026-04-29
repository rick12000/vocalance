Dictation Flow
##############

.. sectnum::

The command flow described in :doc:`command_flow` handles short, discrete
inputs. Dictation is the other consumer of ``AudioChunkCapturedEvent``: a
long-running session that transcribes continuous speech, optionally rewrites
it with a local language model, and types the result. Sessions can last
anywhere from a few seconds to several minutes.

The Session Model
==================

The key design choice is that dictation uses **two separate recognizers running
in parallel**. One produces the dictated prose; the other watches for control
words (stop trigger, formatting modifiers). The outputs of both feed into a
single coordinator that decides what to do with them.

This separation exists because the two recognition tasks have opposite
requirements. Dictation requires a model that handles continuous natural speech,
manages its own phrase boundaries, and produces a stream of progressively refined
partial results. Command recognition requires a model that is fast, deterministic,
and precise on a small vocabulary of exact phrases. No single speech model is
optimally suited to both; using two specialised ones is simpler than compromising
with a single general one.

.. mermaid::

   flowchart LR
       Cmd[DictationCommandParsedEvent] --> Coord[DictationCoordinator]
       subgraph Audio["Audio routing"]
           Chunks[AudioChunkCapturedEvent] --> Coord
           Chunks --> CSpeech[CommandSpeechService<br/>Vosk side channel]
       end
       subgraph Recognition["Recognition"]
           Moon[MoonshineEngine<br/>streaming dictation]
           Coord --> Moon
           Moon -->|partials and finals| Coord
           CSpeech -->|stop trigger or<br/>modifier phrase| Coord
       end
       subgraph Output["Output"]
           Coord --> LLM[LLMService<br/>Smart / Amend]
           Coord --> Type[Typed text]
           LLM --> Type
       end

``DictationCoordinator`` is the central hub. It receives the start command from
the parser, feeds audio to the Moonshine streaming engine, monitors the Vosk
side channel, and applies all text transformations before typing the final result.

The Speech Engine: Moonshine
-----------------------------

The primary recognizer for dictation is **Moonshine**, an offline speech-to-text
model optimised for streaming inference on consumer hardware. Unlike the Vosk
model used for commands — which waits for a complete audio clip — Moonshine
receives a continuous stream of PCM samples and produces results in two forms:

- **Partials**: the engine's current best guess for the phrase it is hearing,
  updated frequently as more audio arrives. A partial can change character
  by character as the user speaks.
- **Finals**: a phrase the engine has decided is complete, typically bounded by
  a natural pause in speech. Once a phrase is finalised, Moonshine moves on
  and the text is fixed.

The application enqueues each audio chunk into the Moonshine session via
``MoonshineStreamSession.add_audio_pcm16``. The call is non-blocking; a
dedicated worker thread inside the session drains the queue in batches and
runs the native ``add_audio_to_stream`` and ``update_transcription`` calls.
This off-load matters because a continuously-spoken segment causes the native
streaming decoder to re-decode from scratch on every partial refresh — the
cost grows with segment duration, and on weaker hardware a single refresh on a
long segment can stall for hundreds of milliseconds. With the worker thread
the asyncio loop, the popup, and the modifier suppression timer all remain
responsive regardless of how long a refresh takes.

Segment boundaries are owned entirely by the Moonshine native VAD. The library
breaks audio into lines at natural pauses (configurable via ``vad_threshold``
and ``vad_window_duration`` in ``MoonshineStreamingConfig``) and force-closes
any line that grows past ``vad_max_segment_duration``, which puts a hard
ceiling on per-refresh decoder cost without relying on any Python-side
rotation. The streaming session therefore feeds audio into a single long-lived
native stream for the duration of the dictation; running the C VAD as the only
segmenter avoids races that would otherwise split a single utterance across two
streams and produce duplicated or missing words at the boundary. All tunable
parameters (thresholds, window durations, segment caps, look-behind, etc.) are
exposed in ``MoonshineStreamingConfig`` with documented defaults and ranges.

When the session is stopped, the worker drains all queued chunks into the
native stream. ``MoonshineStreamSession.stop`` then deactivates the native VAD
directly via the C API, issues a forced ``update_transcription`` to capture any
tail audio, and finally closes the stream. This order is deliberate: the Python
``Stream.stop()`` wrapper calls ``update_transcription`` a second time
internally, which would re-emit ``LineCompleted`` for every already-finalized
segment and produce duplicate output. Bypassing the wrapper and calling the C
stop primitive directly avoids that second emission entirely.

The Vosk Side Channel
----------------------

``CommandSpeechService`` — the same service used for command recognition —
continues to receive audio clips from ``CommandSegmenterService`` while
dictation is active. During a session, it ignores all recognitions except two
specific patterns: the configured **stop trigger** phrase and any **modifier
phrases**. Everything else is silently dropped.

The stop trigger travels through the parser like a normal command
(``→ DictationStopCommand``), which is why it can also be activated by a sound
mapped to the stop phrase. Modifier phrases bypass the parser entirely; they
carry no command meaning and are delivered as ``DictationModifierPhraseEvent``
directly to the coordinator.

Session Modes and the State Machine
=====================================

Session Modes
--------------

A **mode** is fixed when the session starts and cannot change during the session.
All six modes share the same audio routing and the same modifier and alias
system; they differ only in what happens to the recognised text during and after
the session.

The fundamental split is between **immediate** modes (Standard and Type), which
type each finalised Moonshine phrase as soon as it is committed, and **buffered**
modes (Visual, Hidden, Smart, Amend), which hold the text in a popup buffer and
act on the complete transcript when the session stops.

.. list-table::
   :widths: 15 85
   :header-rows: 1
   :class: uniform-rows

   * - Mode
     - Behaviour
   * - Standard
     - Type each Moonshine final immediately as it is committed. The user
       sees text appearing word-by-word as they speak.
   * - Type
     - Identical to Standard with one addition: a configurable silence timer.
       If no Moonshine final arrives within the timeout, the session stops
       automatically. Useful for dictating into fields where a stop command
       would be disruptive.
   * - Visual
     - Buffer all finals in the popup. Display streaming partials as live
       preview text. On stop: type the complete buffer as a single block.
   * - Hidden
     - Same as Visual but partials and finals are not shown in the popup —
       the user cannot see what was captured until it is typed on stop.
   * - Smart
     - Same as Visual but on stop: pass the buffer to the local LLM for
       rewriting before typing. The LLM can fix grammar, change tone, or
       apply any transformation defined by the user's rewrite prompt.
   * - Amend
     - Same as Smart with a different input: instead of the session
       transcript, the LLM receives the user's *currently selected text* as
       the text to rewrite. The spoken dictation becomes the rewrite
       instruction (e.g. "make this more formal").

The State Machine
------------------

The coordinator tracks three active states plus ``SHUTTING_DOWN``.

.. mermaid::

   flowchart LR
       Idle[IDLE] -->|start command, any mode| Rec[RECORDING<br/>Moonshine active]
       Rec -->|stop — Standard, Type, Visual, Hidden| Idle
       Rec -->|stop — Smart or Amend| Proc[PROCESSING_LLM<br/>LLM rewriting]
       Proc -->|LLM done| Idle

State transitions are validated against an explicit table. Any invalid
transition raises immediately, which lets the rest of the coordinator rely on
invariants ("at most one active session", "no LLM run while recording") without
defensive checks:

.. code-block:: python

   VALID_DICTATION_STATE_TRANSITIONS = {
       DictationState.IDLE: frozenset({
           DictationState.RECORDING, DictationState.SHUTTING_DOWN
       }),
       DictationState.RECORDING: frozenset({
           DictationState.PROCESSING_LLM,
           DictationState.IDLE,
           DictationState.SHUTTING_DOWN,
       }),
       DictationState.PROCESSING_LLM: frozenset({
           DictationState.IDLE, DictationState.SHUTTING_DOWN
       }),
       DictationState.SHUTTING_DOWN: frozenset(),
   }

Audio Routing
==============

Routing to Moonshine
---------------------

The coordinator subscribes to ``AudioChunkCapturedEvent`` and forwards each
chunk to the Moonshine streaming engine while a session is in the ``RECORDING``
state:

.. code-block:: python

   self.subscribe(AudioChunkCapturedEvent, self._handle_audio_chunk)

   def _handle_audio_chunk(self, event):
       self.feed_moonshine_audio_chunk(event.pcm_bytes, event.sample_rate)

The forward is unconditional at the coordinator level. The inner Moonshine
controller checks the session state and drops chunks when no session is active,
so the coordinator does not need to gate on state here.

Moonshine surfaces its output through two callbacks registered per stream:

- ``on_partial(text)`` — called frequently while the user is mid-phrase. The
  text is Moonshine's current best transcription and may change on every call.
  In buffered modes (Visual, Smart, Amend, Hidden) this drives the live preview
  text in the popup.
- ``on_final(text)`` — called when Moonshine commits a phrase. In Standard and
  Type modes, this text is typed immediately via ``KeyboardInputService``. In
  buffered modes, it is appended to the session buffer.

The Vosk Side Channel During Dictation
----------------------------------------

While ``RECORDING``, ``CommandSpeechService`` continues to receive Vosk
recognitions because the command segmenter does not stop running during
dictation. The service filters its output: during a session, only two categories
of Vosk output are acted on.

.. mermaid::

   flowchart LR
       Vosk[Vosk recognition] --> Q1{Stop trigger<br/>present?}
       Q1 -->|yes| Stop[CommandTextRecognizedEvent<br/>→ parser → DictationStopCommand]
       Q1 -->|no| Q2{Modifier phrase<br/>present?}
       Q2 -->|yes| Mod[DictationModifierPhraseEvent<br/>direct to coordinator]
       Q2 -->|no| Drop[Drop]

The stop trigger goes through the full parser pipeline, which makes it
indistinguishable from any other stop command — including one triggered by a
mapped sound. Modifier phrases bypass the parser because they are not commands;
they are formatting instructions specific to the dictation system.

Text Shaping
=============

Between Moonshine committing a final phrase and that phrase reaching the output,
two transformations can be applied: modifiers reformat the text, and aliases
substitute known trigger phrases with stored expansions.

Modifiers
----------

A **modifier** changes the formatting of all subsequent text without altering
its content. Modifiers are toggled by speaking a modifier phrase during the
session (e.g. ``snake case``). Speaking the same phrase again toggles the
modifier off; speaking a different modifier in the same exclusion group replaces
the active one.

Modifiers belong to two independent exclusion groups:

.. list-table::
   :widths: 25 75
   :header-rows: 1
   :class: uniform-rows

   * - Group
     - Members
   * - Casing
     - ``upper``, ``capitals``, ``camel``, ``snake``, ``kebab``, ``diminish``
   * - Punctuation
     - ``spelling``, ``strip``

Within a group, at most one modifier can be active at a time. Activating a new
casing modifier deactivates any previous casing modifier. The two groups are
independent — a session can simultaneously be in ``snake`` casing and
``spelling`` punctuation mode.

When the user speaks a modifier phrase, both recognizers process the same audio.
Vosk picks up the phrase and fires ``DictationModifierPhraseEvent`` so the
modifier activates. Moonshine, processing the same audio slightly later, will
also produce a final that contains the modifier phrase as plain text. Without
intervention, that plain text would appear in the dictated output.

The coordinator prevents this with a **suppression window**: on receiving
``DictationModifierPhraseEvent`` it sets a "suppress until" timestamp ~500 ms
in the future. While the window is open, every Moonshine partial and final is
discarded. The window is long enough to swallow the phrase and short enough that
the next phrase begins cleanly:

.. code-block:: python

   def output_suppressed(self) -> bool:
       return time.monotonic() < self.moonshine_suppress_until

This is the only point where the outputs of the two recognizers are coupled.

Aliases
--------

An **alias** is a short trigger phrase that expands to a longer stored text
block. For example, ``insert address`` might expand to a full postal address.
Aliases are configured in the Dictation tab and handled by ``DictationAliasService``
(``vocalance/app/services/dictation_flow/dictation_alias_service.py``).

Substitution runs as a two-pass process to interact correctly with active
modifiers. Without the two-pass approach, a modifier like ``snake_case`` would
also apply to the *contents* of the alias body, which is almost never what the
user wants.

1. The alias trigger phrase in the text is replaced with a unique placeholder.
2. Modifier post-processing runs on the surrounding text, including the
   placeholder. The placeholder itself is immune to casing transformation.
3. The placeholder is replaced with the alias body verbatim.

Stopping a Session
===================

A session can end in three ways:

1. **Stop trigger.** The user speaks the stop phrase. Vosk's side channel
   delivers it as ``DictationStopCommand`` via the parser.
2. **Type-mode timeout.** In Type mode only: a background timer monitors the
   timestamp of the last Moonshine final. If the silence gap exceeds the
   configured threshold, the timer calls ``stop_session`` directly.
3. **Application shutdown.** ``DictationCoordinator.shutdown`` drives any
   in-flight session to completion before releasing resources.

In all three cases, ``stop_session`` branches on the session's mode:

.. mermaid::

   flowchart TD
       Stop[stop_session] --> Halt[Halt Moonshine, collect buffer]
       Halt --> Mode{Mode?}
       Mode -->|Standard / Type| End1[Set IDLE, exit popup]
       Mode -->|Visual| End2[Type buffer, exit popup]
       Mode -->|Hidden| End3[Type buffer, no popup shown]
       Mode -->|Smart / Amend| Proc[Set PROCESSING_LLM<br/>publish LLMProcessingStartedEvent]

Smart and Amend do not invoke the LLM inline at this point. They publish
``LLMProcessingStartedEvent`` and suspend, waiting for the popup to confirm that
its UI transition is complete before the first token arrives.

LLM Rewriting
==============

Smart and Amend pass the session buffer to ``LLMService`` (which wraps
``llama-cpp-python``) for rewriting. The rewritten text replaces the popup
content as it streams in token by token, and is typed as a single block when
generation completes.

The coordination problem is timing: the popup must complete its UI swap (from
transcript view to token-stream view) *before* the first token arrives.
Publishing tokens to a popup that is still mid-transition causes tokens to
render in the wrong widget. The coordinator solves this with an explicit
readiness exchange on the bus.

.. mermaid::

   flowchart LR
       Coord[DictationCoordinator]
       Popup[Dictation popup]
       LLM[LLMService]
       App[Focused app]

       Coord -->|1. LLMProcessingStartedEvent| Popup
       Popup -->|2. complete UI swap<br/>LLMProcessingReadyEvent| Coord
       Coord -->|3. start streaming inference| LLM
       LLM -.->|4. LLMPartialTokenEvent<br/>per token| Popup
       LLM -->|5. LLMProcessingCompletedEvent| Coord
       Coord -->|6. type complete text| App

Steps in detail:

1. The coordinator publishes ``LLMProcessingStartedEvent`` and suspends on the
   bus, awaiting the popup's reply.
2. The popup receives the event, swaps to token-stream view, and publishes
   ``LLMProcessingReadyEvent``.
3. The coordinator resumes and calls ``LLMService`` to begin streaming
   inference. The LLM receives the session buffer as input (Smart) or the
   selection as input plus the spoken text as instructions (Amend).
4. As the model generates tokens, ``LLMService`` publishes
   ``LLMPartialTokenEvent`` for each one. These go directly from the LLM
   service to the popup — the coordinator is not in this path.
5. When generation completes, ``LLMService`` publishes
   ``LLMProcessingCompletedEvent`` with the full rewritten text.
6. The coordinator receives the completed event and types the full text as a
   single block into the focused application via ``KeyboardInputService``.

The coordinator never addresses the popup directly. It publishes an event and
awaits a reply on the bus. This means the popup is entirely optional — the
coordinator would function identically with no popup present, typing the result
after generation finishes.

**Amend vs Smart.** Before step 1, Amend performs one extra action: it copies
the user's currently selected text (the text to rewrite), stashes it, and passes
it to the LLM alongside the dictated instructions. The ``LLMService`` receives
both pieces and formulates the rewrite accordingly. From step 1 onward, the flow
is identical to Smart.

The Popup as a State Mirror
============================

The dictation popup owns no state. Everything it displays is a reflection of
events the coordinator publishes. The popup's controllers subscribe and re-render
on each event; there are no direct method calls between the coordinator and the
popup in either direction.

.. list-table::
   :widths: 45 55
   :header-rows: 1
   :class: uniform-rows

   * - Event
     - Popup reaction
   * - ``DictationStatusChangedEvent``
     - Show or hide the popup; display the active mode label.
   * - ``DictationModifierStateChangedEvent``
     - Update the active modifier indicator.
   * - ``PartialDictationTextEvent``
     - Replace the live preview text.
   * - ``FinalDictationTextEvent``
     - Append a committed line to the transcript.
   * - ``LLMProcessingStartedEvent``
     - Begin the UI swap to token-stream view.
   * - ``LLMPartialTokenEvent``
     - Append the new token to the streaming display.
   * - ``LLMProcessingCompletedEvent``
     - Hide the popup.

The coordinator could run without any popup present, and a different popup
implementation could be substituted without changing a line of coordinator code.

With both pipeline chapters covered, the next chapter — :doc:`user_interface` —
describes how the Qt front end renders this back-end state and how the
three-role architecture keeps views and services from coupling directly.
