Command Flow
############

.. sectnum::

This chapter picks up where :doc:`capture` leaves off. Every
``AudioChunkCapturedEvent`` published by the capture layer is a candidate input
to the command pipeline. The pipeline's job is to turn that raw audio into a
single discrete OS action.

Overview
=========

When an audio chunk arrives, it is passed to a segmenter. The segmenter
accumulates successive chunks into a growing buffer, waiting until a condition
— usually a sustained period of silence — signals that a complete word, phrase,
or sound has ended. It then emits that complete clip as a new bus event. The
clip travels to a recognizer, which converts it to either transcribed text
(speech path) or a named label (sound path). Both paths converge at the parser,
which matches the text or label against a list of pre-configured commands and
publishes a structured command event. An executor receives that event and
produces the final OS input.

.. mermaid::

   flowchart LR
       Chunks[AudioChunkCapturedEvent] --> S1[Segment]
       S1 --> S2[Recognize]
       S2 --> S3[Parse]
       S3 --> S4[Execute]
       S4 --> OS[OS input]

The two recognition paths run in parallel: speech and sound clips arrive
independently on the bus and each follows its own recognizer before converging
at the parser.

.. mermaid::

   flowchart LR
       Chunks[AudioChunkCapturedEvent]
       subgraph Speech["Speech path"]
           CSeg[CommandSegmenterService] -->|CommandAudioSegmentReadyEvent| CSpeech[CommandSpeechService<br/>Vosk]
       end
       subgraph Sound["Sound path"]
           SSeg[SoundSegmenterService] -->|ProcessAudioChunkForSoundRecognitionEvent| SRec[SoundService<br/>YAMNet + k-NN]
       end
       Chunks --> CSeg
       Chunks --> SSeg
       CSpeech -->|CommandTextRecognizedEvent| Parser[CentralizedCommandParser]
       SRec -->|CustomSoundRecognizedEvent| Parser
       Parser -->|*CommandParsedEvent| Exec[Executors]
       Exec --> OS

- **Speech path** — ``CommandSegmenterService`` accumulates chunks into
  utterance clips; ``CommandSpeechService`` runs Vosk offline speech recognition
  to produce a text string.
- **Sound path** — ``SoundSegmenterService`` accumulates chunks into transient
  clips; ``SoundService`` extracts a YAMNet embedding and runs a k-NN vote to
  produce a named label, which is then mapped to a command phrase.

The remainder of this chapter walks through each stage in turn.

Segmentation
=============

The segmenters solve a fundamental mismatch: recognition models need a complete
audio clip, but the capture layer delivers a continuous trickle of 30 ms
buffers. A segmenter bridges this gap by accumulating buffers until it detects
the boundary of an acoustic event, then emitting the collected clip.

Voice Activity Detection
-------------------------

The boundary-detection mechanism is **voice activity detection** (VAD). For
each incoming buffer the segmenter computes its RMS amplitude — a single number
representing the buffer's energy — and compares it to an adaptive threshold.

The threshold adapts to the ambient noise floor: the segmenter tracks a rolling
average of recent silent-buffer energies and multiplies it by a configurable
factor. In a quiet room the threshold is low; in a noisier one it rises
automatically. This means the same vocal effort triggers recognition in either
environment.

The segmenter is stateful. It moves through three states:

.. mermaid::

   flowchart LR
       Idle[Idle<br/>filling pre-roll] -->|buffer energy above threshold| Cap[Capturing<br/>accumulating chunks]
       Cap -->|silence streak long enough| Done[Emit clip]
       Cap -->|duration cap reached| Done
       Done --> Idle

**Idle** — the segmenter continuously fills a short **pre-roll buffer**: a
circular queue holding the last few hundred milliseconds of audio. When a
buffer crosses the threshold, the segmenter immediately prepends the pre-roll
contents to the new clip (capturing the onset that would otherwise be missed)
and enters **Capturing**.

**Capturing** — every subsequent buffer is appended to the clip. The segmenter
exits this state on one of two conditions: a configurable streak of consecutive
below-threshold buffers (the normal end of a phrase), or a hard duration cap
(which prevents unbounded accumulation from continuous speech or noise).

On either exit, the segmenter emits the collected clip and resets to Idle.

CommandSegmenterService
------------------------

``CommandSegmenterService``
(``vocalance/app/services/command_flow/segmenting/command_segmenter_service.py``)
is tuned for spoken utterances. It uses a ~500 ms silence streak and allows
clips up to several seconds. No extra rejection gate is applied — soft-spoken
commands must pass through reliably. Output: ``CommandAudioSegmentReadyEvent``.

SoundSegmenterService
----------------------

``SoundSegmenterService``
(``vocalance/app/services/command_flow/segmenting/sound_segmenter_service.py``)
is tuned for short transients — knocks, snaps, claps. Two adjustments reflect
the different acoustic target:

- **Shorter silence streak (~150 ms)** — transients end abruptly, not with a
  gradual tail.
- **Peak-amplitude gate** — sustained background noise can drift above the RMS
  threshold without ever spiking. The gate rejects steady-state noise that lacks
  a sharp transient peak, reducing false positives.

Additionally, ``SoundSegmenterService`` subscribes to
``DictationModeDisableOthersEvent`` and suspends processing while a dictation
session is active — every spoken word during dictation would otherwise produce a
false-positive sound clip. Output: ``ProcessAudioChunkForSoundRecognitionEvent``.

With a complete audio clip in hand, the next stage is to understand what was
said or heard.

Recognition
============

The two recognizers receive their respective clips independently and publish
their results to the same parser.

Speech Recognition
-------------------

``CommandSpeechService``
(``vocalance/app/services/command_flow/speech_recognition/command_speech_service.py``)
wraps **Vosk**, an offline speech-recognition toolkit built on the Kaldi
acoustic modelling framework. A ~50 MB model is bundled with the application.

The service subscribes to ``CommandAudioSegmentReadyEvent``. When a clip
arrives, it feeds the PCM bytes to Vosk and receives a plain lower-case text
string — no punctuation, no confidence score, no alternatives. Recognition
takes 100–300 ms on typical hardware. To avoid blocking the main thread for
that duration, the service offloads the call to a background thread via
``run_blocking`` and awaits the result asynchronously.

Output: ``CommandTextRecognizedEvent(text, processing_time_ms, engine, mode)``.

Sound Recognition
------------------

``SoundService``
(``vocalance/app/services/command_flow/sound_recognition/sound_service.py``)
classifies transient sounds and maps them to user-defined command phrases via a
two-step pipeline.

**Step 1 — YAMNet embedding.** YAMNet is a neural network pre-trained on
environmental sounds. Given a clip, it produces a fixed 5 120-dimensional
vector — an *embedding* — that encodes the acoustic character of the clip. The
user does not need to fine-tune a neural network; a handful of example
recordings per sound is enough.

**Step 2 — k-NN classification.** The embedding is compared against all stored
embeddings using k-nearest-neighbour voting. Each stored embedding votes for
its label; the label with the most votes wins. New sounds are available
immediately after training — no retraining is needed.

.. mermaid::

   flowchart LR
       Clip[Sound clip] --> Pre[Resample and normalize]
       Pre --> Emb[YAMNet<br/>5120-D embedding]
       Emb --> KNN[k-NN vote<br/>user samples + ESC-50 negatives]
       KNN --> Gate{Winner a<br/>user label?}
       Gate -->|yes| Pub[CustomSoundRecognizedEvent]
       Gate -->|no| Drop[Drop silently]

k-NN has no built-in rejection: it always returns a winner. To prevent every
unrelated noise from being assigned to a user label, the recognizer keeps a
background set of embeddings drawn from **ESC-50**, a public library of 50
environmental sound categories. ESC-50 embeddings participate in the vote; if
one wins, the clip is silently dropped. The ESC-50 set acts as a statistical
floor that absorbs generic noise.

Output: ``CustomSoundRecognizedEvent(label, confidence, mapped_command)``.

Once recognition produces text or a label, the parser maps it to a structured
command.

Parsing
========

``CentralizedCommandParser``
(``vocalance/app/services/command_flow/parsing/parser.py``) subscribes to both
``CommandTextRecognizedEvent`` and ``CustomSoundRecognizedEvent``. For sound
events, the ``mapped_command`` phrase replaces the raw label before any
processing — from this point forward the parser operates on text regardless of
origin.

The parser tries each command family in a fixed priority order. The first family
that matches claims the input; the remaining families are skipped. Priority
ordering prevents any family from shadowing commands in a higher-priority one.

.. list-table::
   :widths: 20 80
   :header-rows: 1
   :class: uniform-rows

   * - Family
     - Priority rationale
   * - **System** — ``pause``, ``resume``
     - Highest priority. No user-configured command can shadow these.
   * - **Dictation triggers** — ``dictate``, ``type``, ``smart``, ``visual``,
       ``hidden``, ``amend``, ``stop``
     - Before marks. Single-word triggers like "type" must be claimed here or
       a mark named "type" would shadow them.
   * - **Marks** — ``mark <label>``, ``visualize``, ``reset``, label fallback
     - Before grid. A mark named "five" must still work when a grid is shown.
   * - **Grid** — ``grid show``, ``grid hover``, ``grid drag``, cell numbers
     - After marks to avoid shadowing named marks.
   * - **Automation** — all user-configured actions by their trigger phrase
     - Before the mark fallback.
   * - **Mark fallback** — any unmatched single word
     - Lowest priority. Any single word that matches nothing else is treated
       as a mark-execute command.

Inputs that match no family are silently dropped.

Output Events
--------------

A matched command is packaged into a typed event and published for the execution
layer. Each family produces a distinct event type carrying a structured command
value object:

.. list-table::
   :widths: 40 60
   :header-rows: 1
   :class: uniform-rows

   * - Family matched
     - Event published
   * - System
     - ``SystemControlCommandParsedEvent``
   * - Dictation
     - ``DictationCommandParsedEvent``
   * - Mark
     - ``MarkCommandParsedEvent``
   * - Grid
     - ``GridCommandParsedEvent``
   * - Automation
     - ``AutomationCommandParsedEvent``

``DictationCommandParsedEvent`` is routed to ``DictationCoordinator`` and kicks
off the dictation flow described in the next chapter. The remaining four event
types feed the executors below.

Execution
==========

All executors that produce OS input route their calls through
``KeyboardInputService``, which holds an asyncio lock that serializes every OS
call into a strict FIFO queue. Without it, three quickly-spoken commands could
each dispatch to the OS on separate background threads and arrive in
unpredictable order.

Automation
-----------

``AutomationService`` handles user-configured OS actions: pressing a hotkey,
typing a key sequence, clicking, scrolling. It dispatches on the command's
``action_type``, supports counted repetition (e.g. "scroll down five"), and
applies a per-action cooldown to prevent double-execution from rapid-fire
recognitions. Large scroll amounts are stepped into small increments to match
what the OS expects from hardware scroll events.

Marks
------

``MarkService`` maps short user-defined labels to screen positions and clicks
them on command. Marks persist via ``StorageService``. Cursor coordinates are
captured at *parse time* — before the event is published — so the position
reflects exactly where the cursor was when the user spoke.

Grid
-----

``GridService`` divides the screen into a numbered grid and lets the user click
any cell by speaking its number. Cells are ranked by historical click frequency
so the most common targets receive the lowest numbers. Ranking is maintained by
``ClickTrackerService``, which batches rapid clicks and debounces disk writes to
avoid one file write per click.

Pause / Resume
---------------

``PauseStateManager`` owns a single boolean flag that the parser reads on every
input. While paused, every parsed command is dropped except ``ResumeCommand``.
