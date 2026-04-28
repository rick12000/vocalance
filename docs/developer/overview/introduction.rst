Overview
########

.. sectnum::

Vocalance is an offline desktop voice-control application. It listens
continuously to the microphone and, depending on what it hears, either executes
a discrete OS action or produces a stream of typed text.

The Audio Pipeline
==================

At the top, Vocalance is a one-direction pipeline from the microphone to the
operating system.

.. mermaid::

   flowchart LR
       Mic[Microphone] --> Cap[Capture]
       Cap --> CF[Command flow]
       Cap --> DF[Dictation flow]
       CF --> OS[OS input<br/>click, keypress]
       DF --> Type[Typed text]

Audio enters at a single point: the capture layer records the microphone and
broadcasts each buffer of raw sound to the rest of the application. Two
independent pipelines consume that stream:

- **Command flow** — classifies audio as either spoken text or a trained sound,
  maps the result to a pre-configured action, and executes it as an OS input
  event. See :doc:`../features/command_flow`.
- **Dictation flow** — opens a transcription session that converts continuous
  speech into text, optionally rewrites it with a local language model, and
  types the result into the focused application. See
  :doc:`../features/dictation_flow`.

Start with :doc:`../features/capture` to understand how audio enters the system.

Services
=========

Each stage of the pipeline is implemented by one or more **services**. A
service is a self-contained Python class with a single responsibility: it starts
up, does its job for the life of the application, and shuts down cleanly. It
does not know the identities of the other services it collaborates with.

The base class every service implements:

.. code-block:: python

   class Service(ABC):
       async def initialize(self) -> bool: ...
       async def shutdown(self) -> None: ...

``initialize`` handles any one-time setup that cannot run in ``__init__`` —
loading a model file, opening a device stream, reading configuration from disk.
``shutdown`` releases those resources in the reverse order they were acquired.
Services are constructed in a fixed order at startup and torn down in the
reverse order at shutdown.

The Event Bus
==============

Services communicate exclusively by publishing and subscribing to **events** on
a shared bus. An event is a plain data object describing something that just
happened, named for the fact, not for what should happen next.

.. mermaid::

   flowchart LR
       A[Service A] -->|publish event| Bus((Event bus))
       Bus -->|deliver to subscribers| B[Service B]
       Bus -->|deliver to subscribers| C[Service C]

When a service finishes a task — recognizing a word, emitting an audio clip,
completing an LLM response — it packages the result into a typed event and
publishes it. Any number of other services that care about that result subscribe
to that event type and receive it. Neither publisher nor subscriber holds a
reference to the other. This has two practical consequences:

- The same raw audio chunk consumed by the command pipeline is simultaneously
  visible to the dictation pipeline and the UI wave meter, with no additional
  wiring.
- Every service is independently testable by publishing events directly in a
  test, without constructing the services that would normally produce them.

Front End and Back End
=======================

The bus is the only boundary between the back-end services and the front-end
user interface. Back-end services never import Qt; the UI never holds back-end
state. The UI subscribes to events, renders what they contain, and publishes
events in response to user input. Both sides see only the events — never each
other's internals.

This split means the entire processing pipeline can be exercised in a headless
test environment, and a new UI surface can be added without touching a single
service.
