Architecture
############

This chapter establishes the vocabulary the rest of the guide uses.
It introduces the application as a pipeline, the unit of code the
pipeline is built from, and the mechanism those units use to talk
to each other. The *features* and *foundations* chapters fill in
the detail.

The pipeline
============

At the top level, Vocalance is a one-direction pipeline from the
microphone to the operating system, with two parallel back-end
flows downstream of capture.

.. mermaid::

   flowchart LR
       Mic[Microphone] --> Cap[Capture]
       Cap --> CF[Command flow]
       Cap --> DF[Dictation flow]
       CF --> OS[OS input<br/><i>click, keypress</i>]
       DF --> Type[Typed text<br/><i>into focused app</i>]

Three properties of this picture matter:

#. **Single input.** The microphone is the only source. The capture
   layer is therefore the single entry point and gets its own
   chapter (:doc:`../features/capture`).
#. **Two flows.** *Command flow* turns short utterances or trained
   sounds into single OS-level actions
   (:doc:`../features/command_flow`). *Dictation flow* turns
   continuous speech into typed text
   (:doc:`../features/dictation_flow`).
#. **One direction.** OS output never loops back into capture.
   Every event has a single, traceable origin.

Zooming into the back end one level reveals which services live
inside each flow and how the event bus sits between them and the
front end.

.. mermaid::

   flowchart LR
       subgraph Capture
           ACS[AudioCaptureService]
       end
       subgraph CommandFlow["Command flow"]
           CSeg[CommandSegmenterService]
           SSeg[SoundSegmenterService]
           CSpeech[CommandSpeechService]
           SRec[SoundService]
           Parser[CentralizedCommandParser]
           Exec[Automation /<br/>Mark / Grid]
       end
       subgraph DictationFlow["Dictation flow"]
           DCo[DictationCoordinator]
           Moon[MoonshineEngine]
           LLM[LLMService]
       end
       Bus((Event bus))
       UI[Qt views<br/>and overlays]
       ACS --> Bus
       Bus --> CSeg --> Bus
       Bus --> SSeg --> Bus
       Bus --> CSpeech --> Bus
       Bus --> SRec --> Bus
       Bus --> Parser --> Bus
       Bus --> Exec
       Bus --> DCo --> Bus
       DCo <--> Moon
       Bus --> LLM --> Bus
       Bus --> UI

Every solid arrow is either a publish onto the bus or a delivery
from it. The only direct ownership relation that bypasses the bus
is ``DictationCoordinator`` ↔ ``MoonshineEngine``: the coordinator
owns the engine because the engine is not itself a service (it has
no bus duties).

Services
========

Each box on the back-end side is a **service**: a regular Python
class with three properties.

- One responsibility (the recorder, the parser, an executor
  family, …).
- An event interface, never direct method calls from outside.
- An explicit lifetime: constructed once at startup, released once
  at shutdown.

The base contract is one abstract class
(``vocalance/app/services/base_service.py``):

.. code-block:: python

   class Service(ABC):
       def __init__(self, event_bus: EventBus) -> None: ...
       def subscribe(self, event_type, handler) -> None: ...
       async def initialize(self) -> bool: ...
       async def shutdown(self) -> None: ...

``initialize`` is for async setup that cannot run in ``__init__``
(loading a heavy model, reading a file). ``shutdown`` releases
resources. ``subscribe`` registers a handler through a tracker
that ``shutdown`` later releases for free
(:doc:`../foundations/event_bus`).

Events
======

The unit of communication between services is an **event**: a
frozen Pydantic model carrying the fields a subscriber needs to
react. Every event derives from ``BaseEvent`` and is named for
*what happened*, not for *what should happen next*.

.. code-block:: python

   class CommandTextRecognizedEvent(BaseEvent):
       text: str
       processing_time_ms: float
       engine: str
       mode: str

Two services interact by one publishing the event and the other
subscribing to its type. Neither knows the other exists.

.. mermaid::

   flowchart LR
       P[Publisher] -->|publish| Bus((Event bus))
       Bus -->|deliver| S1[Subscriber A]
       Bus -->|deliver| S2[Subscriber B]

Three properties of the bus matter even before the mechanics:

- **Subscription is by event type.** A handler asks for *every*
  ``CommandTextRecognizedEvent``; it does not subscribe to a
  particular publisher.
- **Delivery is sequential.** While the bus is delivering one
  event, the next one waits. Causal ordering is a guarantee.
- **Subscriptions clean themselves up.** Services register through
  a tracker; ``shutdown`` releases everything. No service contains
  a manual ``unsubscribe`` call.

The internals — the queue, the worker, the dispatch strategy —
live in :doc:`../foundations/event_bus`.

Front end and back end
======================

Vocalance has a strict split:

- **Back end.** Every service in the pipeline. Owns state, runs
  the models, publishes events. No Qt imports anywhere.
- **Front end.** The Qt windows, tabs, and overlays. Owns no
  back-end state; renders whatever the back end has published.

The contract between the two halves is the bus. The back end
publishes; the front end subscribes and re-renders. There are no
direct references in either direction. The mic-level meter and
the dictation popup are both examples; both are covered in their
respective chapters.

This split is not aesthetic. It means the back end is testable
without instantiating Qt, and a different front end could be
plugged in without touching the back end.

Where to read next
==================

The chapters are arranged in the order audio actually flows
through the system:

#. :doc:`../features/capture` — the microphone end.
#. :doc:`../features/command_flow` — the command path.
#. :doc:`../features/dictation_flow` — the dictation path.
#. :doc:`../features/user_interface` — what the user sees.

The *foundations* chapters explain the machinery underneath.
