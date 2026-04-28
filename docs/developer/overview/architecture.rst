Architecture
############

This chapter establishes the vocabulary the rest of the guide uses.
It introduces the application as a pipeline, the unit of code the
pipeline is built from, and the mechanism those units use to talk to
each other. Detail is kept light on purpose; the *features* and
*foundations* chapters fill it in later.

The pipeline
============

At the highest level, Vocalance is a one-direction pipeline from the
microphone to the operating system.

.. mermaid::

   flowchart LR
       Mic[Microphone] --> Cap[Capture]
       Cap --> Cmd[Commands]
       Cap --> Dict[Dictation]
       Cmd --> OS1[OS input]
       Dict --> Type[Typed text]

There are three observations to make about that picture:

1. **The microphone is the only input.** No other source — keyboard,
   network, file system — can drive the pipeline. The capture layer
   is therefore the single entry point and gets its own chapter.
2. **Two paths leave the capture layer.** A command path that turns
   short utterances or trained sounds into single OS-level actions,
   and a dictation path that turns continuous speech into typed text.
   Each path is a self-contained feature with its own chapter.
3. **The pipeline is one-way.** Output from the OS does not loop back
   into the capture layer. This is what makes the architecture easy
   to reason about — every event has a single, traceable origin.

Services
========

The pipeline is built from objects called **services**. A service is
a regular Python class with three properties:

- It owns one well-defined responsibility (the audio recorder, the
  parser, a single executor family, …).
- It exposes its capabilities through an event interface, not through
  direct method calls from outside.
- It has an explicit lifetime: it is constructed once at startup and
  released once at shutdown.

The base contract is a single abstract class, ``Service``
(``vocalance/app/services/base_service.py``):

.. code-block:: python

   class Service(ABC):
       def __init__(self, event_bus: EventBus) -> None: ...
       def subscribe(self, event_type, handler) -> None: ...
       async def initialize(self) -> bool: ...
       async def shutdown(self) -> None: ...

Every service in the application — and there are roughly twenty of
them — derives from this class. The ``initialize`` hook is for async
setup work that cannot run in ``__init__`` (loading a heavy model,
reading a file). The ``shutdown`` hook releases resources. The
``subscribe`` helper is the mechanism services use to declare which
events they care about.

Events
======

The unit of communication between services is an **event**: a frozen
Pydantic model carrying the fields a subscriber needs to react. Every
event in the application derives from a common ``BaseEvent`` and is
named for what happened, not for what should happen next.

A typical event definition is small:

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
       P[Publisher service] -->|publish| Bus[Event bus]
       Bus -->|deliver| S1[Subscriber service]
       Bus -->|deliver| S2[Another subscriber]

The bus that ferries events between them has three properties worth
holding in mind, even before its mechanics are explained:

- **Publishing is by event type.** A subscriber asks for *every*
  ``CommandTextRecognizedEvent`` ever published; it does not
  subscribe to a particular publisher.
- **Delivery is sequential.** While the bus is delivering one event
  to its subscribers, the next event waits. Causal ordering is a
  guarantee, not an accident.
- **Subscriptions clean themselves up.** Services register handlers
  through a tracker recorded at construction; ``shutdown`` releases
  them all in one call. There is no manual ``unsubscribe`` anywhere
  in the feature code.

The internals of the bus — the queue, the worker, the dispatch
strategy — are covered in :doc:`../foundations/event_bus`. For the
rest of the *features* chapters, "the service publishes X" and "the
other service subscribes to X" is enough.

Front-end and back-end
======================

Vocalance has a clean split between two halves:

- **Back-end.** Every service in the pipeline. Owns state, runs the
  models, publishes events. Has no Qt imports anywhere.
- **Front-end.** The Qt windows, tabs, and overlays. Owns no state of
  its own; renders whatever the back-end has published.

The contract between the two halves is the event bus. The back-end
publishes events that describe what happened; the front-end
subscribes and re-renders. There are no direct references in either
direction. The capture layer's UI mic-level meter is one example, the
dictation popup is another; both are covered in their respective
chapters.

This split is not aesthetic. It means the back-end can be tested
without instantiating Qt, and a different front-end could be plugged
in without touching the back-end.

Where to read next
==================

You now have enough vocabulary to read any feature chapter in any
order. The recommended sequence, however, is the order audio actually
flows through the system:

1. :doc:`../features/capture` — the microphone end of the pipeline.
2. :doc:`../features/commands` — the command path.
3. :doc:`../features/dictation` — the dictation path.
4. :doc:`../features/user_interface` — what the user sees.

Once those are clear, the *foundations* chapters explain the
machinery underneath.
