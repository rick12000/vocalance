Introduction
############

Vocalance is a desktop voice-control application. It listens to the
microphone, turns what it hears into either a short command or a
stream of dictated text, and acts on the operating system on the
user's behalf. Everything runs locally; no network calls are made
during normal use.

This guide is for developers. It covers architecture, the parts the
architecture is built from, and the conventions that hold them
together. End-user features and installation live elsewhere.

The guide assumes Python, Pydantic, and the publish-subscribe
pattern. It does not assume prior knowledge of asyncio internals,
Qt, or the speech / sound libraries Vocalance uses; each is
introduced where it matters.

Guide layout
============

The chapters are stacked in three layers.

.. mermaid::

   flowchart TB
       O[<b>Overview</b><br/><i>what Vocalance is<br/>and how it is composed</i>]
       F[<b>Features</b><br/><i>each user-facing capability,<br/>end to end</i>]
       I[<b>Foundations</b><br/><i>the systems that make<br/>everything work</i>]
       O --> F --> I

**Overview** (this chapter and :doc:`architecture`) introduces the
vocabulary used throughout the rest of the guide.

**Features** has one chapter per user-facing capability:
:doc:`../features/capture`, :doc:`../features/command_flow`,
:doc:`../features/dictation_flow`, :doc:`../features/user_interface`.
Each tells the end-to-end story of one feature in terms of *what
happens*. Threading and lifecycle questions are deferred.

**Foundations** answers what the feature chapters left open:
:doc:`../foundations/event_bus`,
:doc:`../foundations/concurrency`,
:doc:`../foundations/lifecycle`,
:doc:`../foundations/storage`.

Reading order
=============

Read top to bottom. Every chapter assumes the chapters before it.
For reference, each chapter ends with a pointer to the next.
