Introduction
############

Vocalance is a desktop voice-control application. It listens to the
microphone, converts what it hears into either a short command or a
stream of dictated text, and acts on the operating system on the
user's behalf. The whole system runs locally — no cloud, no network
calls during normal use.

What this guide covers
======================

This is the developer guide. It explains the architecture, the parts
the architecture is built from, and the conventions that hold them
together. It does *not* cover end-user features or installation;
those live elsewhere.

The guide assumes:

- Comfort with Python, type hints, and Pydantic.
- Basic familiarity with the publish-subscribe pattern.
- No prior knowledge of Vocalance, asyncio internals, Qt, or any of
  the speech / sound libraries it uses. Each is introduced where it
  matters.

How the guide is organized
==========================

The chapters are arranged in three layers, each one zooming in on
the one above.

.. mermaid::

   flowchart TB
       O[Overview<br/><i>what Vocalance is, how it is composed</i>]
       F[Features<br/><i>each user-facing capability, end to end</i>]
       I[Foundations<br/><i>the systems that make all of it work</i>]
       O --> F
       F --> I

**Overview.** :doc:`introduction` (this page) and :doc:`architecture`.
After these two chapters you will know what Vocalance does, what its
moving parts are, and the vocabulary the rest of the guide uses.

**Features.** One chapter per user-facing capability:
:doc:`../features/capture`, :doc:`../features/command_flow`,
:doc:`../features/dictation`, :doc:`../features/user_interface`. Each
chapter tells the end-to-end story of a single feature, written at the
level of *what happens*, not *how it is scheduled*. Concurrency,
threading, and lifecycle questions are deliberately deferred.

**Foundations.** :doc:`../foundations/event_bus`,
:doc:`../foundations/concurrency`, :doc:`../foundations/lifecycle`,
:doc:`../foundations/storage`. These chapters answer the questions
the feature chapters left open: how the bus actually delivers events,
which thread runs what, how the application starts and stops cleanly,
and where state lives on disk.

Reading order
=============

Read top to bottom. Every chapter assumes the chapters before it and
introduces the vocabulary it needs the first time it uses it. If you
come back later for reference, the index at the end of each chapter
points to the related chapters around it.
