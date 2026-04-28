User interface
##############

The previous chapters described what the back end does. The user
interface is what makes the back end visible: the dictation
transcript, the active modifier, the saved marks, the numbered
grid, the trained sounds, the configuration. It also collects
user input — clicks, hotkeys, slider drags — and forwards it to
the back end as events.

This chapter describes the UI architecture, not individual
widgets.

Three roles, one contract
=========================

The UI layer is built from three roles. They communicate only
through the bus and through local Qt signals.

.. mermaid::

   flowchart LR
       Bus((Event bus)) -->|events in| Ctrl[Controllers]
       Ctrl -->|Qt signals| View[Views<br/><i>Qt widgets</i>]
       View -->|method calls| Ctrl
       Ctrl -->|publish| Bus

A **view** is a Qt widget. It draws itself, accepts user input,
and forwards that input to its controller through method calls.
A view never imports a service and never reads from the bus.

A **controller** owns the connection between a view and the back
end. It subscribes to the events the view needs to react to,
translates each event into a Qt signal the view can consume, and
emits the signal. Going the other way: a user interaction calls
a method on the controller, and the controller publishes a
back-end event.

The back end is invisible from the UI's perspective. Controllers
know the *event types* they care about, never which service
emits them.

The result: views never call services, services never touch
widgets, controllers never know how things are drawn.

Main window and tabs
====================

A single **UI registry** constructs every controller at startup
and tears them down at shutdown. The registry guarantees that
every controller is built exactly once, in deterministic order,
and torn down in reverse.

The main window holds five sidebar tabs.

================  ===========================================================
Tab               What it shows
================  ===========================================================
Commands          Voice command bindings.
Marks             Saved screen positions.
Dictation         Modes, agentic prompts, aliases.
Sounds            Trained sound mappings.
Settings          Global configuration.
================  ===========================================================

Tabs build their views *lazily*: a tab's widget is constructed
the first time the user clicks the tab, not at startup.
Cold-start latency is therefore paid in pieces, and a user who
never opens a tab never instantiates that view.

Overlay windows
===============

Three pieces of UI float above whatever application the user is
currently in.

================  ====================================================================
Overlay           Driven by
================  ====================================================================
Dictation popup   ``DictationStatusChangedEvent`` and the dictation streaming events.
Grid overlay      ``GridStateEvent`` from ``GridService``.
Mark overlay      ``MarkVisualizationStateChangedEventData`` from ``MarkService``.
================  ====================================================================

Each overlay is owned by its corresponding controller and follows
the same view / controller / bus pattern as the tabs. The
dictation popup is the most elaborate: it renders the wave
meter, the live transcript, the active modifier label, and (in
Smart and Amend modes) the LLM's streaming output, all as
reactions to events.

The viewless system controller
==============================

One controller in the application has no view of its own:
``QtSystemController``. Its job is to surface global problems —
currently a single audio-device failure event from the capture
layer — as a modal dialog over the main window. Not every
back-end event needs a tab; some only need a one-shot
notification.

Where to read next
==================

The Qt event loop and the asyncio event loop run on the same
operating-system thread, so a controller's reaction to a back-end
event runs on the same thread that draws the widget. The
mechanism that makes that possible — and the rare cases where
work *does* hop between threads — is the subject of
:doc:`../foundations/concurrency`.
