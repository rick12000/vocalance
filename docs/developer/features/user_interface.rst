User interface
##############

The previous chapters described what the back-end does. The user
interface is what makes the back-end visible: it shows the user the
current dictation transcript, the active modifier, the saved marks,
the numbered grid, the trained sounds, the configuration. It also
collects the user's input — clicking a tab, pressing a hotkey,
adjusting a slider — and forwards it to the back-end as events.

This chapter describes the UI architecture. It does *not* cover
individual widgets or layouts; those are implementation details of
each tab and overlay.

The three roles
===============

The UI layer is built from three roles. They are deliberately
separated, and they communicate only through the bus and through
local Qt signals.

.. mermaid::

   flowchart LR
       Bus[Event bus] --> Ctrl[Controllers]
       Ctrl -->|Qt signals| View[Views]
       View -->|user input| Ctrl
       Ctrl -->|publish| Bus

A **view** is a Qt widget. It draws itself, accepts user input, and
forwards that input to its controller through method calls. A view
never imports a service and never reads from the bus.

A **controller** owns the connection between a view and the
back-end. It subscribes to the back-end events the view needs to
react to, translates each event into a Qt signal the view can
consume, and emits the signal. Going the other way: when the user
interacts with the view, the view calls a method on the controller,
and the controller publishes a back-end event on the bus.

The back-end is invisible from the UI's perspective. Controllers
know the *event types* they care about, never which service emits
them.

This split keeps each layer simple: views never call services,
services never touch widgets, and controllers never know how things
are drawn.

The registry and the main window
================================

A single **UI registry** constructs every controller at startup and
tears them down at shutdown. The registry guarantees:

- Every controller is built exactly once.
- Construction order is deterministic.
- Teardown is the reverse of construction.

The main window holds five sidebar tabs:

================  ===========================================================
Tab               What it shows
================  ===========================================================
Commands          Voice command bindings.
Marks             Saved screen positions.
Dictation         Modes, agentic prompts, aliases.
Sounds            Trained sound mappings.
Settings          Global configuration.
================  ===========================================================

Tabs build their views *lazily*: a tab's widget is constructed the
first time the user clicks the tab, not at startup. Cold-start
latency is therefore paid in pieces, and a user who never opens the
Sounds tab never instantiates the Sounds view.

The three overlay windows
=========================

Three pieces of UI live outside the main window. Each one floats
above whatever application the user is currently in.

================  ===================================================================
Overlay           Driven by
================  ===================================================================
Dictation popup   ``DictationStatusChangedEvent`` and the dictation streaming events.
Grid overlay      ``GridStateEvent`` from the grid service.
Mark overlay      ``MarkVisualizationStateChangedEventData`` from the mark service.
================  ===================================================================

Each overlay is owned by its corresponding controller and follows
the same view / controller / bus pattern as the tabs. The dictation
popup is the most elaborate: it renders the wave meter, the live
transcript, the active modifier label, and (in Smart and Amend
modes) the LLM's streaming output, all as reactions to events.

The system controller
=====================

There is one controller in the application without a view of its
own: ``QtSystemController``. Its job is to surface global problems —
currently a single audio-device failure event from the capture
layer — as a modal dialog over the main window. It is a useful
reminder that not every back-end event needs a tab; some only need a
one-shot notification.

Where to read next
==================

The Qt event loop and the asyncio event loop run on the same
operating-system thread, so a controller's reaction to a back-end
event runs on the same thread that draws the widget. The mechanism
that makes that possible — and the rare cases where work *does* hop
between threads — is the subject of :doc:`../foundations/concurrency`.
