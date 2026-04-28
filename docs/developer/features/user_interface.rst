User Interface
##############

.. sectnum::

The command and dictation pipelines described in the previous chapters produce
events. This chapter explains how those events reach the screen and how user
interactions travel in the opposite direction — from widgets back to the bus.

The UI layer renders back-end state and forwards user input. It is built from a
strict three-role architecture that prevents any direct coupling between widgets
and services.

The Three-Role Architecture
============================

Every piece of UI is divided into three roles: a **view**, a **controller**,
and the **bus**. Each role has a clear boundary it cannot cross.

.. mermaid::

   flowchart LR
       Bus((Event bus)) -->|back-end events| Ctrl[Controllers]
       Ctrl -->|Qt signals| View[Views: Qt widgets]
       View -->|method calls| Ctrl
       Ctrl -->|publish events| Bus

Views
------

A view is a pure Qt widget. It draws itself, handles user interactions (clicks,
slider drags, text input), and delegates all logic to its controller. A view
never imports a service, never reads from the bus, and holds no application
state — only presentation state (which tab is selected, which input field has
focus).

Controllers
------------

A controller is the mediator between a view and the back end. It has two
directions of responsibility.

**Back end → view.** The controller subscribes to the bus events its view needs
to react to. When an event arrives, the controller extracts the relevant data
and emits a Qt signal carrying that data. The view connects a slot to the signal
and updates its display. The controller never calls view methods directly; it
always goes through a signal so the view can update on the Qt thread safely.

**View → back end.** When the user interacts with the view, the view calls a
method on the controller. The controller translates that call into a back-end
event and publishes it. The view never knows what service handles the event.

The result: views are unaware of services, services are unaware of widgets, and
the bus is the only shared interface. Adding a new UI surface requires only a
new controller that subscribes and publishes — no changes to any service.

Main Window and Tabs
=====================

Startup and the UI Registry
-----------------------------

A single **UI registry** constructs every controller at startup and registers
each with the lifecycle for teardown. The registry is the only place in the
application where controllers are instantiated. It guarantees every controller
is built exactly once, in a deterministic order, and torn down in the reverse
order at shutdown.

The main window holds five sidebar tabs. Each tab is associated with a view
class and a controller class at registration time, but the view is **not
constructed at startup**. Instead, the widget is built the first time the user
clicks the tab — lazy construction. This spreads cold-start cost across the
session and prevents model-loading services from racing with widget construction
before initialization is complete.

Tab Summary
------------

.. list-table::
   :widths: 20 80
   :header-rows: 1
   :class: uniform-rows

   * - Tab
     - Content
   * - Commands
     - Voice command bindings and automation configuration. Each row
       maps a trigger phrase to an action type and action parameters.
   * - Marks
     - Saved screen positions and their labels. Shows current marks,
       allows manual deletion, and displays the mark overlay toggle.
   * - Dictation
     - Mode selection, LLM rewrite prompt templates (agentic prompts),
       and alias expansions. The alias table maps trigger phrases to
       stored text blocks.
   * - Sounds
     - Trained sound-to-command mappings. Allows the user to train new
       sounds by label, review existing mappings, and retrain or delete
       individual labels.
   * - Settings
     - Global configuration: VAD sensitivity, model selection, silence
       timeout, and other parameters exposed through
       ``RuntimeConfigurationStore``.

Overlay Windows
================

Three floating overlays appear on top of whatever application the user is
currently working in. Each overlay follows the same view / controller / bus
pattern as the tabs.

The Dictation Popup
--------------------

The dictation popup is the most elaborate overlay. It appears when a dictation
session starts and disappears when it ends. In different modes it renders
different content:

- **Standard and Type**: a compact indicator that a session is active.
- **Visual and Hidden**: a transcript view showing finalised lines. Visual shows
  a live partial-text preview between finals; Hidden does not.
- **Smart and Amend**: initially a transcript view, then during LLM generation
  a token-stream view that shows the model's output as it is generated.

The popup also displays a wave-meter bar driven by the RMS amplitude of incoming
audio chunks, giving the user continuous confirmation that the microphone is
active.

All of these states are driven purely by events. The popup controllers subscribe
to ``DictationStatusChangedEvent``, ``PartialDictationTextEvent``,
``FinalDictationTextEvent``, ``LLMProcessingStartedEvent``, and related events.
The popup never reaches back to ``DictationCoordinator``.

The Grid Overlay
-----------------

``GridService`` publishes ``GridStateEvent`` when the grid should appear or
disappear. The overlay controller subscribes to this event and constructs or
destroys the overlay window in response. The overlay renders the cell layout
received in the event and updates the cell numbering when the service publishes
a re-rank event.

The Mark Overlay
-----------------

``MarkService`` publishes ``MarkVisualizationStateChangedEventData`` to control
the mark overlay's visibility and contents. The overlay renders the current set
of marks as labelled circles over their stored screen coordinates.

The System Controller
======================

``QtSystemController`` is a controller with no corresponding view. Its sole
role is to display a modal dialog when ``AudioDeviceErrorEvent`` arrives from
the capture layer — audio-device failures need immediate user attention but
warrant no permanent UI surface. Not every back-end event maps to a tab;
some only need a one-shot notification.

The features chapters have shown how audio flows from the microphone to the OS.
The foundations chapters that follow cover the shared machinery underneath every
feature. Start with :doc:`../foundations/event_bus`, which explains in detail
how the bus delivers events and guarantees ordering.
