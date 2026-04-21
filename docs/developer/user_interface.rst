User Interface Architecture
############################

Vocalance's user interface is built with PySide6 (Qt framework) and follows a separation of concerns through the **Control-View pattern**, where controllers act as mediators between the view layer and backend services through an event-driven architecture. This document explains how the UI is organized, how user interactions flow through the system, and how the UI remains responsive across multiple threads.

The Core Pattern: Controllers Orchestrating Views and Services
==============================================================

The Vocalance UI operates on a simple principle: **views never call services directly**. Instead, controllers intercede, translating user interactions into events published to the event bus, and subscribing to service responses to update the UI. This creates a clean, testable, and decoupled architecture.

When a user interacts with the UI—clicking a button, entering text, or selecting an item—the view's callback fires. The view passes control to the controller, which then publishes an event describing the user's intent (e.g., "MarkUiRequestEvent" when creating a mark). The event bus routes this to the backend service handling marks. When the service completes the operation, it publishes a completion event (e.g., "MarkUiResponseEvent" or "MarksChangedEventData"). The controller, which has subscribed to this event, receives it and updates the view accordingly.

This pattern ensures views are pure presentation logic—they know nothing about business logic or services. Controllers are thin orchestrators—they know about events and service workflows but contain no UI rendering code. Services implement the actual domain logic and publish events describing what happened. The separation means each layer can be tested independently, modified without cascading changes, and understood easily.

The Main Window: VocalanceMainWindow
====================================

The root of the UI hierarchy is the ``VocalanceMainWindow`` (inheriting from QMainWindow), which manages the main window containing a sidebar for navigation and a content area that displays one active tab at a time. The VocalanceMainWindow is responsible for creating all controllers and views, orchestrating tab switching, and maintaining the specialized overlay windows (grid overlay, mark visualization, dictation popup).

**Window Layout**: The main window is divided into a narrow sidebar with icon buttons and a large content area. The sidebar buttons correspond to tabs: Marks, Commands, Dictation, Sounds, and Settings. Each button switches the content area to the corresponding view. The window is designed for touch and voice interaction, with large buttons and high-contrast visuals.

**Lazy-Loaded Views**: Views are created on demand. The ``VocalanceMainWindow`` uses a ``UiRegistry`` to instantiate controllers upfront, but maintains a cache (``_tab_views``) and creates each view only when the user first navigates to it. This reduces startup time and memory usage. When switching to a tab, VocalanceMainWindow checks if the view is cached. If not, it creates the view and binds its associated controller. If already cached, it reuses the existing view. Only one view is displayed at a time—switching tabs hides the current view and displays the target view using Qt's ``QStackedWidget``.

This lazy loading design means the application starts quickly even though it supports several different functional areas. The first access to any tab incurs a small delay as the view initializes, but subsequent accesses are instant.

Controllers: The Coordination Layer
====================================

Each functional area has a dedicated controller that inherits from ``QtBaseController`` (which inherits from ``QObject``). The controller's responsibility is singular and focused: translate user actions into events, handle service responses, and coordinate view updates.

**Event Subscriptions**: During initialization, a controller subscribes to the event types it cares about. For example, ``QtMarksController`` subscribes to ``MarksChangedEventData``, ``MarkUiResponseEvent``, and ``MarkVisualizationStateChangedEventData``. When any of these events arrive from the service (via the event bus), the controller's handler method is invoked.

**Publishing User Actions**: When the view calls a controller method in response to user interaction, the controller publishes an event. For example, when the user clicks "Create Mark", the view calls ``controller.create_mark(name, x, y)``, which publishes a ``MarkUiRequestEvent``. The controller doesn't wait for a response—it publishes the event and returns immediately. The view remains responsive.

**Updating the View**: When a service publishes a response event, the controller's event handler is invoked asynchronously in the GUI event loop. The handler receives the event data, formats or transforms it as needed, and then emits a Qt Signal (e.g., ``marks_updated.emit(marks_dict)``). Because Qt signals are thread-safe across threads (using queued connections), the signal safely marshals the data back to the main Qt thread where the view's slot updates the UI.

Views: Pure Presentation
==========================

Views inherit from Qt components (typically QWidget or QFrame) and are responsible only for rendering the UI and exposing callbacks. They contain no business logic, no service calls, and no event publishing. They are simple: create widgets, lay them out with proper theming, and provide slots for the controller to update the display.

**Widget Creation**: Views use themed components from PySide6 that apply the application's color scheme, fonts, and spacing defined in the Qt theme. Themed utilities like ``QtAssetCache`` and the theme configuration ensure every UI element looks identical and responds to theme changes.

**Callbacks to Controller**: When a user interacts with a widget—clicking a button, submitting a form, selecting an item—the widget's signal fires. The view connects this signal to a callback method that calls the appropriate method on its controller. For example, a button might trigger ``self.controller.create_mark(name)``. The controller handles the rest.

**Updating from Controller**: The controller updates the view through Qt Signals connected to view Slots. For example, ``controller.marks_updated.connect(self.populate_marks_list)``. These slots are simple: they create widgets, update labels, append items to lists, or show dialogs. No computation, no state management—just UI operations.

**Async-Safe Updates**: Because views are Qt-based and Qt UI operations must occur on the main thread, all view updates rely on Qt's cross-thread signal emission. When an event handler (running in the asyncio event loop thread) emits a signal, Qt automatically queues the slot execution on the main thread.

Specialized Overlay Windows
==============================

Beyond the main tabbed interface, Vocalance uses three specialized overlay windows for specific interactions: the grid overlay, mark visualization overlay, and dictation popup. These are created on demand and controlled directly by their controllers.

**Grid Overlay**: When the user requests the grid (by voice or clicking the grid button), the ``QtGridController`` shows the ``QtGridView``, a full-screen transparent overlay. The overlay divides the primary screen into cells based on the grid configuration. Each cell is labeled with a number (1, 2, 3, etc.). The grid supports three modes, chosen by the voice phrase used to open it:

- **Click mode** (default phrase ``go``): selecting a cell moves the cursor to that cell and performs a left click.
- **Hover mode** (default phrase ``hover``): selecting a cell moves the cursor only; no click.
- **Drag mode** (default phrase ``move``): when the overlay appears, the current pointer position is recorded. After the user picks a cell by number, the pointer returns to that recorded position, presses the left button, moves along an interpolated path to the cell center, pauses briefly so the target can register the drag, then releases at the cell center—equivalent to drag-and-drop from the original point to the chosen cell.

The grid listens for voice or keyboard digit input. After a cell is selected, the grid hides automatically.

**Mark Visualization**: Similarly, when the user requests mark visualization (by saying "show marks" or clicking a button), the ``QtMarksController`` shows the ``QtMarkView``, another full-screen overlay. This overlay draws circles at the exact screen coordinates of each mark and labels each circle with the mark's name. The mark overlay is always visible unless explicitly hidden, allowing the user to see where marks are on screen.

**Dictation Popup**: During dictation, a frameless overlay shows live feedback. Standard and type modes use a compact sound-wave “simple listening” state. Visual mode shows a single transcription pane. **Smart** and **amend** modes use the same dual-pane layout: streaming text on the left (column title “Dictation” vs “Prompt”), LLM output and a processing state on the right. Partial and final streaming segments update the left pane; tokens stream into the right pane after stop. The popup controller listens for ``DictationSessionEvent`` and branches on ``event.mode`` (``"smart"`` or ``"amend"``) and ``event.state``.

Event Flow: How User Actions Become Results
=============================================

Understanding the complete flow from user action to visible result is key to understanding the architecture. Here is a concrete example: the user says "Mark home" to create a mark.

1. **Capture**: The audio service captures the voice and sends audio chunks to the command segmenter.
2. **Recognition**: The command parser (via Vosk) recognizes "Mark home" as text.
3. **Parsing**: The parser identifies this as a mark creation command and publishes ``MarkCommandParsedEvent(command=MarkCreateCommand(label="home", ...))``.
4. **Service Handling**: The mark service receives this event, creates the mark in storage at the current cursor position, and publishes ``MarksChangedEventData(marks=...)``.
5. **Controller Update**: The marks controller, subscribed to ``MarksChangedEventData``, receives the event and emits the ``marks_updated`` Qt Signal.
6. **View Update**: The view's slot, connected to ``marks_updated``, adds the mark to its displayed list.
7. **Display**: The user sees the mark appear in the marks list.

This flow is entirely asynchronous. Step 1 occurs in the audio thread, steps 2-4 occur in the GUI event loop, step 5 occurs in the event handler (also GUI event loop), step 6 is queued to the main thread via Qt signals, and step 7 is rendered by PySide6 on the main thread.

At no point does any component block or wait. The view remains responsive to user input throughout, and the event bus processes other events while any long-running operation (like mark creation) is happening.

Thread Safety in the UI Layer
==============================

Vocalance runs three threads: the main thread (Qt UI loop), the GUI event loop thread (asyncio), and the audio thread (audio capture). The UI layer must handle cross-thread coordination carefully.

**Main Thread**: This is where Qt (PySide6) runs. All widget creation, configuration, and signal/slot handling must occur here. When the user clicks a button or types in a text field, the signal fires on the main thread.

**GUI Event Loop Thread**: This is where the event bus worker runs, service event handlers execute, and asyncio operations happen. Controllers and services run in this thread.

**Cross-Thread Updates**: When a service event handler (GUI event loop thread) needs to update a view (main thread), it must not call the view directly. Instead, the controller emits a Qt Signal. PySide6 automatically detects the cross-thread emission and queues the slot execution on the main thread.

The Base Controller Pattern
=============================

All controllers inherit from ``QtBaseController`` (which inherits from ``QObject``), which provides:

- **Event bus and logger references**: Common dependencies injected at startup.
- **View attachment**: ``set_view`` and ``get_view`` methods.
- **Status signals**: A common ``status_updated`` signal for showing errors or success messages.

By inheriting from the base controller class, each concrete controller gets these capabilities without duplicating code. The concrete controller then implements its specific event handlers, Qt Signals, and view methods.

Concrete Controllers
---------------------

**QtMarksController**: Handles mark creation, deletion, visualization, and execution. Subscribes to mark events from the mark service. Updates the marks list view. Controls the mark visualization overlay.

**QtCommandsController**: Displays command history and custom commands. Allows users to create, edit, and delete custom commands. Updates the commands view in real time.

**QtDictationController**: Manages dictation mode activation, shows dictation status and LLM model loading progress. Allows configuration of dictation modes and parameters.

**QtGridController**: Configures grid dimensions and appearance. Shows the grid overlay when requested. Handles grid cell selection and cursor movement.

**QtSoundController**: Manages sound training (teaching the system to recognize custom sounds) and mapping sounds to commands. Updates the sound list and training status in the view.

**QtSettingsController**: Displays and manages application settings. Validates user input. Publishes setting changes to the settings service for persistence and propagation to other services.

**QtDictationAliasController**: Manages custom dictation aliases—shorthand phrases that expand to full text during dictation output.

**QtDictationPopupController**: Controls the dictation popup window overlay.

Theming and Styling
====================

Vocalance uses PySide6 (Qt) with a custom dark theme. All visual elements use a consistent color scheme and typography defined in the theme configuration (``qt_theme.py``). Custom fonts are loaded and cached at startup via ``QtAssetCache``, avoiding repeated disk I/O. Styled components apply these colors and fonts automatically, ensuring consistency without repetitive configuration.

The theme uses QSS (Qt Style Sheets) for theming and Python configuration for dynamic property management. Because UI elements use the centralized theme, a global theme change is reflected everywhere automatically.

Summary and Architecture Overview
===================================

Vocalance's UI architecture is built on these principles:

1. **Separation of concerns**: Views, controllers, and services are distinct layers with clear responsibilities.
2. **Event-driven communication**: Components communicate via events, not direct calls.
3. **Thread safety**: Cross-thread UI updates are marshalled to the main thread using Qt Signals.
4. **Lazy loading**: Views are created on-demand, reducing startup time and memory.
5. **Responsive interaction**: The event bus is non-blocking, so the UI remains responsive even during long operations.

The flow from user action to visual result involves multiple threads, multiple layers, and multiple events—yet the system responds quickly to the user. This is the result of coordination: using async operations in the event loop, using thread pools for CPU-intensive work, marshalling UI updates to the main Qt thread, and designing services to publish completion events so the UI knows when to update.

The underlying infrastructure enabling this coordination—the event bus, threading model, and service lifecycle—is covered in detail in :doc:`event_bus_and_infrastructure`.
