User Interface Architecture
############################

Vocalance's user interface is built with PySide6 (Qt framework) and follows a separation of concerns through the **Control-View pattern**, where controllers act as mediators between the view layer and backend services through an event-driven architecture. This document explains how the UI is organized, how user interactions flow through the system, and how the UI remains responsive across multiple threads.

The Core Pattern: Controllers Orchestrating Views and Services
==============================================================

The Vocalance UI operates on a simple principle: **views never call services directly**. Instead, controllers intercede, translating user interactions into events published to the event bus, and subscribing to service responses to update the UI. This creates a clean, testable, and decoupled architecture.

When a user interacts with the UI—clicking a button, entering text, or selecting an item—the view's callback fires. The view passes control to the controller, which then publishes an event describing the user's intent (e.g., "MarkCreateRequestEventData" when creating a mark). The event bus routes this to the backend service handling marks. When the service completes the operation, it publishes a completion event (e.g., "MarkCreatedEventData"). The controller, which has subscribed to this event, receives it and updates the view accordingly.

This pattern ensures views are pure presentation logic—they know nothing about business logic or services. Controllers are thin orchestrators—they know about events and service workflows but contain no UI rendering code. Services implement the actual domain logic and publish events describing what happened. The separation means each layer can be tested independently, modified without cascading changes, and understood easily.

The Main Window: VocalanceMainWindow
====================================

The root of the UI hierarchy is the `VocalanceMainWindow` (inheriting from QMainWindow), which manages the main window containing a sidebar for navigation and a content area that displays one active tab at a time. The VocalanceMainWindow is responsible for creating all controllers and views, orchestrating tab switching, and maintaining the specialized overlay windows (grid overlay, mark visualization, dictation popup).

**Window Layout**: The main window is divided into a narrow sidebar with icon buttons and a large content area. The sidebar buttons correspond to tabs: Marks, Commands, Dictation, Grid, Sound, Settings, and System. Each button switches the content area to the corresponding view. The window is designed for touch and voice interaction, with large buttons and high-contrast visuals.

**Lazy-Loaded Views**: Views are not created upfront. Instead, the VocalanceMainWindow maintains a cache (`_view_cache`) and creates each view only when the user first navigates to it. This reduces startup time and memory usage. The cache is protected by a threading.RLock to allow safe access from multiple threads (the main Qt thread and the GUI event loop thread). When switching to a tab, VocalanceMainWindow checks if the view is cached. If not, it creates the view and its associated controller. If already cached, it reuses the existing view. Only one view is displayed at a time—switching tabs hides the current view and displays the target view using Qt's stacked widget or layout management.

This lazy loading design means the application starts quickly even though it supports seven different functional areas. The first access to any tab incurs a small delay as the view and controller initialize, but subsequent accesses are instant.

Controllers: The Coordination Layer
====================================

Each functional area (marks, commands, dictation, grid, sound, settings, system) has a dedicated controller that inherits from `BaseController`. The controller's responsibility is singular and focused: translate user actions into events, handle service responses, and coordinate view updates.

**Event Subscriptions**: During initialization, a controller subscribes to the event types it cares about. For example, `MarksController` subscribes to `MarkCreatedEventData`, `MarkDeletedEventData`, `MarksChangedEventData`, and `MarkVisualizationStateChangedEventData`. When any of these events arrive from the service (via the event bus), the controller's handler method is invoked.

**Publishing User Actions**: When the view calls a controller method in response to user interaction, the controller publishes an event. For example, when the user clicks "Create Mark", the view calls `controller.create_mark(name, x, y)`, which publishes a `MarkCreateRequestEventData` event. The controller doesn't wait for a response—it publishes the event and returns immediately. The view remains responsive.

**Updating the View**: When a service publishes a response event (like `MarkCreatedEventData`), the controller's event handler is invoked asynchronously in the GUI event loop. The handler receives the event data, formats or transforms it as needed, and then calls a view method to update the display. To ensure thread safety, the controller uses `schedule_ui_update()` to marshal the view update back to the main Tkinter thread.

**State Management**: Controllers maintain minimal state—typically just references to their view, the event bus, and event loop. State that tracks controller behavior (like whether mark visualization is active) is protected by `_state_lock`, a threading.RLock that allows safe access from both the main Tkinter thread and the GUI event loop thread.

Views: Pure Presentation
==========================

Views inherit from Qt components (typically QWidget or QFrame) and are responsible only for rendering the UI and exposing callbacks. They contain no business logic, no service calls, and no event publishing. They are simple: create widgets, lay them out with proper theming, and provide methods for the controller to update the display.

**Widget Creation**: Views use themed components from PySide6 that apply the application's color scheme, fonts, and spacing defined in the Qt theme. Themed utilities like QtAssetCache and the theme configuration ensure every UI element looks identical and responds to theme changes.

**Callbacks to Controller**: When a user interacts with a widget—clicking a button, submitting a form, selecting an item—the widget's signal fires. The view connects this signal to a callback method that calls the appropriate method on its controller. For example, a button might trigger `self.controller.create_mark(name)` with the name from a text entry. The controller handles the rest.

**Updating from Controller**: The controller updates the view through public methods. For example, `view.add_mark_to_list(mark_data)` or `view.show_error(title, message)`. These methods are simple: they create widgets, update labels, append items to lists, or show dialogs. No computation, no state management—just UI operations.

**Async-Safe Updates**: Because views are Qt-based and Qt signals/slots operate on the main thread, all view updates must occur on the main thread. When a controller event handler (which runs in the GUI event loop thread) needs to update the view, it publishes the update to the event loop which marshals it back to the main Qt thread. This ensures thread safety without blocking the GUI event loop.

Specialized Overlay Windows
==============================

Beyond the main tabbed interface, Vocalance uses three specialized overlay windows for specific interactions: the grid overlay, mark visualization overlay, and dictation popup. These are created on demand and controlled directly by their controllers.

**Grid Overlay**: When the user requests the grid (by saying "go" or "hover", or clicking the grid button), the `GridController` shows the `GridView`, a full-screen transparent overlay. The overlay divides the screen into cells based on the grid configuration (columns and rows). Each cell is labeled with a number (1, 2, 3, etc.). The grid supports two modes: **click mode** (triggered by "go") where selecting a cell moves the cursor and clicks, and **hover mode** (triggered by "hover") where selecting a cell only moves the cursor without clicking. The grid listens for voice commands or mouse clicks. After a cell is selected, the grid hides automatically.

**Mark Visualization**: Similarly, when the user requests mark visualization (by saying "show marks" or clicking a button), the `MarksController` shows the `MarkView`, another full-screen overlay. This overlay draws circles at the exact screen coordinates of each mark and labels each circle with the mark's name. The mark overlay is always visible unless explicitly hidden, allowing the user to see where marks are on screen. Hovering near a mark highlights it. Clicking on a mark's label clicks that mark's position.

**Dictation Popup**: During dictation, a frameless overlay shows live feedback. Standard and type modes use a compact sound-wave “simple listening” state. Visual mode shows a single transcription pane. **Smart** and **amend** modes use the same dual-pane layout: streaming text on the left (column title “Dictation” vs “Prompt”), LLM output and a processing state on the right. Partial and final streaming segments update the left pane; tokens stream into the right pane after stop. The popup controller listens for ``SmartDictationStartedEvent`` / ``StoppedEvent`` and branches on ``event.mode`` (``"smart"`` or ``"amend"``).

Event Flow: How User Actions Become Results
=============================================

Understanding the complete flow from user action to visible result is key to understanding the architecture. Here is a concrete example: the user says "Mark home" to create a mark.

1. **Capture**: The audio service captures the voice and sends audio chunks to the command parser.
2. **Recognition**: The command parser (via Vosk) recognizes "Mark home" as text.
3. **Parsing**: The parser identifies this as a mark creation command and publishes `MarkCreateRequestEventData(name="home")`.
4. **Service Handling**: The mark service receives this event, creates the mark in storage at the current cursor position, and publishes `MarkCreatedEventData(name="home", ...)`.
5. **Controller Update**: The marks controller, subscribed to `MarkCreatedEventData`, receives the event and calls `self.view.add_mark_to_list(mark_data)`.
6. **View Update**: The view adds the mark to its displayed list and optionally shows a success message.
7. **Display**: The user sees the mark appear in the marks list.

This flow is entirely asynchronous. Step 1 occurs in the audio thread, steps 2-4 occur in the GUI event loop, step 5 occurs in the event handler (also GUI event loop), step 6 is scheduled on the main thread via `schedule_ui_update()`, and step 7 is rendered by Tkinter on the main thread.

At no point does any component block or wait. The view remains responsive to user input throughout, and the event bus processes other events while any long-running operation (like mark creation) is happening.

Thread Safety in the UI Layer
==============================

Vocalance runs three threads: the main thread (Qt UI loop), the GUI event loop thread (asyncio), and the audio thread (audio capture). The UI layer must handle cross-thread coordination carefully.

**Main Thread**: This is where Qt (PySide6) runs. All widget creation, configuration, and signal/slot handling must occur here. When the user clicks a button or types in a text field, the signal fires on the main thread.

**GUI Event Loop Thread**: This is where the event bus worker runs, service event handlers execute, and asyncio operations happen. Controllers and services run in this thread.

**Cross-Thread Updates**: When a service event handler (GUI event loop thread) needs to update a view (main thread), it must not call the view directly. Instead, the event loop marshals the callback to run on the main thread using Qt's signal/slot mechanism or async scheduling.

**State Protection**: If a controller maintains state that can be accessed from both threads (e.g., whether mark visualization is active), that state must be protected by a lock (threading.RLock). The lock prevents one thread from modifying state while another thread is reading it.

The Base Controller Pattern
=============================

All controllers inherit from `BaseController` (or Qt-specific variants), which provides:

- **Event subscription management**: Methods for registering handlers for specific event types.
- **Event publishing**: Thread-safe publication of events to the event bus.
- **UI scheduling**: Methods to safely call view methods from event handlers running in the event loop thread.
- **State locking**: `_state_lock` (threading.RLock) protects controller state shared between threads.
- **View callbacks**: Reference management for specialized views (grid, mark overlay, dictation popup).

By inheriting from the base controller class, each concrete controller gets these capabilities without duplicating code. The concrete controller then implements its specific event handlers and view methods.

Concrete Controllers
---------------------

**MarksController** (`qt_marks_controller.py`): Handles mark creation, deletion, visualization, and execution. Subscribes to mark events from the mark service. Updates the marks list view. Controls the mark visualization overlay.

**CommandsController** (`qt_commands_controller.py`): Displays command history and custom commands. Allows users to create, edit, and delete custom commands. Updates the commands view in real time.

**DictationController** (`qt_dictation_controller.py`): Manages dictation mode activation, shows dictation status and LLM model loading progress. Allows configuration of dictation modes and parameters. Controls the dictation popup window.

**GridController** (`qt_grid_controller.py`): Configures grid dimensions and appearance. Shows the grid overlay when requested. Handles grid cell selection and cursor movement.

**SoundController** (`qt_sound_controller.py`): Manages sound training (teaching the system to recognize custom sounds) and mapping sounds to commands. Updates the sound list and training status in the view.

**SettingsController** (`qt_settings_controller.py`): Displays and manages application settings. Validates user input. Publishes setting changes to the settings service for persistence and propagation to other services.

**DictationAliasController** (`qt_dictation_alias_controller.py`): Manages custom dictation aliases—shorthand phrases that expand to full text during dictation output.

Theming and Styling
====================

Vocalance uses PySide6 (Qt) with a custom dark theme. All visual elements use a consistent color scheme and typography defined in the theme configuration (`qt_theme.py`). Custom fonts are loaded and cached at startup via `QtAssetCache`, avoiding repeated disk I/O. Styled components apply these colors and fonts automatically, ensuring consistency without repetitive configuration.

The theme uses QSS (Qt Style Sheets) for theming and Python configuration for dynamic property management. When the user changes theme settings (if supported), the settings coordinator propagates the change to all affected services and widgets. Because UI elements use the centralized theme, a global theme change is reflected everywhere automatically.

Summary and Architecture Overview
===================================

Vocalance's UI architecture is built on these principles:

1. **Separation of concerns**: Views, controllers, and services are distinct layers with clear responsibilities.
2. **Event-driven communication**: Components communicate via events, not direct calls.
3. **Thread safety**: Cross-thread state is protected by locks, and cross-thread UI updates are marshalled to the main thread.
4. **Lazy loading**: Views are created on-demand, reducing startup time and memory.
5. **Responsive interaction**: The event bus is non-blocking, so the UI remains responsive even during long operations.

The flow from user action to visual result involves multiple threads, multiple layers, and multiple events—yet the system responds quickly to the user. This is the result of coordination: using async operations in the event loop, using thread pools for CPU-intensive work, marshalling UI updates to the main Qt thread, and designing services to publish completion events so the UI knows when to update.

The underlying infrastructure enabling this coordination—the event bus, threading model, and service lifecycle—is covered in detail in :doc:`event_bus_and_infrastructure`.
