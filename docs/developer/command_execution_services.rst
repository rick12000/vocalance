Command Execution Services
############################

This page explains how parsed commands are executed through three independent services: MarkService for position bookmarking, GridService for click grid overlays, and AutomationService for keyboard and mouse automation.

System Overview
================

After commands are parsed (see :doc:`command_parsing`), they are routed to execution services through typed events. Each service listens for its relevant event type and operates independently from the others.

.. mermaid::

   flowchart TD
       A[CentralizedCommandParser] --> B[MarkCommandParsedEvent]
       A --> C[GridCommandParsedEvent]
       A --> D[AutomationCommandParsedEvent]

       B --> E[MarkService]
       C --> F[GridService]
       D --> G[AutomationService]

       E --> H[Mouse Jump<br/>Position Storage]
       F --> I[Grid Overlay<br/>Cell Click]
       G --> J[PyAutoGUI<br/>Keyboard/Mouse]

       H --> K[Execution Log]
       I --> K
       J --> K

       style E fill:#fff4e1
       style F fill:#e1f5ff
       style G fill:#fce4ec

The parser publishes different event types based on command type. Each service subscribes only to events relevant to its domain. After execution, all services log their status.

MarkService
============

The ``MarkService`` manages named screen positions. You can create marks at your current cursor location, then later jump to those positions by voice.

Creating Marks
--------------

When you say "mark home", the parser captures the current cursor position and the service stores it with a label:

.. code-block:: python

   async def add_mark(self, label: str, x: int, y: int) -> Tuple[bool, str]:
       normalized_label: str = label.lower().strip()
       is_valid, reason = await self.is_label_valid(normalized_label)
       if not is_valid:
           return False, reason

       marks_data = await self.storage.read(model_type=MarksData)
       marks_data.marks[normalized_label] = Coordinate(x=x, y=y)
       success: bool = await self.storage.write(data=marks_data)

       if success:
           self.protected_terms_validator.invalidate_cache()
           return True, f"Mark '{normalized_label}' created."
       return False, "Failed to save mark to storage."

The parser captures position coordinates when the command is created. The service normalizes the label to lowercase, validates it against protected terms (reserved command names and existing marks), then stores it persistently to disk.

Executing Marks
---------------

When you say a mark name (e.g., "home"), the service looks up the stored position and clicks at that location:

.. code-block:: python

   elif isinstance(command, MarkExecuteCommand):
       coords = await self.get_mark_coordinates_internal(command.label)
       if coords:
           x, y = coords
           loop = asyncio.get_running_loop()
           await loop.run_in_executor(shared_input_executor, pyautogui.click, x, y)
           logger.info("Navigated to mark '%s' at (%s, %s) and clicked.", command.label, x, y)

The service clicks at the stored position using a shared thread pool executor. This is different from just moving the mouse—it actually performs a click action at the mark location.

Managing Marks
--------------

In addition to creating and executing marks, the service supports deletion, visualization, and bulk reset:

**Delete**: Remove a specific mark by name.

**Visualize**: Display all mark positions on an overlay:

.. code-block:: python

   async def set_visualization(self, show: bool) -> None:
       self.is_viz_active = show
       marks_payload: Optional[Dict[str, Dict[str, Any]]] = None
       if show:
           marks_payload = await self.get_all_marks()
       await self.event_bus.publish(
           MarkVisualizationStateChangedEventData(is_visible=show, marks=marks_payload)
       )

When visualization is active, an event is published to the UI to display the marks overlay.

**Reset**: Clear all marks with a single command.

GridService
============

The ``GridService`` displays an overlay grid that divides the primary screen into numbered cells. You can select cells by voice (or keyboard while the overlay has focus) to click, hover, or drag between precise locations.

Displaying the Grid
-------------------

The grid supports three modes, selected by which show phrase was recognized. Each maps to a ``GridShowCommand`` with a ``click_mode`` of ``"click"``, ``"hover"``, or ``"drag"``. Default phrases are configured on ``GridConfig`` (``show_grid_phrase``, ``hover_grid_phrase``, ``drag_grid_phrase``).

When a ``GridShowCommand`` is handled, the service computes rows and columns, stores ``click_mode`` for the next selection, and publishes ``GridStateEvent``:

.. code-block:: python

   if isinstance(command, GridShowCommand):
       num_rects = command.num_rects or self._config.grid.default_rect_count
       rows, cols = self._calculate_grid_dimensions(num_rects)
       self._current_click_mode = command.click_mode
       self._visible = True
       await self._event_bus.publish(
           GridStateEvent(
               state="visible",
               config={"rows": rows, "cols": cols, "click_mode": command.click_mode}
           )
       )

**Click mode** (e.g. saying ``go``): opening phrase shows the grid for point-and-click targeting.

**Hover mode** (e.g. saying ``hover``): selecting a cell moves the pointer only.

**Drag mode** (e.g. saying ``move``): the view records the pointer position when the overlay is shown; selecting a cell performs a left-button drag from that recorded point to the cell center.

The service optimizes dimensions to keep the layout nearly square (for example, 36 cells → 6×6, 100 cells → 10×10).

Cell Selection: Click, Hover, and Drag
----------------------------------------

Once displayed, a spoken number selects a cell. The service reads the stored ``click_mode`` and publishes ``GridStateEvent``:

.. code-block:: python

   elif isinstance(command, GridSelectCommand):
       if not self._visible:
           return
       await self._event_bus.publish(
           GridStateEvent(
               state="interaction_request",
               config={"cell_label": str(command.selected_number), "click_mode": self._current_click_mode},
           )
       )

The view handles the actual interaction: hiding the overlay first, then using PyAutoGUI to perform the click, hover, or drag operation.

AutomationService
==================

The ``AutomationService`` executes keyboard and mouse automation commands. It uses PyAutoGUI as the underlying automation library and manages execution timing to prevent conflicts.

Command Dispatch
-----------------

When an automation command arrives, the service creates an action function from the action type and value, then executes it through a shared thread pool:

.. code-block:: python

   def create_action_function(self, action_type: ActionType, action_value: str) -> Optional[Callable[[], None]]:
       if action_type == "hotkey":
           keys = [k.strip() for k in action_value.replace(" ", "+").split("+")]
           return lambda: pyautogui.hotkey(*keys)
       if action_type == "key":
           return lambda: pyautogui.press(action_value)
       if action_type == "click":
           return {
               "click": lambda: pyautogui.click(button="left"),
               "left_click": lambda: pyautogui.click(button="left"),
               "right_click": lambda: pyautogui.click(button="right"),
               "double_click": pyautogui.doubleClick,
               "triple_click": pyautogui.tripleClick,
           }.get(action_value)

Each action type maps to a specific PyAutoGUI call. The service creates a lambda function that encapsulates the PyAutoGUI call, then executes it in the thread pool.

Cooldown Management
-------------------

To prevent accidental rapid-fire execution and avoid overwhelming the system, each command has a cooldown period after execution:

.. code-block:: python

   def check_cooldown(self, command_key: str) -> bool:
       return time.time() - self.cooldown_timers.get(command_key, 0) >= self.config.automation_cooldown_seconds

**Default cooldown**: Configurable via `automation_cooldown_seconds` in app config. This prevents misrecognitions from causing multiple rapid executions.

**Per-command tracking**: Each command maintains its own timer in a dictionary. "click" and "press enter" don't interfere with each other.

Non-Blocking Execution
-----------------------

PyAutoGUI calls are synchronous and can block for 50-500ms. To prevent blocking the async event loop, the service runs PyAutoGUI in a shared thread pool (``shared_input_executor``):

.. code-block:: python

   async def handle_automation_command_parsed(self, event: AutomationCommandParsedEvent) -> None:
       # ... setup and cooldown checks ...
       loop = asyncio.get_running_loop()
       await loop.run_in_executor(shared_input_executor, lambda: self.run_action(action_fn, count))
       self.cooldown_timers[command.command_key] = time.time()

This ensures automation commands don't block STT recognition, event processing, or UI updates.

Repeat Counts
-------------

Parameterized commands can include repeat counts. For example:

- "press down" → Press down once
- "press down 5" → Press down five times
- "click 3" → Click three times

The service extracts the count and executes the action repeatedly:

.. code-block:: python

   def run_action(self, action_fn: Callable[[], None], count: int) -> None:
       for _ in range(count):
           action_fn()

What Happens Next
==================

After command execution completes:

- **Dictation commands** follow a separate path through the DictationCoordinator
- **System returns** to idle state waiting for the next command

The specialized dictation system, which operates independently from command execution, is covered in :doc:`dictation_system`.
