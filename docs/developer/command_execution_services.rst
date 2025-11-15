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

       H --> K[CommandExecutedStatusEvent]
       I --> K
       J --> K

       style E fill:#fff4e1
       style F fill:#e1f5ff
       style G fill:#fce4ec

The parser publishes different event types based on command type. Each service subscribes only to events relevant to its domain. After execution, all services publish status events to report results.

MarkService
============

The ``MarkService`` manages named screen positions. You can create marks at your current cursor location, then later jump to those positions by voice.

Creating Marks
--------------

When you say "mark home", the parser captures the current cursor position and the service stores it with a label:

.. code-block:: python

   async def _add_mark(self, label: str, x: int, y: int) -> Tuple[bool, str]:
       normalized_label = label.lower().strip()
       is_valid, reason = await self._is_label_valid(normalized_label)
       if not is_valid:
           return False, reason

       # Load current marks, add new one, save
       marks_data = await self._storage.read(model_type=MarksData)
       marks_data.marks[normalized_label] = Coordinate(x=x, y=y)
       success = await self._storage.write(data=marks_data)

       if success:
           return True, f"Mark '{normalized_label}' created."
       else:
           return False, "Failed to save mark to storage."

The parser captures position coordinates when the command is created. The service normalizes the label to lowercase, validates it against protected terms (reserved command names and existing marks), then stores it persistently to disk.

Executing Marks
---------------

When you say a mark name (e.g., "home"), the service looks up the stored position and clicks at that location:

.. code-block:: python

   elif isinstance(command, MarkExecuteCommand):
       coords = await self._get_mark_coordinates(command.label)
       if coords:
           x, y = coords
           logger.debug(f"Moving mouse to ({x}, {y}) and clicking for mark '{command.label}'")
           pyautogui.click(x, y)

           success = True
           message = f"Navigated to mark '{command.label}' at ({x}, {y}) and clicked."
           logger.info(message)

The service clicks at the stored position. This is different from just moving the mouse—it actually performs a click action at the mark location. The service publishes status events so the UI can display confirmation.

Managing Marks
--------------

In addition to creating and executing marks, the service supports deletion, visualization, and bulk reset:

**Delete**: Remove a specific mark by name:

.. code-block:: python

   if isinstance(event.command, MarkDeleteCommand):
       await self._storage.delete_mark(event.command.label)

**Visualize**: Display all mark positions on an overlay:

.. mermaid::

   sequenceDiagram
       participant U as User
       participant M as MarkService
       participant UI as UI Overlay
       participant C as CursorMonitor

       U->>M: "show marks"
       M->>UI: MarkVisualizeAllRequestEvent
       UI->>UI: Display overlay with labels
       M->>C: Start monitoring cursor

       loop Every 50ms
           C->>C: Check cursor position
           C->>UI: Update hover highlight
       end

       U->>M: "hide marks"
       M->>C: Stop monitoring
       M->>UI: Hide overlay

When visualization is active, a background task monitors cursor position every 50ms to highlight nearby marks. This helps with spatial awareness and discovery.

**Reset**: Clear all marks with a single command:

.. code-block:: python

   if isinstance(event.command, MarkResetCommand):
       await self._storage.reset_all_marks()

GridService
============

The ``GridService`` displays an overlay grid that divides the screen into numbered cells. You can select cells by voice to click or hover over precise locations without using the mouse.

Displaying the Grid
-------------------

The grid supports two modes: **click mode** (default) and **hover mode**.

When you say "**go**", the service displays the grid in click mode:

.. code-block:: python

   if isinstance(command, GridShowCommand):
       num_rects = command.num_rects or self._config.grid.default_rect_count
       rows, cols = self._calculate_grid_dimensions(num_rects)
       click_mode = command.click_mode  # "click" or "hover"

       show_event = ShowGridRequestEventData(rows=rows, cols=cols, click_mode=click_mode)
       self.event_publisher.publish(show_event)
       await self._publish_visibility_event(True, rows, cols)

When you say "**hover**", the grid displays in hover mode instead. In hover mode, selecting a cell moves the mouse without clicking.

The service optimizes dimensions to minimize aspect ratio distortion:

- 36 cells → 6 columns × 6 rows (square)
- 50 cells → 8 columns × 7 rows (near-square)
- 100 cells → 10 columns × 10 rows (square)

Both grid phrases support optional cell counts: "go 100" or "hover 50".

Cell Selection: Click vs Hover Modes
--------------------------------------

Once displayed, you can select cells by voice. The behavior depends on which trigger phrase you used:

**Click Mode** (triggered by "go"): When you say a number, the mouse moves to that cell and clicks.

**Hover Mode** (triggered by "hover"): When you say a number, the mouse only moves to that cell without clicking.

The service tracks the current mode and applies it when processing cell selections:

.. code-block:: python

   elif isinstance(command, GridSelectCommand):
       async with self._state_lock:
           is_visible = self._visible
           if not is_visible:
               return
           click_mode = self._current_click_mode  # Stored from GridShowCommand

       click_event = ClickGridCellRequestEventData(
           cell_label=str(command.selected_number),
           click_mode=click_mode
       )
       self.event_publisher.publish(click_event)

The view layer (GridView) respects the click_mode:

.. code-block:: python

   if click_mode == "hover":
       pyautogui.moveTo(center_x, center_y)  # Only move mouse
   else:
       pyautogui.click(center_x, center_y)  # Move and click
       )

**Cell selection**: Cells are identified by number (1, 2, 3, etc.). The service sends the cell number to the UI, which calculates the cell center and clicks. The grid auto-hides after a successful click.

**Click tracking**: The system logs which cells are clicked to identify frequently-used areas. This passive learning helps suggest marks for commonly-accessed positions.

**Auto-hide**: The grid automatically hides after a successful click to avoid cluttering the screen.

Configuration
--------------

Grid behavior can be customized through settings:

- **Default cell count**: How many cells to display (default: 36)
- **Cell colors**: Overlay appearance and contrast
- **Label font**: Cell label typography
- **Transparency**: Overlay opacity for visibility

Configuration changes are propagated through ``GridConfigUpdatedEvent``, which the service handles by updating internal settings.

AutomationService
==================

The ``AutomationService`` executes keyboard and mouse automation commands. It uses PyAutoGUI as the underlying automation library and manages execution timing to prevent conflicts.

Command Dispatch
-----------------

When an automation command arrives, the service creates an action function from the action type and value, then executes it through the thread pool:

.. code-block:: python

   def _create_action_function(self, action_type: ActionType, action_value: str) -> Optional[Callable[[], None]]:
       if action_type == "hotkey":
           keys = [key.strip() for key in action_value.replace(" ", "+").split("+")]
           return lambda: pyautogui.hotkey(*keys)
       elif action_type == "key":
           return lambda: pyautogui.press(action_value)
       elif action_type == "click":
           click_actions = {
               "click": lambda: pyautogui.click(button="left"),
               "left_click": lambda: pyautogui.click(button="left"),
               "right_click": lambda: pyautogui.click(button="right"),
               "double_click": pyautogui.doubleClick,
               "triple_click": pyautogui.tripleClick,
           }
           return click_actions.get(action_value)
       elif action_type == "scroll":
           # ... scroll direction handling
           return scroll_directions.get(action_value)

       return None

Each action type maps to a specific PyAutoGUI call. The service creates a lambda function that encapsulates the PyAutoGUI call, then executes it in the thread pool.

Cooldown Management
-------------------

To prevent accidental rapid-fire execution and avoid overwhelming the system, each command has a cooldown period after execution:

.. mermaid::

   sequenceDiagram
       participant U as User
       participant A as AutomationService
       participant T as Cooldown Timer
       participant P as PyAutoGUI

       U->>A: "click" (t=0ms)
       A->>T: Check cooldown for "click"
       T->>A: No recent execution
       A->>P: Execute click
       A->>T: Record execution time

       U->>A: "click" (t=50ms)
       A->>T: Check cooldown for "click"
       T->>A: Too soon! (within 200ms)
       A->>U: Cooldown message

       U->>A: "click" (t=250ms)
       A->>T: Check cooldown for "click"
       T->>A: OK, cooldown expired
       A->>P: Execute click

**Default cooldown**: Configurable via `automation_cooldown_seconds` in app config (typically 200ms). This prevents misrecognitions from causing multiple rapid executions.

**Per-command tracking**: Each command maintains its own timer. "click" and "press enter" don't interfere with each other.

**Configurable**: Cooldown duration can be adjusted globally or per-command based on requirements.

Non-Blocking Execution
-----------------------

PyAutoGUI calls are synchronous and can block for 50-500ms. To prevent blocking the async event loop, the service runs PyAutoGUI in a thread pool:

.. code-block:: python

   def __init__(self, event_bus: EventBus, app_config: GlobalAppConfig) -> None:
       self._thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(
           max_workers=app_config.automation_service.thread_pool_max_workers
       )

   async def _execute_command(self, action_type: ActionType, action_value: str, count: int = 1) -> bool:
       action_function = self._create_action_function(action_type, action_value)
       if not action_function:
           return False

       if not self._execution_lock.locked():
           async with self._execution_lock:
               loop = asyncio.get_running_loop()
               return await loop.run_in_executor(self._thread_pool, lambda: self._execute_action(action_function, count))
       else:
           logger.warning("Could not acquire execution lock - another command in progress")
           return False

This ensures automation commands don't block STT recognition, event processing, or UI updates. The execution lock ensures commands execute serially rather than simultaneously, preventing race conditions.

Repeat Counts
-------------

Parameterized commands can include repeat counts. For example:

- "press down" → Press down once
- "press down 5" → Press down five times
- "click 3" → Click three times

The service extracts the count and executes the action repeatedly:

.. code-block:: python

   count = getattr(command, "count", 1)

   # Validate count
   if count <= 0 or count > 100:
       await self._publish_status(command, False, "Invalid count")
       return

   # Execute count times
   for i in range(count):
       success = await self._execute_command(
           command.action_type,
           command.action_value
       )
       if not success:
           break

**Safety limits**: Maximum repeat count is 100 to prevent runaway execution from misrecognized numbers.

**Early termination**: If an iteration fails, execution stops rather than continuing with remaining iterations.

Action Value Formats
--------------------

The service handles various action value formats depending on action type:

- **Click**: "click", "left_click", "right_click", "double_click", "triple_click"
- **Key**: Single key names ("enter", "escape", "shift", "control", etc.)
- **Key Sequence**: Comma-separated key combinations with delays between steps
- **Hotkey**: Key combinations with "+" separator ("ctrl+c", "alt+tab", "ctrl+shift+n")
- **Scroll**: "up" or "down" with animated scrolling over multiple steps
- **Type**: Literal text (not shown as a direct action type in code)

Execution Status and Error Handling
====================================

All three services publish ``CommandExecutedStatusEvent`` after each execution:

.. code-block:: python

   CommandExecutedStatusEvent(
       command={"command_type": "MarkExecuteCommand", "label": "home"},
       success=True,
       message="Jumped to mark 'home'",
       source="mark_service"
   )

These status events flow through the event bus to:

- **UI**: Display success/failure notifications and feedback
- **Logging**: Maintain audit trail of all executions
- **History**: Track execution patterns for analytics

**Common error scenarios**:

- Mark not found (execution failed)
- Grid not visible when trying to select cell
- PyAutoGUI failure (screen locked, permission denied)
- Invalid parameters (negative scroll, unknown key name)

Thread Safety
=============

All services use ``asyncio.Lock`` to protect mutable state from concurrent modification:

.. code-block:: python

   # MarkService: Protect visualization state
   async with self._viz_lock:
       self._is_viz_active = True

   # GridService: Protect visibility state
   async with self._state_lock:
       self._visible = True

   # AutomationService: Protect cooldown timers
   async with self._cooldown_lock:
       self._cooldown_timers[command_key] = time.time()

These locks prevent race conditions when multiple commands arrive simultaneously or when voice and UI commands interact. Proper locking ensures consistent state across async operations.

What Happens Next
==================

After command execution completes:

- **Status events** are published so the UI displays results
- **Dictation commands** follow a separate path through the DictationCoordinator
- **Command history** records successful executions for Markov prediction
- **System returns** to idle state waiting for the next command

The specialized dictation system, which operates independently from command execution, is covered in :doc:`dictation_system`.
