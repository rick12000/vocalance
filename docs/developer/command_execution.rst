Command Execution
==================

The final stage of the pipeline executes the parsed commands by performing keyboard input, mouse clicks, or other automation actions.

AutomationService: Keyboard & Mouse Control
--------------------------------------------

The ``AutomationService`` executes automation commands using ``pyautogui`` for keyboard and mouse control.

Event Subscription
~~~~~~~~~~~~~~~~~~

The service subscribes to ``AutomationCommandParsedEvent``, which contains a structured command object with:

- ``command_key``: Unique identifier (e.g., "click", "scroll_down")
- ``action_type``: ActionType enum (hotkey, key, click, scroll, move, etc.)
- ``action_value``: Parameter for the action (e.g., "enter", "ctrl+c", "10")
- ``count``: Number of times to repeat (default 1)

Execution Flow
~~~~~~~~~~~~~~

This flowchart shows the complete execution pipeline from event reception to action execution. The service validates the command, checks cooldowns, creates the appropriate pyautogui action, and executes it in a thread pool:

.. mermaid::

   flowchart TD
       A[AutomationCommandParsed Event] --> B{count > 0?}
       B -->|No| C[Publish error status]
       B -->|Yes| D[Check cooldown]

       D --> E{On cooldown?}
       E -->|Yes| F[Publish cooldown status]
       E -->|No| G[Create action function]

       G --> H{action_type?}
       H -->|hotkey| I[lambda: pyautogui.hotkey]
       H -->|key| J[lambda: pyautogui.press]
       H -->|click| K[lambda: pyautogui.click]
       H -->|scroll| L[lambda: pyautogui.scroll]
       H -->|move| M[lambda: pyautogui.move]

       I --> N[Execute in thread pool]
       J --> N
       K --> N
       L --> N
       M --> N

       N --> O[Update cooldown timer]
       O --> P[Publish success status]

Action Function Creation
~~~~~~~~~~~~~~~~~~~~~~~~

The service maps action types to ``pyautogui`` calls:

.. code-block:: python

   # From AutomationService._create_action_function()
   if action_type == "hotkey":
       keys = [key.strip() for key in action_value.replace(" ", "+").split("+")]
       return lambda: pyautogui.hotkey(*keys)

   elif action_type == "key":
       return lambda: pyautogui.press(action_value)

   elif action_type == "click":
       return lambda: pyautogui.click()

   elif action_type == "scroll":
       amount = int(action_value)
       return lambda: pyautogui.scroll(amount)

Thread Pool Execution
~~~~~~~~~~~~~~~~~~~~~

PyAutoGUI is synchronous and blocks during execution. To avoid blocking the async event loop:

.. code-block:: python

   # From _execute_command()
   async with self._execution_lock:
   loop = asyncio.get_running_loop()
   return await loop.run_in_executor(
           self._thread_pool,  # ThreadPoolExecutor(max_workers=2)
       lambda: self._execute_action(action_function, count)
   )

The ``_execution_lock`` ensures only one automation command executes at a time.
