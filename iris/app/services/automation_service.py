"""
Automation Service

Centralized service for automation command handling.
Executes automation commands with cooldown management and error handling.
"""
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import pyautogui

from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.command_types import ActionType, ParameterizedCommand
from iris.app.event_bus import EventBus
from iris.app.events.command_events import AutomationCommandParsedEvent
from iris.app.events.command_management_events import CommandMappingsUpdatedEvent
from iris.app.events.core_events import CommandExecutedStatusEvent

logger = logging.getLogger(__name__)


class AutomationService:
    """
    Service responsible for processing and executing automation commands.

    Handles command execution with cooldown management, thread safety,
    and status reporting via events.
    """

    def __init__(self, event_bus: EventBus, app_config: GlobalAppConfig):
        self._event_bus = event_bus
        self._app_config = app_config

        # Execution components
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        self._execution_lock = threading.Lock()

        # Cooldown management
        self._cooldown_timers: Dict[str, float] = {}

        logger.info("AutomationService initialized")

    def setup_subscriptions(self) -> None:
        """Set up event subscriptions for automation commands"""
        self._event_bus.subscribe(event_type=AutomationCommandParsedEvent, handler=self._handle_automation_command)
        self._event_bus.subscribe(event_type=CommandMappingsUpdatedEvent, handler=self._handle_command_mappings_updated)
        logger.info("AutomationService subscriptions set up")

    async def _handle_automation_command(self, event_data: AutomationCommandParsedEvent) -> None:
        """Handle automation command execution"""
        command = event_data.command
        count = getattr(command, "count", 1)

        # Validate count for parameterized commands
        if isinstance(command, ParameterizedCommand) and count <= 0:
            await self._publish_status(command, event_data.source, False, f"Invalid repeat count: {count}")
            return

        # Check cooldown
        if not self._check_cooldown(command.command_key):
            await self._publish_status(command, event_data.source, False, f"Command '{command.command_key}' is on cooldown")
            return

        # Execute command
        success = await self._execute_command(command.action_type, command.action_value, count)

        # Update cooldown on success
        if success:
            self._cooldown_timers[command.command_key] = time.time()

        # Report status
        count_text = f" {count} times" if count > 1 else ""
        status = "successfully" if success else "failed"
        message = f"Command '{command.command_key}' executed{count_text} {status}"
        await self._publish_status(command, event_data.source, success, message)

    async def _execute_command(self, action_type: ActionType, action_value: str, count: int = 1) -> bool:
        """Execute automation action with thread safety"""
        try:
            action_function = self._create_action_function(action_type, action_value)
            if not action_function:
                return False

            # Execute in thread pool with locking
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._thread_pool, lambda: self._execute_with_lock(action_function, count))

        except Exception as e:
            logger.error(f"Error executing automation action {action_type}:{action_value}: {e}")
            return False

    def _execute_with_lock(self, action_function: Callable, count: int) -> bool:
        """Execute action function with thread safety"""
        if not self._execution_lock.acquire(blocking=False):
            logger.warning("Could not acquire execution lock - another command in progress")
            return False

        try:
            for _ in range(count):
                action_function()
            return True
        except Exception as e:
            logger.error(f"Error during action execution: {e}")
            return False
        finally:
            self._execution_lock.release()

    def _create_action_function(self, action_type: ActionType, action_value: str) -> Optional[Callable]:
        """Create pyautogui action function from action type and value"""
        try:
            if action_type == "hotkey":
                keys = [key.strip() for key in action_value.replace(" ", "+").split("+")]
                return lambda: pyautogui.hotkey(*keys)

            elif action_type == "key":
                return lambda: pyautogui.press(action_value)

            elif action_type == "key_sequence":
                key_list = [key.strip() for key in action_value.split(",")]
                return lambda: self._execute_key_sequence(key_list)

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
                scroll_amount = self._app_config.scroll_amount_vertical
                if action_value == "scroll_up":
                    return lambda: pyautogui.scroll(scroll_amount)
                elif action_value == "scroll_down":
                    return lambda: pyautogui.scroll(-scroll_amount)

        except Exception as e:
            logger.error(f"Error creating action function for {action_type}:{action_value}: {e}")

        return None

    def _execute_key_sequence(self, key_list: list) -> None:
        """Execute a sequence of keys or hotkeys in order"""
        for key_combination in key_list:
            if "+" in key_combination:
                keys = [k.strip() for k in key_combination.split("+")]
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(key_combination.strip())
            time.sleep(0.25)

    def _check_cooldown(self, command_key: str) -> bool:
        """Check if command is not on cooldown"""
        current_time = time.time()
        last_execution = self._cooldown_timers.get(command_key, 0)
        cooldown_period = self._app_config.automation_cooldown_seconds
        return current_time - last_execution >= cooldown_period

    async def _publish_status(self, command, source: Optional[str], success: bool, message: str) -> None:
        """Publish command execution status event"""
        try:
            status_event = CommandExecutedStatusEvent(
                command={
                    "command_key": command.command_key,
                    "action_type": command.action_type,
                    "action_value": command.action_value,
                },
                success=success,
                message=message,
                source=source,
            )
            await self._event_bus.publish(status_event)
        except Exception as e:
            logger.error(f"Error publishing command status: {e}")

    async def _handle_command_mappings_updated(self, event_data: CommandMappingsUpdatedEvent) -> None:
        """Clear cooldown timers when command mappings change"""
        self._cooldown_timers.clear()
        logger.info("Cleared automation command cooldown timers after mappings update")

    async def shutdown(self) -> None:
        """Shutdown the automation service"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        logger.info("AutomationService shutdown")
