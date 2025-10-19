import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import pyautogui

from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.command_types import ActionType, BaseCommand, ParameterizedCommand
from iris.app.event_bus import EventBus
from iris.app.events.command_events import AutomationCommandParsedEvent
from iris.app.events.command_management_events import CommandMappingsUpdatedEvent
from iris.app.events.core_events import CommandExecutedStatusEvent

logger = logging.getLogger(__name__)


class AutomationService:
    """Service for executing automation commands with cooldown management.

    Processes automation commands through pyautogui, manages execution timing
    with per-command cooldowns, and provides async-safe execution with status
    reporting through the event bus.
    """

    def __init__(self, event_bus: EventBus, app_config: GlobalAppConfig) -> None:
        self._event_bus = event_bus
        self._app_config = app_config
        self._thread_pool = ThreadPoolExecutor(max_workers=app_config.automation_service.thread_pool_max_workers)
        self._execution_lock = asyncio.Lock()
        self._cooldown_lock = asyncio.Lock()
        self._cooldown_timers: Dict[str, float] = {}

        logger.info("AutomationService initialized")

    def setup_subscriptions(self) -> None:
        self._event_bus.subscribe(event_type=AutomationCommandParsedEvent, handler=self._handle_automation_command)
        self._event_bus.subscribe(event_type=CommandMappingsUpdatedEvent, handler=self._handle_command_mappings_updated)
        logger.info("AutomationService subscriptions set up")

    async def _handle_automation_command(self, event_data: AutomationCommandParsedEvent) -> None:
        """Process and execute automation commands with cooldown and count handling.

        Validates command parameters, checks cooldown status, executes the command
        through thread pool, and publishes execution status events.

        Args:
            event_data: Event containing the automation command to execute
        """
        command = event_data.command
        count = getattr(command, "count", 1)

        if isinstance(command, ParameterizedCommand) and count <= 0:
            await self._publish_status(command, event_data.source, False, f"Invalid repeat count: {count}")
            return

        if not await self._check_cooldown(command.command_key):
            await self._publish_status(command, event_data.source, False, f"Command '{command.command_key}' is on cooldown")
            return

        success = await self._execute_command(command.action_type, command.action_value, count)

        if success:
            async with self._cooldown_lock:
                self._cooldown_timers[command.command_key] = time.time()

        count_text = f" {count} times" if count > 1 else ""
        status = "successfully" if success else "failed"
        message = f"Command '{command.command_key}' executed{count_text} {status}"
        await self._publish_status(command, event_data.source, success, message)

    async def _execute_command(self, action_type: ActionType, action_value: str, count: int = 1) -> bool:
        """Execute automation action in thread pool.

        Creates action function from type and value, then executes it in a
        thread pool to avoid blocking the event loop.

        Args:
            action_type: Type of automation action (hotkey, key, click, scroll, etc.)
            action_value: Value/parameter for the action
            count: Number of times to repeat the action

        Returns:
            True if execution succeeded, False otherwise
        """
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

    def _execute_action(self, action_function: Callable[[], None], count: int) -> bool:
        for _ in range(count):
            action_function()
        return True

    def _create_action_function(self, action_type: ActionType, action_value: str) -> Optional[Callable[[], None]]:
        """Create pyautogui action function from action type and value.

        Maps action types to corresponding pyautogui function calls with proper
        parameter handling for hotkeys, key sequences, clicks, and scrolls.

        Args:
            action_type: Type of automation action
            action_value: Value/parameter for the action

        Returns:
            Callable function that executes the action, or None if invalid
        """
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

        return None

    def _execute_key_sequence(self, key_list: list[str]) -> None:
        for key_combination in key_list:
            if "+" in key_combination:
                keys = [k.strip() for k in key_combination.split("+")]
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(key_combination.strip())
            time.sleep(self._app_config.automation_service.key_sequence_delay_seconds)

    async def _check_cooldown(self, command_key: str) -> bool:
        current_time = time.time()
        async with self._cooldown_lock:
            last_execution = self._cooldown_timers.get(command_key, 0)
        cooldown_period = self._app_config.automation_cooldown_seconds
        return current_time - last_execution >= cooldown_period

    async def _publish_status(self, command: BaseCommand, source: Optional[str], success: bool, message: str) -> None:
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

    async def _handle_command_mappings_updated(self, event_data: CommandMappingsUpdatedEvent) -> None:
        async with self._cooldown_lock:
            self._cooldown_timers.clear()
        logger.info("Cleared automation command cooldown timers after mappings update")

    async def shutdown(self) -> None:
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        logger.info("AutomationService shutdown")
