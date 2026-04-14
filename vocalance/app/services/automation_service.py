import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import pyautogui

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import ActionType, ParameterizedCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import AutomationCommandParsedEvent
from vocalance.app.events.command_management_events import CommandMappingsUpdatedEvent
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class AutomationService(Service):
    """Execute automation commands (hotkeys, clicks, scrolls) via pyautogui in a thread pool."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        self._event_bus = event_bus
        self._config = config
        self._thread_pool = ThreadPoolExecutor(max_workers=config.automation_service.thread_pool_max_workers)
        self._cooldown_timers: Dict[str, float] = {}
        event_bus.subscribe(AutomationCommandParsedEvent, self._handle_automation_command)
        event_bus.subscribe(CommandMappingsUpdatedEvent, self._handle_command_mappings_updated)

    async def _handle_automation_command(self, event: AutomationCommandParsedEvent) -> None:
        command = event.command
        count = getattr(command, "count", 1)
        if isinstance(command, ParameterizedCommand) and count <= 0:
            return
        if not self._check_cooldown(command.command_key):
            return
        action_fn = self._create_action_function(command.action_type, command.action_value)
        if not action_fn:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._thread_pool, lambda: self._run_action(action_fn, count))
        self._cooldown_timers[command.command_key] = time.time()

    def _run_action(self, action_fn: Callable[[], None], count: int) -> None:
        for _ in range(count):
            action_fn()

    def _create_action_function(self, action_type: ActionType, action_value: str) -> Optional[Callable[[], None]]:
        if action_type == "hotkey":
            keys = [k.strip() for k in action_value.replace(" ", "+").split("+")]
            return lambda: pyautogui.hotkey(*keys)
        if action_type == "key":
            return lambda: pyautogui.press(action_value)
        if action_type == "key_sequence":
            key_list = [k.strip() for k in action_value.split(",")]
            return lambda: self._execute_key_sequence(key_list)
        if action_type == "click":
            return {
                "click": lambda: pyautogui.click(button="left"),
                "left_click": lambda: pyautogui.click(button="left"),
                "right_click": lambda: pyautogui.click(button="right"),
                "double_click": pyautogui.doubleClick,
                "triple_click": pyautogui.tripleClick,
            }.get(action_value)
        if action_type == "scroll":
            if action_value in ("up", "down"):
                return lambda: self._execute_animated_scroll(action_value)
        return None

    def _execute_key_sequence(self, key_list: list[str]) -> None:
        for combo in key_list:
            if "+" in combo:
                pyautogui.hotkey(*[k.strip() for k in combo.split("+")])
            else:
                pyautogui.press(combo.strip())
            time.sleep(self._config.automation_service.key_sequence_delay_seconds)

    def _execute_animated_scroll(self, direction: str) -> None:
        cfg = self._config.automation_service
        multiplier = 1 if direction == "up" else -1
        clicks_per_step = cfg.scroll_total_clicks // cfg.scroll_animation_steps
        remainder = cfg.scroll_total_clicks % cfg.scroll_animation_steps
        for step in range(cfg.scroll_animation_steps):
            step_clicks = clicks_per_step + (1 if step < remainder else 0)
            pyautogui.scroll(step_clicks * multiplier)
            if step < cfg.scroll_animation_steps - 1:
                time.sleep(cfg.scroll_animation_delay_seconds)

    def _check_cooldown(self, command_key: str) -> bool:
        return time.time() - self._cooldown_timers.get(command_key, 0) >= self._config.automation_cooldown_seconds

    async def _handle_command_mappings_updated(self, _: CommandMappingsUpdatedEvent) -> None:
        self._cooldown_timers.clear()

    async def shutdown(self) -> None:
        self._event_bus.unsubscribe(AutomationCommandParsedEvent, self._handle_automation_command)
        self._event_bus.unsubscribe(CommandMappingsUpdatedEvent, self._handle_command_mappings_updated)
        await asyncio.to_thread(self._thread_pool.shutdown, wait=True)
