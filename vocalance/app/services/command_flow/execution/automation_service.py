import time
from typing import Callable, Dict, Optional

import pyautogui

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import ActionType, ParameterizedCommand
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import AutomationCommandParsedEvent
from vocalance.app.events.command_management_events import CommandMappingsUpdatedEvent
from vocalance.app.services.base_service import Service
from vocalance.app.services.keyboard_input_service import KeyboardInputService


class AutomationService(Service):
    """Runs automation commands (hotkeys, clicks, scrolls) via pyautogui on a thread pool."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, input_service: KeyboardInputService) -> None:
        super().__init__(event_bus)
        self.config = config
        self.input_service = input_service
        self.cooldown_timers: Dict[str, float] = {}
        self.subscribe(AutomationCommandParsedEvent, self.handle_automation_command_parsed)
        self.subscribe(CommandMappingsUpdatedEvent, self.handle_command_mappings_updated)

    async def handle_automation_command_parsed(self, event: AutomationCommandParsedEvent) -> None:
        command = event.command
        count = getattr(command, "count", 1)
        if isinstance(command, ParameterizedCommand) and count <= 0:
            return
        if not self.check_cooldown(command.command_key):
            return
        action_fn = self.create_action_function(command.action_type, command.action_value)
        if not action_fn:
            return
        await self.input_service.run(self.run_action, action_fn, count)
        self.cooldown_timers[command.command_key] = time.time()

    def run_action(self, action_fn: Callable[[], None], count: int) -> None:
        for _ in range(count):
            action_fn()

    def create_action_function(self, action_type: ActionType, action_value: str) -> Optional[Callable[[], None]]:
        if action_type == "hotkey":
            keys = [k.strip() for k in action_value.replace(" ", "+").split("+")]
            return lambda: pyautogui.hotkey(*keys)
        if action_type == "key":
            return lambda: pyautogui.press(action_value)
        if action_type == "key_sequence":
            key_list = [k.strip() for k in action_value.split(",")]
            return lambda: self.execute_key_sequence(key_list)
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
                return lambda: self.execute_animated_scroll(action_value)
        return None

    def execute_key_sequence(self, key_list: list[str]) -> None:
        for combo in key_list:
            if "+" in combo:
                pyautogui.hotkey(*[k.strip() for k in combo.split("+")])
            else:
                pyautogui.press(combo.strip())
            time.sleep(self.config.automation_service.key_sequence_delay_seconds)

    def execute_animated_scroll(self, direction: str) -> None:
        cfg = self.config.automation_service
        multiplier = 1 if direction == "up" else -1
        clicks_per_step = cfg.scroll_total_clicks // cfg.scroll_animation_steps
        remainder = cfg.scroll_total_clicks % cfg.scroll_animation_steps
        for step in range(cfg.scroll_animation_steps):
            step_clicks = clicks_per_step + (1 if step < remainder else 0)
            pyautogui.scroll(step_clicks * multiplier)
            if step < cfg.scroll_animation_steps - 1:
                time.sleep(cfg.scroll_animation_delay_seconds)

    def check_cooldown(self, command_key: str) -> bool:
        return time.time() - self.cooldown_timers.get(command_key, 0) >= self.config.automation_cooldown_seconds

    async def handle_command_mappings_updated(self, _: CommandMappingsUpdatedEvent) -> None:
        self.cooldown_timers.clear()
