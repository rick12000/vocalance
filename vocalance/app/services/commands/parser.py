"""Subscribe to recognized text / sounds and publish parsed commands on the event bus."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, Type

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import (
    BaseCommand,
    DictationAmendStartCommand,
    DictationHiddenStartCommand,
    DictationSmartStartCommand,
    DictationStartCommand,
    DictationStopCommand,
    DictationTypeCommand,
    DictationVisualStartCommand,
    ExactMatchCommand,
    GridSelectCommand,
    GridShowCommand,
    MarkCreateCommand,
    MarkDeleteCommand,
    MarkExecuteCommand,
    MarkResetCommand,
    MarkVisualizeCancelCommand,
    MarkVisualizeCommand,
    ParameterizedCommand,
    PauseCommand,
    ResumeCommand,
)
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import (
    AutomationCommandParsedEvent,
    DictationCommandParsedEvent,
    GridCommandParsedEvent,
    MarkCommandParsedEvent,
    SystemControlCommandParsedEvent,
)
from vocalance.app.events.core_events import CommandTextRecognizedEvent, CustomSoundRecognizedEvent
from vocalance.app.events.sound_events import SoundMappingsResponseEvent, SoundToCommandMappingUpdatedEvent
from vocalance.app.services.base_service import Service
from vocalance.app.services.commands.utilities.command_projection import load_action_map
from vocalance.app.services.commands.utilities.text_command_parse import (
    CommandParserTriggers,
    build_triggers_from_config,
    parse_full_command_text,
)
from vocalance.app.services.pause_state_manager import PauseStateManager
from vocalance.app.services.storage.storage_service import StorageService

PARSED_EVENT_BY_COMMAND: Dict[Type[BaseCommand], Any] = {
    DictationStartCommand: DictationCommandParsedEvent,
    DictationStopCommand: DictationCommandParsedEvent,
    DictationTypeCommand: DictationCommandParsedEvent,
    DictationSmartStartCommand: DictationCommandParsedEvent,
    DictationVisualStartCommand: DictationCommandParsedEvent,
    DictationHiddenStartCommand: DictationCommandParsedEvent,
    DictationAmendStartCommand: DictationCommandParsedEvent,
    ExactMatchCommand: AutomationCommandParsedEvent,
    ParameterizedCommand: AutomationCommandParsedEvent,
    MarkCreateCommand: MarkCommandParsedEvent,
    MarkExecuteCommand: MarkCommandParsedEvent,
    MarkDeleteCommand: MarkCommandParsedEvent,
    MarkVisualizeCommand: MarkCommandParsedEvent,
    MarkResetCommand: MarkCommandParsedEvent,
    MarkVisualizeCancelCommand: MarkCommandParsedEvent,
    GridShowCommand: GridCommandParsedEvent,
    GridSelectCommand: GridCommandParsedEvent,
    PauseCommand: SystemControlCommandParsedEvent,
    ResumeCommand: SystemControlCommandParsedEvent,
}


class CentralizedCommandParser(Service):
    """Orchestrates text → command: min interval gate, pause gate, one action-map load, parse pipeline, publish."""

    def __init__(
        self,
        event_bus: EventBus,
        app_config: GlobalAppConfig,
        storage: StorageService,
        pause_state_manager: Optional[PauseStateManager] = None,
    ) -> None:
        self.event_bus = event_bus
        self.app_config = app_config
        self.storage = storage
        self.sound_to_command_mapping: Dict[str, str] = {}
        self.pause_state_manager = pause_state_manager
        self.triggers: CommandParserTriggers = build_triggers_from_config(app_config)
        self._command_interval_lock = asyncio.Lock()
        self._last_command_executed_mono: Optional[float] = None

        event_bus.subscribe(CommandTextRecognizedEvent, self.handle_command_text_recognized)
        event_bus.subscribe(CustomSoundRecognizedEvent, self.handle_custom_sound_recognized)
        event_bus.subscribe(SoundToCommandMappingUpdatedEvent, self.handle_sound_mapping_updated)
        event_bus.subscribe(SoundMappingsResponseEvent, self.handle_sound_mappings_response)

    async def initialize(self) -> bool:
        return True

    async def shutdown(self) -> None:
        for event_type, handler in [
            (CommandTextRecognizedEvent, self.handle_command_text_recognized),
            (CustomSoundRecognizedEvent, self.handle_custom_sound_recognized),
            (SoundToCommandMappingUpdatedEvent, self.handle_sound_mapping_updated),
            (SoundMappingsResponseEvent, self.handle_sound_mappings_response),
        ]:
            self.event_bus.unsubscribe(event_type, handler)

    async def process_text_input(self, text: str, source: Optional[str] = None) -> None:
        if not text or not text.strip():
            return
        src = source or "unknown"
        async with self._command_interval_lock:
            now = time.monotonic()
            min_interval_s = self.app_config.command_parser.min_command_interval_ms / 1000.0
            if self._last_command_executed_mono is not None and (now - self._last_command_executed_mono) < min_interval_s:
                return

            normalized = text.lower().strip()
            action_map = await load_action_map(self.storage)
            parsed = parse_full_command_text(normalized, self.triggers, action_map)

            if isinstance(parsed, BaseCommand):
                if self.pause_state_manager and not isinstance(parsed, ResumeCommand):
                    if self.pause_state_manager.is_paused():
                        return

                await self.publish_command_event(parsed, src)
                self._last_command_executed_mono = time.monotonic()

    async def publish_command_event(self, command: BaseCommand, source: Optional[str]) -> None:
        event_cls = PARSED_EVENT_BY_COMMAND.get(type(command))
        if event_cls is None:
            raise ValueError(f"No parsed event registered for command type {type(command).__name__}")
        await self.event_bus.publish(event_cls(source=source, command=command))

    async def handle_command_text_recognized(self, text_recognized: CommandTextRecognizedEvent) -> None:
        await self.process_text_input(text=text_recognized.text, source="stt")

    async def handle_custom_sound_recognized(self, sound_recognized: CustomSoundRecognizedEvent) -> None:
        phrase = sound_recognized.mapped_command or self.sound_to_command_mapping.get(sound_recognized.label)
        if not phrase:
            return
        await self.process_text_input(text=phrase, source="sound")

    async def handle_sound_mapping_updated(self, mapping_update: SoundToCommandMappingUpdatedEvent) -> None:
        self.sound_to_command_mapping[mapping_update.sound_label] = mapping_update.command_phrase

    async def handle_sound_mappings_response(self, mappings_snapshot: SoundMappingsResponseEvent) -> None:
        self.sound_to_command_mapping = mappings_snapshot.mappings
