"""Turn recognized text (and sound mappings) into typed commands on the event bus."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Union

import pyautogui

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
    ErrorResult,
    ExactMatchCommand,
    GridSelectCommand,
    GridShowCommand,
    MarkCreateCommand,
    MarkDeleteCommand,
    MarkExecuteCommand,
    MarkResetCommand,
    MarkVisualizeCancelCommand,
    MarkVisualizeCommand,
    NoMatchResult,
    ParameterizedCommand,
    ParseResultType,
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
from vocalance.app.services.commands.action_map_provider import CommandActionMapProvider
from vocalance.app.services.deduplication.event_deduplicator import EventDeduplicator
from vocalance.app.services.pause_state_manager import PauseStateManager
from vocalance.app.utils.number_parser import parse_number

logger = logging.getLogger(__name__)

_PARSED_EVENT_BY_COMMAND: Dict[type, Any] = {
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
    """Ordered parse pipeline (system → dictation → marks → grid → automation → mark click)."""

    def __init__(
        self,
        event_bus: EventBus,
        app_config: GlobalAppConfig,
        action_map_provider: CommandActionMapProvider,
        deduplicator: Optional[EventDeduplicator] = None,
        pause_state_manager: Optional[PauseStateManager] = None,
    ) -> None:
        self._event_bus = event_bus
        self._app_config = app_config
        self._action_map_provider = action_map_provider
        self._sound_to_command_mapping: Dict[str, str] = {}
        self._pause_state_manager = pause_state_manager
        self._load_trigger_strings()
        self._deduplicator = deduplicator or EventDeduplicator(window_ms=app_config.command_parser.duplicate_detection_window_ms)

        event_bus.subscribe(CommandTextRecognizedEvent, self._handle_command_text_recognized)
        event_bus.subscribe(CustomSoundRecognizedEvent, self._handle_custom_sound_recognized)
        event_bus.subscribe(SoundToCommandMappingUpdatedEvent, self._handle_sound_mapping_updated)
        event_bus.subscribe(SoundMappingsResponseEvent, self._handle_sound_mappings_response)

    def _load_trigger_strings(self) -> None:
        g = self._app_config.grid
        self._grid_show_phrase = g.show_grid_phrase.lower()
        self._grid_hover_phrase = g.hover_grid_phrase.lower()
        self._grid_drag_phrase = g.drag_grid_phrase.lower()
        m = self._app_config.mark.triggers
        self._mark_create_prefix = m.create_mark.lower()
        self._mark_delete_prefix = m.delete_mark.lower()
        self._mark_visualize_phrases = [p.lower() for p in m.visualize_marks]
        self._mark_reset_phrases = [p.lower() for p in m.reset_marks]
        self._mark_cancel_visualize_phrases = [p.lower() for p in m.visualization_cancel]
        d = self._app_config.dictation
        self._dictation_start_trigger = d.start_trigger.lower()
        self._dictation_stop_trigger = d.stop_trigger.lower()
        self._dictation_type_trigger = d.type_trigger.lower()
        self._dictation_smart_trigger = d.smart_start_trigger.lower()
        self._dictation_visual_trigger = d.visual_start_trigger.lower()
        self._dictation_hidden_trigger = d.hidden_start_trigger.lower()
        self._dictation_amend_trigger = d.amend_start_trigger.lower()

    async def initialize(self) -> bool:
        return True

    async def shutdown(self) -> None:
        for event_type, handler in [
            (CommandTextRecognizedEvent, self._handle_command_text_recognized),
            (CustomSoundRecognizedEvent, self._handle_custom_sound_recognized),
            (SoundToCommandMappingUpdatedEvent, self._handle_sound_mapping_updated),
            (SoundMappingsResponseEvent, self._handle_sound_mappings_response),
        ]:
            self._event_bus.unsubscribe(event_type, handler)

    async def _process_text_input(self, text: str, source: Optional[str] = None) -> None:
        src = source or "unknown"
        if self._deduplicator.should_deduplicate(text, source=src):
            return

        parsed = await self._parse_text(text)

        if isinstance(parsed, BaseCommand):
            if self._pause_state_manager and not isinstance(parsed, ResumeCommand):
                if self._pause_state_manager.is_paused():
                    logger.debug("Application paused — ignoring command: %s", text)
                    return

            await self._publish_command_event(parsed, source)
            self._deduplicator.record_event(text, source=src)

    async def _parse_text(self, text: str) -> ParseResultType:
        normalized = text.lower().strip()
        if not normalized:
            return NoMatchResult()

        parsers = (
            self._parse_system_control_commands,
            self._parse_dictation_commands,
            self._parse_mark_commands,
            self._parse_grid_commands,
            self._parse_automation_commands,
            self._parse_mark_execute_fallback,
        )
        for parse in parsers:
            if asyncio.iscoroutinefunction(parse):
                result = await parse(normalized)
            else:
                result = parse(normalized)
            if not isinstance(result, NoMatchResult):
                return result
        return NoMatchResult()

    def _parse_system_control_commands(self, normalized_text: str) -> ParseResultType:
        if normalized_text == "pause":
            return PauseCommand()
        if normalized_text == "resume":
            return ResumeCommand()
        return NoMatchResult()

    def _parse_dictation_commands(self, normalized_text: str) -> ParseResultType:
        if normalized_text == self._dictation_start_trigger:
            return DictationStartCommand()
        if normalized_text == self._dictation_stop_trigger:
            return DictationStopCommand()
        if normalized_text == self._dictation_type_trigger:
            return DictationTypeCommand()
        if normalized_text == self._dictation_smart_trigger:
            return DictationSmartStartCommand()
        if normalized_text == self._dictation_visual_trigger:
            return DictationVisualStartCommand()
        if normalized_text == self._dictation_hidden_trigger:
            return DictationHiddenStartCommand()
        if normalized_text == self._dictation_amend_trigger:
            return DictationAmendStartCommand()
        return NoMatchResult()

    def _parse_mark_commands(self, normalized_text: str) -> ParseResultType:
        words = normalized_text.split()
        if not words:
            return NoMatchResult()

        if words[0] == self._mark_create_prefix and len(words) == 2:
            label = words[1]
            if not label:
                return ErrorResult(error_message="Mark label cannot be empty")
            x, y = pyautogui.position()
            return MarkCreateCommand(label=label, x=float(x), y=float(y))

        if normalized_text.startswith(f"{self._mark_delete_prefix} "):
            label_part = normalized_text[len(self._mark_delete_prefix) :].strip()
            if label_part and len(label_part.split()) == 1:
                return MarkDeleteCommand(label=label_part)
            return ErrorResult(error_message="Mark delete requires a single word label")

        if normalized_text in self._mark_visualize_phrases:
            return MarkVisualizeCommand()
        if normalized_text in self._mark_reset_phrases:
            return MarkResetCommand()
        if normalized_text in self._mark_cancel_visualize_phrases:
            return MarkVisualizeCancelCommand()

        return NoMatchResult()

    def _parse_grid_show_for_phrase(
        self, normalized_text: str, phrase: str, click_mode: str
    ) -> Union[GridShowCommand, ErrorResult, None]:
        if not normalized_text.startswith(phrase):
            return None
        if normalized_text == phrase:
            return GridShowCommand(num_rects=None, click_mode=click_mode)
        rest = normalized_text[len(phrase) :].strip()
        if not rest:
            return None
        n = parse_number(text=rest)
        if n is not None and n > 0:
            return GridShowCommand(num_rects=n, click_mode=click_mode)
        return ErrorResult(error_message=f"Invalid number of rectangles: '{rest}'")

    async def _parse_grid_commands(self, normalized_text: str) -> ParseResultType:
        words = normalized_text.split()
        if not words:
            return NoMatchResult()

        for phrase, mode in (
            (self._grid_show_phrase, "click"),
            (self._grid_hover_phrase, "hover"),
            (self._grid_drag_phrase, "drag"),
        ):
            got = self._parse_grid_show_for_phrase(normalized_text, phrase, mode)
            if got is not None:
                return got

        action_map = await self._action_map_provider.get_action_map()
        for i in range(1, len(words) + 1):
            if " ".join(words[:i]) in action_map:
                return NoMatchResult()

        n = parse_number(text=normalized_text)
        if n is not None and n > 0:
            return GridSelectCommand(selected_number=n)
        return NoMatchResult()

    async def _parse_automation_commands(self, normalized_text: str) -> ParseResultType:
        words = normalized_text.split()
        if not words:
            return NoMatchResult()

        action_map = await self._action_map_provider.get_action_map()

        if normalized_text in action_map:
            spec = action_map[normalized_text]
            return ExactMatchCommand(
                command_key=normalized_text,
                action_type=spec.action_type,
                action_value=spec.action_value,
                is_custom=spec.is_custom,
                short_description=spec.short_description,
                long_description=spec.long_description,
            )

        for i in range(len(words) - 1, 0, -1):
            prefix = " ".join(words[:i])
            if prefix not in action_map:
                continue
            tail = words[i:]
            if len(tail) != 1:
                break
            count = parse_number(text=tail[0])
            if count is None or count <= 0:
                break
            spec = action_map[prefix]
            return ParameterizedCommand(
                command_key=prefix,
                action_type=spec.action_type,
                action_value=spec.action_value,
                count=count,
                is_custom=spec.is_custom,
                short_description=spec.short_description,
                long_description=spec.long_description,
            )

        return NoMatchResult()

    def _parse_mark_execute_fallback(self, normalized_text: str) -> ParseResultType:
        words = normalized_text.split()
        if len(words) == 1:
            return MarkExecuteCommand(label=normalized_text)
        return NoMatchResult()

    async def _publish_command_event(self, command: BaseCommand, source: Optional[str]) -> None:
        event_cls = _PARSED_EVENT_BY_COMMAND.get(type(command))
        if event_cls is None:
            logger.warning("Unknown command type: %s", type(command))
            return
        await self._event_bus.publish(event_cls(source=source, command=command))

    async def _handle_command_text_recognized(self, text_recognized: CommandTextRecognizedEvent) -> None:
        text = text_recognized.text
        await self._process_text_input(text=text, source="stt")

    async def _handle_custom_sound_recognized(self, sound_recognized: CustomSoundRecognizedEvent) -> None:
        phrase = sound_recognized.mapped_command or self._sound_to_command_mapping.get(sound_recognized.label)
        if not phrase:
            logger.warning("No command mapping found for sound: %s", sound_recognized.label)
            return

        await self._process_text_input(text=phrase, source="sound")

    async def _handle_sound_mapping_updated(self, mapping_update: SoundToCommandMappingUpdatedEvent) -> None:
        self._sound_to_command_mapping[mapping_update.sound_label] = mapping_update.command_phrase

    async def _handle_sound_mappings_response(self, mappings_snapshot: SoundMappingsResponseEvent) -> None:
        self._sound_to_command_mapping = mappings_snapshot.mappings
