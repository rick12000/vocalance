from __future__ import annotations

from typing import Dict, List, Union

import pyautogui
from pydantic import BaseModel, ConfigDict

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import (
    AutomationCommand,
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
    RepeatCommand,
)
from vocalance.app.utils.number_parser import parse_number


class CommandParserTriggers(BaseModel):
    """Lowercased trigger phrases derived once from ``GlobalAppConfig``."""

    model_config = ConfigDict(frozen=True)

    grid_show_phrase: str
    grid_hover_phrase: str
    grid_drag_phrase: str
    mark_create_prefix: str
    mark_delete_prefix: str
    mark_visualize_phrases: tuple[str, ...]
    mark_reset_phrases: tuple[str, ...]
    mark_cancel_visualize_phrases: tuple[str, ...]
    dictation_start_trigger: str
    dictation_stop_trigger: str
    dictation_type_trigger: str
    dictation_smart_trigger: str
    dictation_visual_trigger: str
    dictation_hidden_trigger: str
    dictation_amend_trigger: str


def build_triggers_from_config(config: GlobalAppConfig) -> CommandParserTriggers:
    """Build frozen trigger strings from the live ``GlobalAppConfig`` subsections."""
    g = config.grid
    m = config.mark.triggers
    d = config.dictation
    return CommandParserTriggers(
        grid_show_phrase=g.show_grid_phrase.lower(),
        grid_hover_phrase=g.hover_grid_phrase.lower(),
        grid_drag_phrase=g.drag_grid_phrase.lower(),
        mark_create_prefix=m.create_mark.lower(),
        mark_delete_prefix=m.delete_mark.lower(),
        mark_visualize_phrases=tuple(p.lower() for p in m.visualize_marks),
        mark_reset_phrases=tuple(p.lower() for p in m.reset_marks),
        mark_cancel_visualize_phrases=tuple(p.lower() for p in m.visualization_cancel),
        dictation_start_trigger=d.start_trigger.lower(),
        dictation_stop_trigger=d.stop_trigger.lower(),
        dictation_type_trigger=d.type_trigger.lower(),
        dictation_smart_trigger=d.smart_start_trigger.lower(),
        dictation_visual_trigger=d.visual_start_trigger.lower(),
        dictation_hidden_trigger=d.hidden_start_trigger.lower(),
        dictation_amend_trigger=d.amend_start_trigger.lower(),
    )


SYSTEM_CONTROL_PHRASES = ("pause", "resume", "repeat")


def parse_system_control(normalized_text: str) -> ParseResultType:
    """Match global pause/resume/repeat phrases."""
    if normalized_text == "pause":
        return PauseCommand()
    if normalized_text == "resume":
        return ResumeCommand()
    if normalized_text == "repeat":
        return RepeatCommand()
    return NoMatchResult()


def parse_dictation(normalized_text: str, triggers: CommandParserTriggers) -> ParseResultType:
    """Match dictation mode start/stop and variant triggers."""
    if normalized_text == triggers.dictation_start_trigger:
        return DictationStartCommand()
    if normalized_text == triggers.dictation_stop_trigger:
        return DictationStopCommand()
    if normalized_text == triggers.dictation_type_trigger:
        return DictationTypeCommand()
    if normalized_text == triggers.dictation_smart_trigger:
        return DictationSmartStartCommand()
    if normalized_text == triggers.dictation_visual_trigger:
        return DictationVisualStartCommand()
    if normalized_text == triggers.dictation_hidden_trigger:
        return DictationHiddenStartCommand()
    if normalized_text == triggers.dictation_amend_trigger:
        return DictationAmendStartCommand()
    return NoMatchResult()


def parse_mark_commands(normalized_text: str, triggers: CommandParserTriggers) -> ParseResultType:
    """Parse mark create/delete/visualize/reset/cancel from ``normalized_text``."""
    words = normalized_text.split()
    if not words:
        return NoMatchResult()

    if words[0] == triggers.mark_create_prefix and len(words) == 2:
        label = words[1]
        if not label:
            return ErrorResult(error_message="Mark label cannot be empty")
        x, y = pyautogui.position()
        return MarkCreateCommand(label=label, x=float(x), y=float(y))

    if normalized_text.startswith(f"{triggers.mark_delete_prefix} "):
        label_part = normalized_text[len(triggers.mark_delete_prefix) :].strip()
        if label_part and len(label_part.split()) == 1:
            return MarkDeleteCommand(label=label_part)
        return ErrorResult(error_message="Mark delete requires a single word label")

    if normalized_text in triggers.mark_visualize_phrases:
        return MarkVisualizeCommand()
    if normalized_text in triggers.mark_reset_phrases:
        return MarkResetCommand()
    if normalized_text in triggers.mark_cancel_visualize_phrases:
        return MarkVisualizeCancelCommand()

    return NoMatchResult()


def grid_show_from_phrase(normalized_text: str, phrase: str, click_mode: str) -> Union[GridShowCommand, ErrorResult, None]:
    """Parse ``grid show`` variants with optional rectangle count for ``click_mode``."""
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


def parse_grid_commands(
    normalized_text: str, triggers: CommandParserTriggers, action_map: Dict[str, AutomationCommand]
) -> ParseResultType:
    """Resolve grid overlay show/hover/drag and numeric cell selection without stealing automation prefixes."""
    words = normalized_text.split()
    if not words:
        return NoMatchResult()

    for phrase, mode in (
        (triggers.grid_show_phrase, "click"),
        (triggers.grid_hover_phrase, "hover"),
        (triggers.grid_drag_phrase, "drag"),
    ):
        got = grid_show_from_phrase(normalized_text, phrase, mode)
        if got is not None:
            return got

    for i in range(1, len(words) + 1):
        if " ".join(words[:i]) in action_map:
            return NoMatchResult()

    n = parse_number(text=normalized_text)
    if n is not None and n > 0:
        return GridSelectCommand(selected_number=n)
    return NoMatchResult()


def parse_automation_commands(normalized_text: str, action_map: Dict[str, AutomationCommand]) -> ParseResultType:
    """Match stored exact or parameterized automation phrases."""
    words = normalized_text.split()
    if not words:
        return NoMatchResult()

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


def parse_mark_execute_fallback(normalized_text: str) -> ParseResultType:
    """Treat a single-token phrase as ``MarkExecuteCommand`` when nothing else matched."""
    words = normalized_text.split()
    if len(words) == 1:
        return MarkExecuteCommand(label=normalized_text)
    return NoMatchResult()


def parse_full_command_text(
    normalized_text: str, triggers: CommandParserTriggers, action_map: Dict[str, AutomationCommand]
) -> ParseResultType:
    """Run the ordered pipeline: system → dictation → marks → grid → automation → mark execute fallback."""
    if not normalized_text:
        return NoMatchResult()

    steps: List[ParseResultType] = [
        parse_system_control(normalized_text),
        parse_dictation(normalized_text, triggers),
        parse_mark_commands(normalized_text, triggers),
        parse_grid_commands(normalized_text, triggers, action_map),
        parse_automation_commands(normalized_text, action_map),
        parse_mark_execute_fallback(normalized_text),
    ]
    for result in steps:
        if not isinstance(result, NoMatchResult):
            return result
    return NoMatchResult()
