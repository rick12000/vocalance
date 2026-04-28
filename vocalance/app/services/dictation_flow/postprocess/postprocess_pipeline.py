from __future__ import annotations

from typing import Optional

from vocalance.app.events.dictation_events import DictationModifierId
from vocalance.app.services.dictation_flow.postprocess.base_postprocess import apply_base_postprocess
from vocalance.app.services.dictation_flow.postprocess.modifier_postprocess import apply_modifier_transform


def apply_dictation_postprocess(
    text: str,
    active_modifiers: Optional[set[DictationModifierId]],
    explicit_modifiers: Optional[set[DictationModifierId]] = None,
    accumulated_text: str = "",
) -> str:
    if not text:
        return text
    result: str = apply_base_postprocess(text)
    if not active_modifiers:
        return result
    return apply_modifier_transform(result, active_modifiers, explicit_modifiers, accumulated_text)


def apply_dictation_postprocess_partial(
    text: str,
    active_modifiers: Optional[set[DictationModifierId]],
    explicit_modifiers: Optional[set[DictationModifierId]] = None,
    accumulated_text: str = "",
) -> str:
    if not text:
        return text
    result: str = apply_base_postprocess(text)
    if not active_modifiers or active_modifiers == {"spelling"}:
        return result
    partial_mods: set[DictationModifierId] = active_modifiers - {"spelling"}
    return apply_modifier_transform(result, partial_mods, explicit_modifiers, accumulated_text)
