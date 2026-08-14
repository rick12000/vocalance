from __future__ import annotations

import re

from vocalance.app.config.app_config import DictationConfig


def strip_config_phrases_case_insensitive(text: str, phrases: tuple[str, ...]) -> str:
    s = " ".join(text.split()).strip()
    if not s:
        return ""
    nonempty = [p.strip() for p in phrases if p and p.strip()]
    nonempty.sort(key=lambda p: (len(p.split()), len(p)), reverse=True)
    for phrase in nonempty:
        pat = r"(?i)\b" + re.escape(phrase) + r"\b"
        s = re.sub(pat, " ", s)
        s = " ".join(s.split()).strip()
    return s


def strip_dictation_triggers(text: str, cfg: DictationConfig) -> str:
    if not text:
        return ""
    trigger_phrases: tuple[str, ...] = (
        cfg.start_trigger,
        cfg.stop_trigger,
        cfg.type_trigger,
        cfg.smart_start_trigger,
        cfg.visual_start_trigger,
        cfg.hidden_start_trigger,
        cfg.amend_start_trigger,
    )
    stripped: str = strip_config_phrases_case_insensitive(text, trigger_phrases)
    modifier_phrases: tuple[str, ...] = (
        cfg.modifier_upper_phrase,
        cfg.modifier_capitals_phrase,
        cfg.modifier_camel_phrase,
        cfg.modifier_snake_phrase,
        cfg.modifier_spelling_phrase,
        cfg.modifier_kebab_phrase,
        cfg.modifier_diminish_phrase,
        cfg.modifier_strip_phrase,
        cfg.modifier_numeral_phrase,
    )
    return strip_config_phrases_case_insensitive(stripped, modifier_phrases)
