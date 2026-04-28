from __future__ import annotations

import re
from typing import Optional

from vocalance.app.events.dictation_events import DictationModifierId
from vocalance.app.services.dictation_flow.types import DictationMode


def dictation_segment_input_options(mode: DictationMode, modifiers: Optional[set[DictationModifierId]]) -> tuple[bool, bool]:
    skip_join = False
    if modifiers:
        skip_join = bool(modifiers.intersection({"camel", "snake", "kebab", "spelling"}))
    add_trailing = mode != DictationMode.TYPE and not skip_join
    return add_trailing, skip_join


def is_isolated_stt_noise_fragment(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t in ("?", "？", "¿", "\ufffd", ""):
        return True
    if len(t) <= 2 and all(not (c.isalnum() or c == "_") for c in t):
        return True
    return False


def is_likely_hallucination_fragment(text: str, prev_text: str = "") -> bool:
    if not text or len(text) < 3:
        return False
    words: list[str] = text.split()
    if len(words) > 10:
        last_words: list[str] = words[-10:]
        unique_words: set[str] = set(last_words)
        if len(unique_words) <= 2 and all(len(w) <= 2 for w in unique_words):
            return True
    if prev_text and not any(ord(c) > 127 for c in prev_text):
        ascii_count: int = sum(1 for c in text if ord(c) < 128)
        if len(text) > 10 and ascii_count < len(text) * 0.3:
            return True
    return False


def remove_stop_trigger_word(text: str, stop_trigger: str) -> str:
    if not stop_trigger or not text:
        return text
    pattern = r"\b" + re.escape(stop_trigger) + r"\b"
    result = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return " ".join(result.split())
