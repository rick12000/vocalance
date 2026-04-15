from __future__ import annotations

import re

from vocalance.app.utils.number_parser import replace_spoken_numbers_in_text


def strip_trailing_period_after_numbers(text: str) -> str:
    if not text:
        return text
    s: str = re.sub(r"(\d)\s+\.(?=\s|$)", r"\1 ", text)
    s = re.sub(r"(\d)\.(?=\s|$)", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def apply_base_postprocess(text: str) -> str:
    if not text:
        return text
    s: str = re.sub(r"\s+", " ", text.strip())
    if not s:
        return ""
    s = replace_spoken_numbers_in_text(s, apply_homophones=False)
    return strip_trailing_period_after_numbers(s)
