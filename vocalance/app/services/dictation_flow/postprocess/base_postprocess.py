from __future__ import annotations

import re


def apply_base_postprocess(text: str) -> str:
    if not text:
        return text
    s: str = re.sub(r"\s+", " ", text.strip())
    return s
