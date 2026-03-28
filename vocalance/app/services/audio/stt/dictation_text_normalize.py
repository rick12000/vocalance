"""Minimal shared normalization for dictation transcripts (engine-agnostic)."""

import re


def normalize_dictation_text(text: str) -> str:
    """Collapse whitespace only; further cleaning happens in DictationCoordinator."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())
