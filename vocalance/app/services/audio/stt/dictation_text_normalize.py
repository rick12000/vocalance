import re


def normalize_dictation_text(text: str) -> str:
    """Collapse runs of whitespace (engine-agnostic)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())
