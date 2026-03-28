"""Shared post-processing for dictation transcripts (engine-agnostic)."""

import re


def normalize_dictation_text(text: str) -> str:
    """Normalize transcribed text by removing filler words and consecutive duplicates.

    Args:
        text: Raw transcribed text.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    text = text.strip()
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(um|uh|like|so)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(um|uh|like|so)$", "", text, flags=re.IGNORECASE)

    words = text.split()
    if len(words) > 1:
        result = [words[0]]
        for word in words[1:]:
            if word.lower() != result[-1].lower():
                result.append(word)
        text = " ".join(result)

    return text.strip()
