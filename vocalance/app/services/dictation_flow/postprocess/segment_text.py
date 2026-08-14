from __future__ import annotations

import re


def clean_dictation_text(text: str, add_trailing_space: bool = True) -> str:
    if not text:
        return ""
    cleaned: str = re.sub(r"\.\.\.", " ", text)
    if add_trailing_space and cleaned and not cleaned[-1].isspace():
        cleaned = cleaned + " "
    return cleaned


def should_add_period_before(last_text: str, current_text: str) -> bool:
    if not last_text or not current_text:
        return False
    if last_text[-1] in ".?!":
        return False
    words = current_text.split(maxsplit=1)
    if not words:
        return False
    first_word = words[0]
    return first_word[0].isupper() and not first_word.isupper()


def remove_formatting(text: str, is_first_word_of_session: bool = False) -> str:
    if not text:
        return ""
    cleaned: str = re.sub(r"[^\w\s\-']", "", text)
    cleaned = cleaned.lower()
    words: list[str] = cleaned.split()
    if words:
        if is_first_word_of_session:
            words[0] = words[0].capitalize()
        words = [word if not (word == "i" or word.startswith("i'")) else word.replace("i", "I", 1) for word in words]
        cleaned = " ".join(words)
    return cleaned.strip()
