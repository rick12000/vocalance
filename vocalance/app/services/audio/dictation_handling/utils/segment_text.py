from __future__ import annotations

import re


def clean_dictation_text(text: str, add_trailing_space: bool = True) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\.\.\.", " ", text)
    if add_trailing_space and cleaned and not cleaned[-1].isspace():
        cleaned = cleaned + " "
    return cleaned


def should_remove_previous_period(last_text: str, current_text: str) -> bool:
    if not last_text or not current_text:
        return False
    return last_text.rstrip().endswith(".") and current_text.strip() and current_text.strip()[0].islower()


def should_lowercase_current_start(last_text: str, current_text: str) -> bool:
    if not last_text or not current_text:
        return False
    last_stripped = last_text.rstrip()
    current_stripped = current_text.strip()
    return last_stripped and not last_stripped.endswith(".") and current_stripped and current_stripped[0].isupper()


def get_trailing_whitespace_count(text: str) -> int:
    if not text:
        return 0
    return len(text) - len(text.rstrip())


def lowercase_first_letter(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:] if len(text) > 1 else text[0].lower()


def remove_formatting(text: str, is_first_word_of_session: bool = False) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s\-']", "", text)
    cleaned = cleaned.lower()
    words = cleaned.split()
    if words:
        if is_first_word_of_session:
            words[0] = words[0].capitalize()
        words = [word if not (word == "i" or word.startswith("i'")) else word.replace("i", "I", 1) for word in words]
        cleaned = " ".join(words)
    return cleaned.strip()
