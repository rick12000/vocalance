"""Spoken and numeric text parsing: one pipeline for commands, dictation, and inline replacement.

Homophone mapping (e.g. *to* → *two*, *won* → *one*) is **on by default** for command recognition.
Dictation passes ``apply_homophones=False`` into :func:`replace_spoken_numbers_in_text` to avoid
rewriting ordinary words.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

NUMBER_WORDS: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

SCALE_WORDS: Set[str] = {"hundred", "thousand", "million", "billion", "trillion"}
SINGLE_DIGIT_WORDS: Set[str] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}

HOMOPHONES: Dict[str, str] = {
    "won": "one",
    "to": "two",
    "too": "two",
    "free": "three",
    "for": "four",
    "fore": "four",
    "ate": "eight",
}

_ORDINAL_WORDS: Dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
}

_SCALES_NON_HUNDRED: Dict[str, int] = {
    "thousand": 10**3,
    "million": 10**6,
    "billion": 10**9,
    "trillion": 10**12,
}


def is_number(text: str) -> bool:
    """True if ``text`` parses as a float (allows commas; e.g. ``1,234``)."""
    if isinstance(text, str):
        text = text.replace(",", "")
    try:
        float(text)
    except Exception:
        return False
    return True


def normalize_homophones(text: str, *, apply_homophones: bool = True) -> str:
    """Lowercase and collapse whitespace; optionally map ASR homophones to number words.

    Default ``True`` matches command recognition. Use ``False`` for dictation-style text.
    """
    if not isinstance(text, str):
        return text

    words = text.lower().split()
    if not apply_homophones:
        return " ".join(words)
    return " ".join(HOMOPHONES.get(word, word) for word in words)


def remove_number_conjunctions(text: str) -> str:
    """Drop *and* when it sits between number-related words (``four hundred and nine`` → ``four hundred nine``)."""
    if not isinstance(text, str):
        return text

    words = text.lower().split()
    all_number_words = NUMBER_WORDS.keys() | SCALE_WORDS

    filtered_words: List[str] = []
    for i, word in enumerate(words):
        if word == "and":
            prev_is_number = i > 0 and words[i - 1] in all_number_words
            next_is_number = i < len(words) - 1 and words[i + 1] in all_number_words
            if prev_is_number and next_is_number:
                continue
        filtered_words.append(word)

    return " ".join(filtered_words)


def detect_digit_sequence(text: str) -> Optional[str]:
    """If all tokens are spoken digits (no scales), return concatenated digits (``four zero nine`` → ``409``)."""
    if not isinstance(text, str):
        return None

    words = text.lower().split()

    if len(words) > 1 and all(word in SINGLE_DIGIT_WORDS for word in words) and not any(
        word in SCALE_WORDS for word in words
    ):
        digit_map = {word: str(NUMBER_WORDS[word]) for word in SINGLE_DIGIT_WORDS}
        return "".join(digit_map[word] for word in words)

    return None


def normalize_spoken_number_phrase(text: str, *, apply_homophones: bool = True) -> str:
    """Normalize whitespace and hyphens; optional homophones; remove number-only *and*."""
    if not text or not isinstance(text, str):
        return ""
    s = text.replace("-", " ").strip()
    s = " ".join(s.split())
    if not s:
        return ""
    s = normalize_homophones(s, apply_homophones=apply_homophones)
    s = remove_number_conjunctions(s)
    return " ".join(s.split())


def parse_cardinal_words(words: List[str]) -> Optional[int]:
    """Parse lowercase cardinal tokens as one integer, or None if invalid."""
    if not words:
        return None
    for w in words:
        if w not in NUMBER_WORDS and w not in SCALE_WORDS:
            return None

    total = 0
    current = 0
    i = 0
    while i < len(words):
        w = words[i]
        if w == "hundred":
            if current == 0:
                current = 1
            current *= 100
            i += 1
            continue
        if w in _SCALES_NON_HUNDRED:
            mag = _SCALES_NON_HUNDRED[w]
            if current == 0:
                current = 1
            total += current * mag
            current = 0
            i += 1
            continue
        if w in NUMBER_WORDS:
            current += NUMBER_WORDS[w]
            i += 1
            continue
        return None

    return total + current


def _parse_single_ordinal_token(word: str) -> Optional[int]:
    """Map one lowercase token (``first``, ``twentieth``) to its integer, if recognized."""
    w = word.lower().strip()
    if not w:
        return None
    return _ORDINAL_WORDS.get(w)


def _strip_token_punct(token: str) -> Tuple[str, str, str]:
    """Return ``(leading, core, trailing)`` punctuation split for a whitespace token."""
    m = re.match(r"^(['\"(\[{<]*)(.*?)(['\")\]}>:;,.!?]*)$", token)
    if not m:
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)


def parse_spoken_integer(text: Optional[str], *, apply_homophones: bool = True) -> Optional[int]:
    """Parse a single spoken or numeric phrase into an integer.

    Pipeline: strip → ASCII digits → normalize (hyphens, optional homophones, *and*) →
    digit-by-digit sequence → cardinals → single-word ordinals.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return None

    if is_number(text):
        try:
            return int(float(text.replace(",", "")))
        except ValueError:
            return None

    prepared = normalize_spoken_number_phrase(text, apply_homophones=apply_homophones)
    if not prepared:
        return None

    if is_number(prepared):
        try:
            return int(float(prepared.replace(",", "")))
        except ValueError:
            return None

    digit_sequence = detect_digit_sequence(prepared)
    if digit_sequence is not None:
        try:
            return int(digit_sequence)
        except ValueError:
            return None

    words = prepared.lower().split()
    cardinal = parse_cardinal_words(words)
    if cardinal is not None:
        return cardinal

    if len(words) == 1:
        return _parse_single_ordinal_token(words[0])

    return None


def _try_spoken_number_from_words(core_words: List[str], *, apply_homophones: bool = True) -> Optional[int]:
    """Parse a run of word cores (from inline token scanning) using :func:`parse_spoken_integer`."""
    if not core_words:
        return None
    return parse_spoken_integer(" ".join(core_words), apply_homophones=apply_homophones)


def replace_spoken_numbers_in_text(
    text: str, max_words_per_number: int = 12, *, apply_homophones: bool = True
) -> str:
    """Replace maximal runs of spoken number words with digit strings; other tokens unchanged.

    Default ``True`` for command-style text. Dictation should pass ``apply_homophones=False``.
    """
    if not text or not text.strip():
        return text

    raw_tokens = text.split()
    if not raw_tokens:
        return text

    rebuilt: List[str] = []
    i = 0
    while i < len(raw_tokens):
        best_val: Optional[int] = None
        best_len = 0
        for take in range(min(max_words_per_number, len(raw_tokens) - i), 0, -1):
            core_parts: List[str] = []
            valid = True
            for t in raw_tokens[i : i + take]:
                _, c, _ = _strip_token_punct(t)
                if not c:
                    valid = False
                    break
                if c.replace(",", "").isdigit():
                    valid = False
                    break
                core_parts.append(c.lower())

            if not valid or not core_parts:
                continue

            n = _try_spoken_number_from_words(core_parts, apply_homophones=apply_homophones)
            if n is not None:
                best_val = n
                best_len = take
                break

        if best_val is not None and best_len > 0:
            fl, _, _ = _strip_token_punct(raw_tokens[i])
            _, _, lt = _strip_token_punct(raw_tokens[i + best_len - 1])
            rebuilt.append(f"{fl}{best_val}{lt}")
            i += best_len
        else:
            rebuilt.append(raw_tokens[i])
            i += 1

    return " ".join(rebuilt)


def parse_number(
    text: Optional[str],
    min_value: int = 1,
    max_value: int = 5000,
    *,
    apply_homophones: bool = True,
) -> Optional[int]:
    """Parse text to an integer and enforce ``[min_value, max_value]``.

    Homophones apply by default for commands; not used from dictation (that path uses
    :func:`replace_spoken_numbers_in_text` with ``apply_homophones=False``).
    """
    if text is None or text == "":
        return None

    if isinstance(text, (int, float)):
        text = str(text)
    elif not isinstance(text, str):
        return None

    n = parse_spoken_integer(text.strip(), apply_homophones=apply_homophones)
    if n is None:
        return None
    if min_value <= n <= max_value:
        return n
    return None
