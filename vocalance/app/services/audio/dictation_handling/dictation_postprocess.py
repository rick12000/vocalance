"""Dictation transcription post-processing: base number normalization and optional modifiers."""

from __future__ import annotations

import re
from typing import Optional

from vocalance.app.events.dictation_events import DictationModifierId
from vocalance.app.utils.number_parser import replace_spoken_numbers_in_text

_MODIFIER_DISPLAY: dict[str, str] = {
    "upper": "Upper",
    "capitals": "Capitals",
    "camel": "Camel",
    "snake": "Snake",
    "spelling": "Spelling",
}


def modifier_display_label(modifier_id: DictationModifierId) -> str:
    """Short UI label for a modifier id (e.g. ``upper`` → ``Upper``)."""
    return _MODIFIER_DISPLAY.get(modifier_id, modifier_id)


def strip_trailing_period_after_numbers(text: str) -> str:
    """Remove a sentence-like period immediately after an integer token; preserve decimals such as ``3.14``."""
    if not text:
        return text
    s = re.sub(r"(\d)\s+\.(?=\s|$)", r"\1 ", text)
    s = re.sub(r"(\d)\.(?=\s|$)", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def apply_base_postprocess(text: str) -> str:
    """Lightweight base layer: spoken cardinals/digit sequences to numerals, then digit/punct cleanup."""
    if not text:
        return text
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return ""
    s = replace_spoken_numbers_in_text(s)
    return strip_trailing_period_after_numbers(s)


def _title_each_word(text: str) -> str:
    def cap_word(w: str) -> str:
        if not w:
            return w
        return w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper()

    return " ".join(cap_word(p) for p in text.split())


def _words_for_camel_snake(text: str) -> list[str]:
    """Split identifier tokens: drop punctuation, treat underscores as separators."""
    s = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return [w for w in s.split() if w]


def _to_camel_case(text: str) -> str:
    """UpperCamelCase (PascalCase): each word capitalized, concatenated, no punctuation."""
    words = _words_for_camel_snake(text)
    if not words:
        return ""
    parts: list[str] = []
    for w in words:
        core = w.lower()
        if not core:
            continue
        if len(core) == 1:
            parts.append(core.upper())
        else:
            parts.append(core[0].upper() + core[1:])
    return "".join(parts)


def _to_snake_case(text: str) -> str:
    """Lowercase words from :func:`_words_for_camel_snake`, joined with underscores."""
    words = _words_for_camel_snake(text)
    return "_".join(w.lower() for w in words if w)


_SPELLING_PUNCT_PHRASES: list[tuple[str, str]] = sorted(
    [
        ("hash tag", "#"),
        ("hashtag", "#"),
        ("hash-tag", "#"),
        ("number sign", "#"),
        ("pound sign", "#"),
        ("at sign", "@"),
        ("at symbol", "@"),
        ("ampersand", "&"),
        ("and sign", "&"),
        ("open bracket", "["),
        ("close bracket", "]"),
        ("open square bracket", "["),
        ("close square bracket", "]"),
        ("left square bracket", "["),
        ("right square bracket", "]"),
        ("open brace", "{"),
        ("close brace", "}"),
        ("open curly bracket", "{"),
        ("close curly bracket", "}"),
        ("left curly bracket", "{"),
        ("right curly bracket", "}"),
        ("open parenthesis", "("),
        ("close parenthesis", ")"),
        ("open paren", "("),
        ("close paren", ")"),
        ("left parenthesis", "("),
        ("right parenthesis", ")"),
        ("open angle bracket", "<"),
        ("close angle bracket", ">"),
        ("less than", "<"),
        ("greater than", ">"),
        ("question mark", "?"),
        ("exclamation mark", "!"),
        ("exclamation point", "!"),
        ("semi colon", ";"),
        ("semi-colon", ";"),
        ("semicolon", ";"),
        ("full stop", "."),
        ("period", "."),
        ("dot", "."),
        ("comma", ","),
        ("colon", ":"),
        ("apostrophe", "'"),
        ("single quote", "'"),
        ("double quote", '"'),
        ("quote", '"'),
        ("back tick", "`"),
        ("backtick", "`"),
        ("hyphen", "-"),
        ("dash", "-"),
        ("minus sign", "-"),
        ("underscore", "_"),
        ("slash", "/"),
        ("forward slash", "/"),
        ("back slash", "\\"),
        ("backslash", "\\"),
        ("pipe", "|"),
        ("vertical bar", "|"),
        ("tilde", "~"),
        ("caret", "^"),
        ("percent", "%"),
        ("percent sign", "%"),
        ("dollar sign", "$"),
        ("dollar", "$"),
        ("euro", "€"),
        ("pound sterling", "£"),
        ("asterisk", "*"),
        ("star", "*"),
        ("plus", "+"),
        ("plus sign", "+"),
        ("equals", "="),
        ("equal sign", "="),
        ("at", "@"),
    ],
    key=lambda x: -len(x[0]),
)


def _apply_spelling_modifier(text: str) -> str:
    """Strip punct/case from spoken words, map spoken punctuation phrases, then sentence casing."""
    if not text:
        return text

    raw = re.sub(r"\s+", " ", text.strip())
    scrubbed = re.sub(r"[^\w\s]", "", raw.lower(), flags=re.UNICODE)
    words = scrubbed.split()
    if not words:
        return ""

    tokens: list[str] = []
    i = 0
    while i < len(words):
        rest = " ".join(words[i:])
        matched = False
        for phrase, sym in _SPELLING_PUNCT_PHRASES:
            plen = len(phrase.split())
            if rest == phrase or rest.startswith(phrase + " "):
                tokens.append(sym)
                i += plen
                matched = True
                break
        if not matched:
            tokens.append(words[i])
            i += 1

    s = _join_spelling_tokens(tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return _apply_sentence_casing_after_punct(s)


def _join_spelling_tokens(parts: list[str]) -> str:
    """Join word tokens and punctuation: glue symbols; no space after a just-inserted punctuation symbol."""
    if not parts:
        return ""
    result = ""
    last_was_punct = False
    for p in parts:
        one_sym = len(p) == 1 and not p.isalnum()
        if not result:
            result = p
            last_was_punct = one_sym
            continue
        if one_sym:
            result = result.rstrip() + p
            last_was_punct = True
        else:
            if last_was_punct:
                result = result + p
            else:
                result = result + " " + p
            last_was_punct = False
    return result


def _apply_sentence_casing_after_punct(text: str) -> str:
    """Capitalize first alpha after . ? ! and start of string."""
    if not text:
        return text
    chars = list(text)
    n = len(chars)
    cap_next = True
    for idx in range(n):
        c = chars[idx]
        if c.isalpha():
            if cap_next:
                chars[idx] = c.upper()
                cap_next = False
        elif c in ".!?":
            cap_next = True
        elif c.isspace():
            pass
    return "".join(chars)


def apply_modifier_transform(text: str, modifier_id: DictationModifierId) -> str:
    """Apply the active modifier transform after the base number pass."""
    if modifier_id == "upper":
        return _title_each_word(text)
    if modifier_id == "capitals":
        return text.upper()
    if modifier_id == "camel":
        return _to_camel_case(text)
    if modifier_id == "snake":
        return _to_snake_case(text)
    if modifier_id == "spelling":
        return _apply_spelling_modifier(text)
    return text


def apply_dictation_postprocess(text: str, modifier_id: Optional[DictationModifierId]) -> str:
    """Run :func:`apply_base_postprocess` then :func:`apply_modifier_transform` when a modifier is active."""
    if not text:
        return text
    result = apply_base_postprocess(text)
    if modifier_id is None:
        return result
    return apply_modifier_transform(result, modifier_id)


def apply_dictation_postprocess_partial(text: str, modifier_id: Optional[DictationModifierId]) -> str:
    """Like :func:`apply_dictation_postprocess`, but leaves the **spelling** modifier unapplied (stable partials)."""
    if not text:
        return text
    result = apply_base_postprocess(text)
    if modifier_id is None or modifier_id == "spelling":
        return result
    return apply_modifier_transform(result, modifier_id)
