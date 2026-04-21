from __future__ import annotations

import re

from vocalance.app.events.dictation_events import DictationModifierId

MODIFIER_DISPLAY: dict[str, str] = {
    "upper": "Upper",
    "capitals": "Capitals",
    "camel": "Camel",
    "snake": "Snake",
    "spelling": "Spelling",
    "kebab": "Kebab",
    "diminish": "Diminish",
    "strip": "Strip",
}


def modifier_display_label(modifier_id: DictationModifierId) -> str:
    return MODIFIER_DISPLAY.get(modifier_id, modifier_id)


def title_each_word(text: str) -> str:
    def cap_word(w: str) -> str:
        if not w:
            return w
        return w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper()

    return " ".join(cap_word(p) for p in text.split())


def words_for_camel_snake(text: str) -> list[str]:
    s = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return [w for w in s.split() if w]


def to_camel_case(text: str) -> str:
    words = words_for_camel_snake(text)
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


def to_snake_case(text: str) -> str:
    words = words_for_camel_snake(text)
    return "_".join(w.lower() for w in words if w)


def to_kebab_case(text: str) -> str:
    words = words_for_camel_snake(text)
    return "-".join(w.lower() for w in words if w)


CONTRACTIONS: set[str] = {
    "i'm",
    "i'll",
    "i've",
    "i'd",
    "you're",
    "you'll",
    "you've",
    "you'd",
    "he's",
    "he'll",
    "he'd",
    "she's",
    "she'll",
    "she'd",
    "it's",
    "it'll",
    "it'd",
    "we're",
    "we'll",
    "we've",
    "we'd",
    "they're",
    "they'll",
    "they've",
    "they'd",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
    "doesn't",
    "don't",
    "didn't",
    "won't",
    "wouldn't",
    "shan't",
    "shouldn't",
    "can't",
    "couldn't",
    "mustn't",
    "mightn't",
    "needn't",
    "daren't",
    "oughtn't",
    "let's",
    "that's",
    "who's",
    "what's",
    "where's",
    "when's",
    "why's",
    "how's",
    "y'all",
    "ma'am",
    "o'clock",
}


def apply_strip_modifier(text: str, retain_grammatical_correctness: bool = False) -> str:
    if not text:
        return text

    if retain_grammatical_correctness:
        protected = text

        def protect_contraction(m: re.Match) -> str:
            word = m.group(0)
            if word.lower() in CONTRACTIONS:
                return word.replace("'", "___apostrophe___")
            return word

        protected = re.sub(r"\b[a-zA-Z]+'[a-zA-Z]+\b", protect_contraction, protected)
        s = re.sub(r"[^\w\s]", "", protected, flags=re.UNICODE)
        s = s.replace("___apostrophe___", "'")
    else:
        s = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    return re.sub(r"\s+", " ", s).strip()


def apply_diminish_modifier(text: str, retain_grammatical_correctness: bool = False) -> str:
    if not text:
        return text
    lowered = text.lower()
    if retain_grammatical_correctness:
        lowered = re.sub(r"\bi(?:'m|'ll|'ve|'d)?\b", lambda m: m.group(0).capitalize(), lowered)
    return lowered


SPELLING_PUNCT_PHRASES: list[tuple[str, str]] = sorted(
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


def apply_spelling_modifier(text: str, accumulated_text: str = "") -> str:
    if not text:
        return text

    raw = re.sub(r"\s+", " ", text.strip())

    # Protect contractions
    protected = raw

    def protect_contraction(m: re.Match) -> str:
        word = m.group(0)
        if word.lower() in CONTRACTIONS:
            return word.replace("'", "___apostrophe___")
        return word

    protected = re.sub(r"\b[a-zA-Z]+'[a-zA-Z]+\b", protect_contraction, protected)
    scrubbed = re.sub(r"[^\w\s]", "", protected.lower(), flags=re.UNICODE)
    scrubbed = scrubbed.replace("___apostrophe___", "'")

    words = scrubbed.split()
    if not words:
        return ""

    tokens: list[str] = []
    i = 0
    while i < len(words):
        rest = " ".join(words[i:])
        matched = False
        for phrase, sym in SPELLING_PUNCT_PHRASES:
            plen = len(phrase.split())
            if rest == phrase or rest.startswith(phrase + " "):
                tokens.append(sym)
                i += plen
                matched = True
                break
        if not matched:
            tokens.append(words[i])
            i += 1

    s = join_spelling_tokens(tokens)
    s = re.sub(r"\s+", " ", s).strip()

    prepend_space = False
    if accumulated_text and not accumulated_text.endswith(" ") and tokens:
        last_char = accumulated_text[-1]
        first_token = tokens[0]

        prev_is_sym = not last_char.isalnum() and last_char not in "'"
        is_sym = len(first_token) == 1 and not first_token.isalnum() and first_token not in CONTRACTIONS

        if is_sym:
            if first_token in "([{<#@$€£":
                if prev_is_sym and last_char in "([{<":
                    prepend_space = False
                else:
                    prepend_space = True
            else:
                prepend_space = False
        else:
            if prev_is_sym:
                if last_char in "([{<#@$€£":
                    prepend_space = False
                else:
                    prepend_space = True
            else:
                prepend_space = True

    result = apply_sentence_casing_after_punct(s, accumulated_text)
    if prepend_space:
        result = " " + result
    return result


def join_spelling_tokens(parts: list[str]) -> str:
    if not parts:
        return ""
    result = ""
    for i, p in enumerate(parts):
        is_sym = len(p) == 1 and not p.isalnum() and p not in CONTRACTIONS

        if not result:
            result = p
            continue

        prev_p = parts[i - 1]
        prev_is_sym = len(prev_p) == 1 and not prev_p.isalnum() and prev_p not in CONTRACTIONS

        if is_sym:
            if p in "([{<#@$€£":
                if prev_is_sym and prev_p in "([{<":
                    result += p
                else:
                    result += " " + p
            else:
                result += p
        else:
            if prev_is_sym:
                if prev_p in "([{<#@$€£":
                    result += p
                else:
                    result += " " + p
            else:
                result += " " + p
    return result


def apply_sentence_casing_after_punct(text: str, accumulated_text: str = "") -> str:
    if not text:
        return text

    cap_next = True
    if accumulated_text:
        last_char = ""
        for c in reversed(accumulated_text):
            if not c.isspace():
                last_char = c
                break
        if last_char:
            if last_char in ".!?":
                cap_next = True
            else:
                cap_next = False

    chars = list(text)
    n = len(chars)
    for idx in range(n):
        c = chars[idx]
        if c.isalpha():
            if cap_next:
                chars[idx] = c.upper()
                cap_next = False
            else:
                chars[idx] = c.lower()
        elif c in ".!?":
            cap_next = True

    joined = "".join(chars)
    joined = re.sub(r"\bi(?:'m|'ll|'ve|'d)?\b", lambda m: m.group(0).capitalize(), joined, flags=re.IGNORECASE)
    return joined


def apply_modifier_transform(
    text: str,
    active_modifiers: set[DictationModifierId],
    explicit_modifiers: set[DictationModifierId] | None = None,
    accumulated_text: str = "",
) -> str:
    if not active_modifiers:
        return text

    if explicit_modifiers is None:
        explicit_modifiers = set()

    if "spelling" in active_modifiers:
        text = apply_spelling_modifier(text, accumulated_text)
    if "strip" in active_modifiers:
        retain_grammatical_correctness = "strip" not in explicit_modifiers
        text = apply_strip_modifier(text, retain_grammatical_correctness)

    if "diminish" in active_modifiers:
        retain_grammatical_correctness = "diminish" not in explicit_modifiers
        text = apply_diminish_modifier(text, retain_grammatical_correctness)
    if "upper" in active_modifiers:
        text = title_each_word(text)
    if "capitals" in active_modifiers:
        text = text.upper()
    if "camel" in active_modifiers:
        text = to_camel_case(text)
    if "snake" in active_modifiers:
        text = to_snake_case(text)
    if "kebab" in active_modifiers:
        text = to_kebab_case(text)

    return text
