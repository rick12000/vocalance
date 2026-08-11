from __future__ import annotations

import re

MODIFIER_KEYS = frozenset(
    {
        "ctrl",
        "control",
        "ctrlleft",
        "ctrlright",
        "alt",
        "altleft",
        "altright",
        "shift",
        "shiftleft",
        "shiftright",
        "win",
        "winleft",
        "winright",
        "command",
        "cmd",
        "option",
        "fn",
    }
)

NAMED_KEYS = frozenset(
    {
        "enter",
        "return",
        "tab",
        "space",
        "backspace",
        "delete",
        "del",
        "escape",
        "esc",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "pageup",
        "pagedown",
        "insert",
        "capslock",
        "numlock",
        "scrolllock",
        "printscreen",
        "pause",
        "menu",
        "apps",
    }
)

SYMBOL_KEYS = frozenset({"-", "=", "/", "\\", ".", "'", "`", "[", "]"})

FUNCTION_KEY_RE = re.compile(r"f([1-9]|1[0-9]|2[0-4])")


def is_allowed_key(token: str) -> bool:
    """Return True if ``token`` is a recognised key name (modifier, named, symbol, alnum, or F-key)."""
    candidate = token.strip().lower()
    return bool(
        candidate
        and (
            (len(candidate) == 1 and (candidate.isalnum() or candidate in SYMBOL_KEYS))
            or candidate in MODIFIER_KEYS
            or candidate in NAMED_KEYS
            or FUNCTION_KEY_RE.fullmatch(candidate)
        )
    )


def is_valid_custom_hotkey(value: str) -> bool:
    """Return True if ``value`` is a safe single-chord custom hotkey.

    A valid custom hotkey is one or more allowlisted key names joined by ``+``.
    Commas and semicolons that could chain multiple commands are rejected.
    """
    if not value or not value.strip():
        return False
    if any(c in value for c in (",", ";")):
        return False
    tokens = [t for t in value.replace(" ", "+").split("+") if t]
    return bool(tokens) and all(is_allowed_key(t) for t in tokens)
