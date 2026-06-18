from __future__ import annotations

import re
from typing import Optional

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


def validate_custom_hotkey(value: str) -> Optional[str]:
    """Validate a user-entered custom hotkey value.

    A valid custom hotkey is a single chord: one or more allowlisted key names
    joined by ``+``. Commas, semicolons, and any other separators that could
    chain multiple commands are rejected outright.

    Returns an error message string when validation fails, otherwise None.
    """
    if not value or not value.strip():
        return "Hotkey must not be empty"
    if any(c in value for c in (",", ";")):
        return "Hotkey must be a single chord — commas and semicolons are not allowed"
    tokens = [t for t in value.replace(" ", "+").split("+") if t != ""]
    if not tokens:
        return "Hotkey must contain at least one key"
    for token in tokens:
        if not is_allowed_key(token):
            return f"Unrecognised key '{token.strip()}' — use letters, numbers, modifier keys (ctrl, alt, shift, win), named keys, or function keys joined with +"
    return None
