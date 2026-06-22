from __future__ import annotations

import re

DANGEROUS_CHAR_RE = re.compile(r"[\x00-\x1f\x7f\u0085\u2028\u2029]")


def is_valid_alias_text(value: str) -> bool:
    """Return True if ``value`` contains no characters that could trigger execution when pasted.

    Blocks all ASCII control characters (including newline, carriage return, ESC,
    Ctrl+C/D/Z) plus Unicode newline-equivalents that terminals may interpret as
    command submission.
    """
    return not DANGEROUS_CHAR_RE.search(value)
