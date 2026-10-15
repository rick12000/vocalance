import re
from typing import List, Sequence

DICTATION_META_STOP_SEQUENCES: tuple[str, ...] = (
    "\n\nHere is the",
    "\n\nHere is ",
    "\nHere is the reformatted",
    "\n\nSummary of changes",
    "\n\nHere are the changes",
    "\n\nHere is a summary",
    "\n\nI've made the following",
    "\n\nI have made the following",
    "\n\nChanges made",
    "\n\nChange log",
    "\n\nChangelog",
    "\n\n**Summary",
    "\n\n**Changes",
    "\n\nNote",
    "\n\nNotes",
    "\n\n---",
)

_META_TAIL_RE = re.compile(
    r"\n\n(?:"
    r"here(?:'s| is) (?:the )?(?:revised|updated|corrected|formatted|cleaned|reformatted)"
    r"|i(?:'ve| have) (?:made|updated|revised|corrected|changed)"
    r"|summary of (?:changes|edits)"
    r"|changes (?:i )?made"
    r"|change\s*log"
    r"|the following changes"
    r"|notes?:\s"
    r")",
    re.IGNORECASE,
)

_META_PREFIX_LINE = re.compile(
    r"^(?:"
    r"(?:okay|sure|alright|yes)[,.\s]*"
    r"|here(?:'s| is)\b.*"
    r"|.*\b(?:reformatted|revised|corrected)\b.*\b(?:dictation|block|text|revision)\b.*"
    r"|.*\bgrammar[,]?\s+punctuation\b.*"
    r"|.*\bpreserving (?:all )?original meaning\b.*"
    r"|.*\bconcise, clear revision\b.*"
    r"|.*\bthis version is\b.*"
    r"|.*\bwith brief, precise additions\b.*"
    r"|---\s*(?:DICTATION|END)\b.*"
    r"|===\s*(?:TRANSCRIPT|END TRANSCRIPT)\s*==="
    r")$",
    re.IGNORECASE,
)

def _capitalize_first_letter(fragment: str) -> str:
    for index, character in enumerate(fragment):
        if character.isalpha():
            return fragment[:index] + character.upper() + fragment[index + 1 :]
    return fragment


def strip_dictation_wrapper_echo(text: str) -> str:
    if "--- DICTATION TO REFORMAT ---" in text:
        _, text = text.split("--- DICTATION TO REFORMAT ---", 1)
    if "--- END DICTATION ---" in text:
        text = text.split("--- END DICTATION ---", 1)[0]
    match = re.search(r"=== TRANSCRIPT ===\s*(.*?)\s*=== END TRANSCRIPT ===", text, re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def normalize_forbidden_output_punctuation(text: str) -> str:
    text = text.replace("\u2014", ", ").replace("\u2013", ", ").replace("—", ", ").replace("–", ", ")

    while ";" in text:
        index = text.index(";")
        prefix = text[:index]
        tail = text[index + 1 :]
        if not tail:
            text = prefix + "."
            break
        leading = len(tail) - len(tail.lstrip())
        ws = tail[:leading]
        rest = tail[leading:]
        text = prefix + "." + ws + _capitalize_first_letter(rest)

    lines: List[str] = []
    for line in text.splitlines():
        if ":" not in line:
            lines.append(line)
            continue
        parts: List[str] = []
        cursor = 0
        while cursor < len(line):
            colon_at = line.find(":", cursor)
            if colon_at < 0:
                parts.append(line[cursor:])
                break
            time_colon = (
                colon_at > 0
                and line[colon_at - 1].isdigit()
                and colon_at + 1 < len(line)
                and line[colon_at + 1].isdigit()
            )
            if time_colon:
                parts.append(line[cursor : colon_at + 1])
                cursor = colon_at + 1
                continue
            after = line[colon_at + 1 :]
            if after.strip():
                parts.append(line[cursor:colon_at])
                parts.append(",")
                cursor = colon_at + 1
                continue
            parts.append(line[cursor : colon_at + 1])
            cursor = colon_at + 1
        lines.append("".join(parts))
    return "\n".join(lines)


def strip_dictation_meta_prefix(text: str) -> str:
    text = strip_dictation_wrapper_echo(text)
    lines = text.splitlines()
    while lines:
        candidate = lines[0].strip()
        if not candidate:
            lines.pop(0)
            continue
        if _META_PREFIX_LINE.match(candidate):
            lines.pop(0)
            continue
        if candidate.endswith(":") and len(lines) > 1:
            joined = f"{candidate} {lines[1].strip()}".lower()
            if "reformatted" in joined or "revised" in joined or "dictation" in joined:
                lines.pop(0)
                continue
        break
    return "\n".join(lines).strip()


def strip_dictation_meta_tail(text: str, min_body_chars: int = 40) -> str:
    match = _META_TAIL_RE.search(text)
    if match is None or match.start() < min_body_chars:
        return text
    return text[: match.start()].rstrip()


def sanitize_llm_formatter_output(text: str) -> str:
    text = strip_dictation_meta_prefix(text)
    text = strip_dictation_meta_tail(text)
    text = strip_dictation_wrapper_echo(text)
    text = normalize_forbidden_output_punctuation(text)
    return text.strip()


def dictation_stop_sequences_for_messages(messages: Sequence[dict]) -> List[str]:
    from vocalance.app.services.dictation_flow.llm.dictation_messages import uses_shared_formatter_output_pipeline

    if uses_shared_formatter_output_pipeline(list(messages)):
        return list(DICTATION_META_STOP_SEQUENCES)
    return []
