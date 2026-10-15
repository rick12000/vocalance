from typing import Dict, List

from vocalance.app.services.dictation_flow.llm.agentic_prompt_service import _DEFAULT_PROMPT
from vocalance.app.services.dictation_flow.llm.formatter_output_rules import OUTPUT_SLOP_AND_FORMATTING_RULE

_DICTATION_SYSTEM_STUB = (
    "You reformat dictated speech. Follow the formatter rules in the user message. "
    "Your reply must be only the reformatted transcript with no other words."
)

_TRANSCRIPT_BEGIN = "=== TRANSCRIPT ==="
_TRANSCRIPT_END = "=== END TRANSCRIPT ==="


def _system_with_output_style_rule(base: str) -> str:
    return f"{base.strip()}\n\n{OUTPUT_SLOP_AND_FORMATTING_RULE}"


def build_dictation_chat_messages(formatter_rules: str, raw_transcript: str) -> List[Dict[str, str]]:
    rules = (formatter_rules or _DEFAULT_PROMPT).strip()
    user_content = (
        f"{rules}\n\n"
        f"{_TRANSCRIPT_BEGIN}\n"
        f"{raw_transcript.strip()}\n"
        f"{_TRANSCRIPT_END}"
    )
    return [
        {"role": "system", "content": _system_with_output_style_rule(_DICTATION_SYSTEM_STUB)},
        {"role": "user", "content": user_content},
    ]


def is_dictation_formatter_messages(messages: List[Dict[str, str]]) -> bool:
    if len(messages) < 2:
        return False
    user = messages[1].get("content") or ""
    if _TRANSCRIPT_BEGIN in user:
        return True
    if "--- TEXT TO TRANSFORM" in user or "--- USER PROMPT" in user:
        return False
    return False


def is_amend_messages(messages: List[Dict[str, str]]) -> bool:
    if len(messages) < 2:
        return False
    return "--- USER PROMPT" in (messages[1].get("content") or "")


def uses_shared_formatter_output_pipeline(messages: List[Dict[str, str]]) -> bool:
    return is_dictation_formatter_messages(messages) or is_amend_messages(messages)
