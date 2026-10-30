import re
from typing import Callable, Dict, List

from tests.integration.llm_formatter_benchmark.cases import AmendCase, SmartDictationCase

_META_PHRASES = (
    "here is the",
    "here's the",
    "summary of changes",
    "i've made",
    "i have made",
    "reformatted dictation",
    "change log",
)
_FORBIDDEN_PUNCT = re.compile(r"[;—–]")


def format_clean(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return False
    if _FORBIDDEN_PUNCT.search(text):
        return False
    if lower.startswith("sure") or lower.startswith("okay,"):
        return False
    return not any(phrase in lower for phrase in _META_PHRASES)


def smart_key_terms(case: SmartDictationCase, text: str) -> bool:
    lower = text.lower()
    if not case.required_terms:
        return True
    hits = sum(1 for term in case.required_terms if term.lower() in lower)
    return hits >= max(1, int(0.6 * len(case.required_terms) + 0.4))


def smart_no_forbidden(case: SmartDictationCase, text: str) -> bool:
    lower = text.lower()
    return all(term.lower() not in lower for term in case.forbidden_terms)


def smart_word_ratio(case: SmartDictationCase, text: str, input_text: str) -> bool:
    in_words = len(input_text.split())
    out_words = len(text.split())
    if in_words == 0:
        return bool(text.strip())
    return out_words / in_words >= case.min_output_word_ratio


def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _bullet_lines(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^([-*•]|\d+[.)])\s+\S", stripped):
            lines.append(stripped)
    return lines


def check_three_paragraphs(clipboard: str, spoken: str, text: str) -> bool:
    return len(_paragraphs(text)) >= 3


def check_mentions_dark_mode(clipboard: str, spoken: str, text: str) -> bool:
    return "dark mode" in text.lower()


def check_mentions_ios(clipboard: str, spoken: str, text: str) -> bool:
    lower = text.lower()
    return "ios" in lower or "17" in lower


def check_mentions_docs(clipboard: str, spoken: str, text: str) -> bool:
    return "doc" in text.lower()


def check_slack_reply_ci(clipboard: str, spoken: str, text: str) -> bool:
    lower = text.lower()
    has_retry = "retry" in lower or "backoff" in lower
    has_idempotent = "idempotent" in lower
    has_thursday = "thursday" in lower
    return has_retry and has_idempotent and has_thursday


def check_shorten_half(clipboard: str, spoken: str, text: str) -> bool:
    return len(text) <= int(len(clipboard) * 0.65)


def check_mentions_280(clipboard: str, spoken: str, text: str) -> bool:
    return "2.8.0" in text or "2.8" in text


def check_mentions_opentelemetry(clipboard: str, spoken: str, text: str) -> bool:
    lower = text.lower()
    return "opentelemetry" in lower or "otel" in lower


def check_scope_pushback(clipboard: str, spoken: str, text: str) -> bool:
    lower = text.lower()
    refuses = any(w in lower for w in ("cannot", "can't", "not fit", "won't", "unable", "1.2"))
    offers_13 = "1.3" in lower
    return refuses and offers_13


def check_bullet_list_four_parts(clipboard: str, spoken: str, text: str) -> bool:
    bullets = _bullet_lines(text)
    return len(bullets) >= 4


def check_mentions_race(clipboard: str, spoken: str, text: str) -> bool:
    return "race" in text.lower()


def check_soften_tone(clipboard: str, spoken: str, text: str) -> bool:
    lower = text.lower()
    if "wrong" in lower and "design is wrong" in lower:
        return False
    polite = any(w in lower for w in ("please", "would", "could", "suggest", "recommend", "thanks"))
    return polite


def check_mentions_tomorrow(clipboard: str, spoken: str, text: str) -> bool:
    return "tomorrow" in text.lower()


def check_mentions_rfc(clipboard: str, spoken: str, text: str) -> bool:
    return "rfc" in text.lower()


def check_format_clean_amend(clipboard: str, spoken: str, text: str) -> bool:
    return format_clean(text)


AmendCheck = Callable[[str, str, str], bool]

AMEND_CHECK_REGISTRY: Dict[str, AmendCheck] = {
    "three_paragraphs": check_three_paragraphs,
    "mentions_dark_mode": check_mentions_dark_mode,
    "mentions_ios": check_mentions_ios,
    "mentions_docs": check_mentions_docs,
    "slack_reply_ci": check_slack_reply_ci,
    "shorten_half": check_shorten_half,
    "mentions_280": check_mentions_280,
    "mentions_opentelemetry": check_mentions_opentelemetry,
    "scope_pushback": check_scope_pushback,
    "bullet_list_four_parts": check_bullet_list_four_parts,
    "mentions_race": check_mentions_race,
    "soften_tone": check_soften_tone,
    "mentions_tomorrow": check_mentions_tomorrow,
    "mentions_rfc": check_mentions_rfc,
    "format_clean": check_format_clean_amend,
}


def score_smart(case: SmartDictationCase, transcript: str, output: str) -> Dict[str, bool]:
    return {
        "format_clean": format_clean(output),
        "key_terms": smart_key_terms(case, output),
        "no_forbidden": smart_no_forbidden(case, output),
        "word_ratio": smart_word_ratio(case, output, transcript),
    }


def score_amend(case: AmendCase, output: str) -> Dict[str, bool]:
    scores: Dict[str, bool] = {"format_clean": format_clean(output)}
    for check_id in case.check_ids:
        fn = AMEND_CHECK_REGISTRY.get(check_id)
        if fn is None:
            scores[check_id] = False
            continue
        if check_id == "format_clean":
            continue
        scores[check_id] = fn(case.clipboard, case.spoken, output)
    return scores
