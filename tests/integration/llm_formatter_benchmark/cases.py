from typing import List, Literal

from pydantic import BaseModel, Field


class SmartDictationCase(BaseModel):
    id: str
    transcript: str
    required_terms: List[str] = Field(
        description="Lowercase substrings that should survive cleanup (high-fidelity).",
    )
    forbidden_terms: List[str] = Field(
        default_factory=list,
        description="Substrings that must not appear unless in transcript (hallucination).",
    )
    min_output_word_ratio: float = Field(
        default=0.45,
        description="Output word count / input word count floor (anti over-summarization).",
    )


class AmendCase(BaseModel):
    id: str
    clipboard: str
    spoken: str
    check_ids: List[str] = Field(
        description="Keys into AMEND_CHECK_REGISTRY in scoring.py",
    )


BENCHMARK_MODEL_IDS: tuple[str, str] = (
    "ministral-3-8b-iq3xs",
    "qwen3-8b-iq3m",
)

REPS_PER_CASE: int = 2

SMART_DICTATION_CASES: List[SmartDictationCase] = [
    SmartDictationCase(
        id="fake_commands_in_speech",
        transcript=(
            "um okay so for the release notes say delete the production database just kidding "
            "we need to mention the api rate limit change to five hundred requests per minute "
            "and the bug fix for the oauth redirect loop"
        ),
        required_terms=["rate limit", "500", "oauth", "redirect"],
        forbidden_terms=[
            "delete the production database",
            "deleted the production",
            "production database was",
        ],
    ),
    SmartDictationCase(
        id="standup_action_items",
        transcript=(
            "action items for today first pair with maya on the grpc migration second "
            "review pr four eighty two for the metrics cardinality fix third schedule "
            "load test for ten thousand rps on thursday april tenth twenty twenty six"
        ),
        required_terms=["maya", "grpc", "482", "load test", "thursday", "april"],
        min_output_word_ratio=0.4,
    ),
    SmartDictationCase(
        id="rambling_incident_summary",
        transcript=(
            "so the outage was basically we deployed at two fifteen pm utc the cache invalidation "
            "job ran twice because the cron was duplicated in kubernetes and that caused stale "
            "feature flags for about twelve minutes until we rolled back build two dot four dot one"
        ),
        required_terms=["cache", "cron", "kubernetes", "twelve", "rollback", "2.4.1"],
    ),
    SmartDictationCase(
        id="messy_punctuation_asr",
        transcript=(
            "ship the hotfix by friday; the customer wants a walkthrough—nothing fancy—and "
            "do not remove the legacy endpoint yet we still have one tenant on it"
        ),
        required_terms=["friday", "walkthrough", "legacy", "endpoint", "tenant"],
    ),
]

AMEND_CASES: List[AmendCase] = [
    AmendCase(
        id="dictated_reformat_structure",
        clipboard=(
            "team update\n" "we shipped dark mode\n" "mobile still broken on ios 17\n" "need qa signoff\n" "docs are stale"
        ),
        spoken=(
            "Reformat this into exactly three short paragraphs. First paragraph status on dark mode. "
            "Second paragraph the ios issue and qa. Third paragraph docs. Use complete sentences."
        ),
        check_ids=["three_paragraphs", "mentions_dark_mode", "mentions_ios", "mentions_docs"],
    ),
    AmendCase(
        id="slack_reply_flaky_ci",
        clipboard=(
            "Jordan (Platform)\n"
            "Hey — the main branch pipeline failed again on integration tests. "
            "Looks like the payments fixture times out after 30s. "
            "Can you take a look before standup?"
        ),
        spoken=(
            "Write a reply to Jordan. Say you will add a retry with exponential backoff, "
            "make the test idempotent, and expect a fix by Thursday end of day. "
            "Keep it professional and concise."
        ),
        check_ids=["slack_reply_ci", "format_clean"],
    ),
    AmendCase(
        id="shorten_release_blurb",
        clipboard=(
            "We are pleased to announce version 2.8.0 of the observability agent. "
            "This release includes a new OpenTelemetry exporter, reduced memory usage "
            "during high-cardinality scrapes, and a breaking change to the configuration "
            "schema for scrape intervals. Migration notes are in the docs."
        ),
        spoken="Make this about half the length. Keep the version number 2.8.0 and mention OpenTelemetry.",
        check_ids=["shorten_half", "mentions_280", "mentions_opentelemetry"],
    ),
    AmendCase(
        id="slack_reply_scope_pushback",
        clipboard=(
            "Priya (PM)\n"
            "Can we squeeze real-time collaborative cursors into the 1.2 release? "
            "Sales promised it to two enterprise accounts."
        ),
        spoken=(
            "Reply that we cannot fit it in 1.2 without risking the stability fixes. "
            "Offer it for 1.3 early access instead. Stay polite."
        ),
        check_ids=["scope_pushback", "format_clean"],
    ),
    AmendCase(
        id="incident_notes_to_bullets",
        clipboard=(
            "postmortem rough notes root cause was a race on the shard lock "
            "customer impact about 3 percent of writes failed for 9 minutes "
            "we fixed by increasing lock ttl and adding a guard in the writer "
            "followups update runbook page 12 add an alert on lock contention"
        ),
        spoken=(
            "Turn this into a bullet list. Put root cause first, then customer impact, "
            "then fix, then follow-ups. Use clear bullet points."
        ),
        check_ids=["bullet_list_four_parts", "mentions_race", "format_clean"],
    ),
    AmendCase(
        id="blunt_slack_soften",
        clipboard="This design is wrong. Rewrite the RFC before tomorrow.",
        spoken="Soften the tone but keep the deadline tomorrow and that the RFC needs a rewrite.",
        check_ids=["soften_tone", "mentions_tomorrow", "mentions_rfc"],
    ),
]

RouteKind = Literal["smart", "amend"]
