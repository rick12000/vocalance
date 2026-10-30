from vocalance.app.services.dictation_flow.llm.dictation_llm_output import (
    dictation_stop_sequences_for_messages,
    normalize_forbidden_output_punctuation,
    sanitize_llm_formatter_output,
    strip_dictation_meta_tail,
)


def test_strip_dictation_meta_tail_removes_trailing_summary():
    body = "Dear team,\n\nPlease review the attached draft."
    raw = f"{body}\n\nSummary of changes:\n- fixed greeting"
    assert strip_dictation_meta_tail(raw) == body


def test_strip_dictation_meta_tail_preserves_short_text():
    assert strip_dictation_meta_tail("Hi there") == "Hi there"


def test_sanitize_strips_reformatted_preamble_and_wrapper_echo():
    raw = (
        "Okay, a little bit better.\n\n"
        "Here is the reformatted DICTATION TO REFORMAT block, with grammar improved:\n\n"
        "--- DICTATION TO REFORMAT ---\n"
        "The actual content we want.\n"
        "--- END DICTATION ---"
    )
    assert sanitize_llm_formatter_output(raw) == "The actual content we want."


def test_sanitize_keeps_reply_after_echoed_transcript_markers():
    raw = "=== TRANSCRIPT ===\n" "this is a quick test\n" "=== END TRANSCRIPT ===\n" "This is a quick test."
    assert sanitize_llm_formatter_output(raw) == "This is a quick test."


def test_dictation_stop_sequences_for_formatter_messages():
    messages = [
        {"role": "system", "content": "You reformat dictated speech."},
        {"role": "user", "content": "rules\n\n=== TRANSCRIPT ===\nplain dictation\n=== END TRANSCRIPT ==="},
    ]
    assert dictation_stop_sequences_for_messages(messages)


def test_sanitize_strips_consultant_style_preamble():
    raw = (
        "Sure. Here's a concise, clear revision that preserves your original meaning:\n\n"
        "In the default formatter, explicitly state bullet points.\n"
    )
    assert sanitize_llm_formatter_output(raw).startswith("In the default formatter")


def test_normalize_semicolon_to_period_and_capitalize():
    assert normalize_forbidden_output_punctuation("first; second") == "first. Second"


def test_normalize_mid_sentence_colon_to_comma():
    assert normalize_forbidden_output_punctuation("costs: they rose") == "costs, they rose"


def test_normalize_preserves_time_colon():
    assert "12:30" in normalize_forbidden_output_punctuation("Meet at 12:30 today.")


def test_amend_messages_get_stop_sequences():
    messages = [
        {"role": "system", "content": "amend"},
        {"role": "user", "content": "--- USER PROMPT (spoken instructions) ---\nx"},
    ]
    assert dictation_stop_sequences_for_messages(messages)
