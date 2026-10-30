import pytest

from vocalance.app.services.dictation_flow.llm.reasoning_stream import ReasoningTokenFilter, strip_reasoning_from_text


def test_reasoning_filter_emits_text_that_has_no_think_span():
    visible = ReasoningTokenFilter().push("Clean dictated sentence.")

    assert visible == "Clean dictated sentence."


def test_reasoning_filter_drops_a_complete_think_span():
    visible = ReasoningTokenFilter().push("<think>hidden steps</think>Final answer")

    assert visible == "Final answer"


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["<thi", "nk>hidden</thi", "nk>Visible"], "Visible"),
        (["before <think>hidden", "</think> after"], "before after"),
    ],
)
def test_reasoning_filter_handles_tags_split_across_tokens(chunks, expected):
    stream = ReasoningTokenFilter()
    visible = "".join(stream.push(chunk) for chunk in chunks) + stream.finish()

    assert visible == expected


def test_reasoning_filter_waits_for_close_when_prompt_already_opened_thinking():
    stream = ReasoningTokenFilter(start_inside=True)
    hidden = stream.push("still reasoning")
    answer = stream.push("</think>\n\nThe pasted sentence.")

    assert hidden == ""
    assert answer == "The pasted sentence."


def test_reasoning_filter_discards_reasoning_that_never_closes():
    stream = ReasoningTokenFilter(start_inside=True)
    stream.push("unfinished thought")

    assert stream.finish() == ""


def test_strip_reasoning_from_text_on_complete_string():
    raw = "<think>plan</think>Only this."

    assert strip_reasoning_from_text(raw) == "Only this."


def test_reasoning_filter_flushes_a_partial_tag_that_is_real_text():
    stream = ReasoningTokenFilter()
    held = stream.push("ends with <")

    assert held == "ends with "
    assert stream.finish() == "<"
