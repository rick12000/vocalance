import pytest

from vocalance.app.services.dictation_flow.speech_recognition.transcript_state_manager import TranscriptStateManager


def test_no_committed_delta_on_first_update(tsm):
    delta, provisional = tsm.update("hello world")
    assert delta == ""
    assert provisional == "hello world"


def test_provisional_tail_excluded_from_committed(tsm):
    tsm.update("one two three four")
    delta, provisional = tsm.update("one two three four")
    assert "three" in provisional or "four" in provisional
    assert delta == "" or "three" not in delta


def test_commits_stable_prefix_after_window(tsm):
    tsm.update("hello world today")
    delta, provisional = tsm.update("hello world today")
    assert delta == "hello"
    assert "world" in provisional or "today" in provisional


def test_committed_count_never_shrinks():
    tsm = TranscriptStateManager(stability_window=2, provisional_words=1)
    tsm.update("apple banana cherry")
    tsm.update("apple banana cherry")
    assert tsm.committed_word_count == 2
    tsm.update("completely different hypothesis")
    assert tsm.committed_word_count >= 2


def test_finalize_commits_everything(tsm):
    tsm.update("the quick brown fox")
    tsm.update("the quick brown fox")
    delta = tsm.finalize("the quick brown fox jumps")
    full = tsm.total_committed_text
    assert "jumps" in full
    assert full == "the quick brown fox jumps"


def test_delta_is_additive_not_repeated():
    tsm = TranscriptStateManager(stability_window=2, provisional_words=1)
    tsm.update("alpha beta gamma")
    first_delta, _ = tsm.update("alpha beta gamma")
    second_delta, _ = tsm.update("alpha beta gamma delta")
    third_delta, _ = tsm.update("alpha beta gamma delta")
    all_deltas = " ".join(d for d in [first_delta, second_delta, third_delta] if d)
    words = all_deltas.split()
    assert len(words) == len(set(words))


def test_reset_clears_state(tsm):
    tsm.update("some text here now")
    tsm.update("some text here now")
    tsm.reset()
    assert tsm.committed_word_count == 0
    assert tsm.total_committed_text == ""
    assert tsm.last_sent_committed_len == 0
    delta, provisional = tsm.update("fresh start")
    assert delta == ""
    assert provisional == "fresh start"


def test_empty_hypothesis_produces_no_output(tsm):
    delta, provisional = tsm.update("")
    assert delta == ""
    assert provisional == ""


def test_stability_window_respected():
    tsm = TranscriptStateManager(stability_window=3, provisional_words=1)
    tsm.update("one two three")
    tsm.update("one two three")
    delta, provisional = tsm.update("one two three")
    assert "one" in delta
    assert "three" in provisional


@pytest.mark.parametrize("hypothesis", ["hello", "hello world", "hello world foo bar baz"])
def test_update_returns_tuple(tsm, hypothesis):
    result = tsm.update(hypothesis)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_finalize_after_no_updates():
    tsm = TranscriptStateManager(stability_window=2, provisional_words=2)
    delta = tsm.finalize("brand new text")
    assert delta == "brand new text"
