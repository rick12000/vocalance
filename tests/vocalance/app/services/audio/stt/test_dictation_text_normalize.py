"""Tests for shared dictation text normalization."""

import pytest

from vocalance.app.services.audio.stt.dictation_text_normalize import normalize_dictation_text


@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        ("", ""),
        ("   ", ""),
        ("hello world", "hello world"),
        ("um hello world", "hello world"),
        ("hello world uh", "hello world"),
        ("hello  world", "hello world"),
        ("the the cat", "the cat"),
    ],
)
def test_normalize_dictation_text(input_text, expected_output):
    assert normalize_dictation_text(input_text) == expected_output
