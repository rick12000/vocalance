import pytest

from vocalance.app.services.dictation_flow.speech_recognition.dictation_text_normalize import normalize_dictation_text


@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        ("", ""),
        ("   ", ""),
        ("hello world", "hello world"),
        ("hello  world", "hello world"),
        ("  hello   world  ", "hello world"),
        ("hello\t\nworld", "hello world"),
    ],
)
def test_normalize_dictation_text(input_text, expected_output):
    assert normalize_dictation_text(input_text) == expected_output
