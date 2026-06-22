from typing import Optional

import pytest

from vocalance.app.events.dictation_events import DictationModifierId
from vocalance.app.services.dictation_flow.postprocess.base_postprocess import (
    apply_base_postprocess,
    strip_trailing_period_after_numbers,
)
from vocalance.app.services.dictation_flow.postprocess.modifier_postprocess import apply_modifier_transform, modifier_display_label
from vocalance.app.services.dictation_flow.postprocess.postprocess_pipeline import (
    apply_dictation_postprocess,
    apply_dictation_postprocess_partial,
)
from vocalance.app.services.dictation_flow.postprocess.segment_text import (
    clean_dictation_text,
    get_trailing_whitespace_count,
    lowercase_first_letter,
    remove_formatting,
    should_lowercase_current_start,
    should_remove_previous_period,
)


@pytest.mark.parametrize(
    "raw,expected_substr",
    [
        ("I have twenty three apples", "23"),
        ("count is one hundred forty two", "142"),
        ("code four zero nine end", "409"),
    ],
)
def test_apply_base_postprocess_replaces_spoken_numbers(raw: str, expected_substr: str) -> None:
    assert expected_substr in apply_base_postprocess(raw)


@pytest.mark.parametrize(
    "raw",
    ["go to the store", "waiting for you"],
)
def test_apply_base_postprocess_keeps_common_homophone_words(raw: str) -> None:
    assert apply_base_postprocess(raw) == raw


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("total 42.", "total 42"),
        ("total 42 .", "total 42"),
        ("pi 3.14", "pi 3.14"),
    ],
)
def test_strip_trailing_period_after_numbers(raw: str, expected: str) -> None:
    assert strip_trailing_period_after_numbers(raw) == expected


@pytest.mark.parametrize(
    "modifier_id,expected_label",
    [
        ("upper", "Upper"),
        ("capitals", "Capitals"),
        ("camel", "Camel"),
        ("snake", "Snake"),
        ("spelling", "Spelling"),
        ("kebab", "Kebab"),
        ("diminish", "Diminish"),
        ("strip", "Strip"),
    ],
)
def test_modifier_display_label(modifier_id: DictationModifierId, expected_label: str) -> None:
    assert modifier_display_label(modifier_id) == expected_label


@pytest.mark.parametrize(
    "raw,modifier_id,expected",
    [
        ("hello world", "upper", "Hello World"),
        ("HELLO WORLD", "upper", "Hello World"),
        ("hi there", "capitals", "HI THERE"),
        ("foo bar baz", "camel", "FooBarBaz"),
        ("foo, bar! baz", "camel", "FooBarBaz"),
        ("FOO BAR", "camel", "FooBar"),
        ("Foo Bar", "snake", "foo_bar"),
        ("foo__bar", "snake", "foo_bar"),
        ("Foo Bar", "kebab", "foo-bar"),
        ("Hello, World!", "strip", "Hello World"),
        ("HELLO World", "diminish", "hello world"),
    ],
)
def test_apply_modifier_transform(raw: str, modifier_id: DictationModifierId, expected: str) -> None:
    assert apply_modifier_transform(raw, {modifier_id}) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello period", "Hello."),
        ("hello comma world", "Hello, world"),
        ("hello question mark there", "Hello? There"),
        ("", ""),
    ],
)
def test_apply_modifier_transform_spelling(raw: str, expected: str) -> None:
    assert apply_modifier_transform(raw, {"spelling"}) == expected


def test_apply_modifier_transform_no_modifiers_is_identity() -> None:
    assert apply_modifier_transform("leave me alone", set()) == "leave me alone"


@pytest.mark.parametrize(
    "raw,modifiers,expected",
    [
        ("hello world", None, "hello world"),
        ("", None, ""),
        ("one two three", {"spelling"}, "123"),
    ],
)
def test_apply_dictation_postprocess(raw: str, modifiers: Optional[set], expected: str) -> None:
    assert apply_dictation_postprocess(raw, modifiers) == expected


@pytest.mark.parametrize(
    "raw,modifier_id,expected",
    [
        ("foo bar", None, "foo bar"),
        ("foo bar", "spelling", "foo bar"),
        ("foo bar", "upper", "Foo Bar"),
        ("foo bar", "camel", "FooBar"),
    ],
)
def test_apply_dictation_postprocess_partial_drops_spelling(
    raw: str, modifier_id: Optional[DictationModifierId], expected: str
) -> None:
    modifiers = {modifier_id} if modifier_id else set()
    assert apply_dictation_postprocess_partial(raw, modifiers) == expected


@pytest.mark.parametrize(
    "raw,add_trailing_space,expected",
    [
        ("hello", True, "hello "),
        ("hello", False, "hello"),
        ("", True, ""),
        ("a...b", True, "a b "),
    ],
)
def test_clean_dictation_text(raw: str, add_trailing_space: bool, expected: str) -> None:
    assert clean_dictation_text(raw, add_trailing_space=add_trailing_space) == expected


@pytest.mark.parametrize(
    "raw,is_first,expected",
    [
        ("Hello, world!", False, "hello world"),
        ("well-known", False, "well-known"),
        ("don't", False, "don't"),
        ("HELLO World", False, "hello world"),
        ("hello world", True, "Hello world"),
        ("i am here", False, "I am here"),
        ("", False, ""),
        ("  hello world  ", False, "hello world"),
    ],
)
def test_remove_formatting(raw: str, is_first: bool, expected: str) -> None:
    assert remove_formatting(raw, is_first_word_of_session=is_first) == expected


@pytest.mark.parametrize(
    "last_text,current_text,expected",
    [
        ("This is a sentence.", "and continues", True),
        ("Sentence.   ", "and continues", True),
        ("This is a sentence.", "Another sentence", False),
        ("This is text", "and continues", False),
        ("", "and continues", False),
        ("This is a sentence.", "", False),
    ],
)
def test_should_remove_previous_period(last_text: str, current_text: str, expected: bool) -> None:
    assert should_remove_previous_period(last_text, current_text) is expected


@pytest.mark.parametrize(
    "last_text,current_text,expected",
    [
        ("No sentence boundary", "Another word", True),
        ("No boundary   ", "Another word", True),
        ("This is a sentence.", "Another word", False),
        ("No boundary", "another word", False),
        ("", "Another word", False),
        ("No boundary", "", False),
    ],
)
def test_should_lowercase_current_start(last_text: str, current_text: str, expected: bool) -> None:
    assert should_lowercase_current_start(last_text, current_text) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello   ", 3),
        ("hello\t\t", 2),
        ("hello", 0),
        ("", 0),
        ("   ", 3),
    ],
)
def test_get_trailing_whitespace_count(raw: str, expected: int) -> None:
    assert get_trailing_whitespace_count(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Hello", "hello"),
        ("hello", "hello"),
        ("H", "h"),
        ("", ""),
        ("1Hello", "1Hello"),
    ],
)
def test_lowercase_first_letter(raw: str, expected: str) -> None:
    assert lowercase_first_letter(raw) == expected
