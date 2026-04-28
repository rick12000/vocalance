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


@pytest.mark.parametrize(
    "raw,expected_substr",
    [
        ("I have twenty three apples", "23"),
        ("count is one hundred forty two", "142"),
        ("code four zero nine end", "409"),
    ],
)
def test_apply_base_postprocess_spoken_numbers(raw: str, expected_substr: str) -> None:
    assert expected_substr in apply_base_postprocess(raw)


def test_apply_base_postprocess_does_not_homophone_map_common_words() -> None:
    """Dictation keeps ordinary *to* / *for*; homophones apply only on command paths."""
    assert apply_base_postprocess("go to the store") == "go to the store"
    assert apply_base_postprocess("waiting for you") == "waiting for you"


def test_strip_trailing_period_after_number() -> None:
    assert strip_trailing_period_after_numbers("total 42.") == "total 42"
    assert strip_trailing_period_after_numbers("total 42 .") == "total 42"
    assert strip_trailing_period_after_numbers("pi 3.14") == "pi 3.14"


@pytest.mark.parametrize(
    "modifier_id,expected_label",
    [
        ("upper", "Upper"),
        ("capitals", "Capitals"),
        ("camel", "Camel"),
        ("snake", "Snake"),
        ("spelling", "Spelling"),
    ],
)
def test_modifier_display_label(modifier_id: DictationModifierId, expected_label: str) -> None:
    assert modifier_display_label(modifier_id) == expected_label


@pytest.mark.parametrize(
    "raw,modifier_id,expected",
    [
        ("hello world", "upper", "Hello World"),
        ("HELLO WORLD", "upper", "Hello World"),
        ("a", "upper", "A"),
        ("", "upper", ""),
        ("hello world", "capitals", "HELLO WORLD"),
        ("Hi There", "capitals", "HI THERE"),
        ("foo bar baz", "camel", "FooBarBaz"),
        ("foo, bar! baz", "camel", "FooBarBaz"),
        ("foo_bar baz", "camel", "FooBarBaz"),
        ("FOO BAR", "camel", "FooBar"),
        ("foo 123 bar", "camel", "Foo123Bar"),
        ("a", "camel", "A"),
        ("", "camel", ""),
        ("Foo Bar", "snake", "foo_bar"),
        ("Foo, Bar!", "snake", "foo_bar"),
        ("foo__bar", "snake", "foo_bar"),
        ("A B", "snake", "a_b"),
        ("", "snake", ""),
    ],
)
def test_apply_modifier_transform_text_transforms(raw: str, modifier_id: DictationModifierId, expected: str) -> None:
    assert apply_modifier_transform(raw, modifier_id) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello period", "Hello."),
        ("hello comma world", "Hello,world"),
        ("hello question mark there", "Hello?There"),
        ("", ""),
    ],
)
def test_apply_modifier_transform_spelling(raw: str, expected: str) -> None:
    assert apply_modifier_transform(raw, "spelling") == expected


def test_apply_dictation_postprocess_none_modifier_is_base_only() -> None:
    assert apply_dictation_postprocess("hello world", None) == "hello world"
    assert apply_dictation_postprocess("", None) == ""


def test_apply_dictation_postprocess_base_then_modifier() -> None:
    assert apply_dictation_postprocess("one two three", {"spelling"}) == "123"


@pytest.mark.parametrize(
    "raw,modifier_id,expected",
    [
        ("foo bar", None, "foo bar"),
        ("foo bar", "spelling", "foo bar"),
        ("foo bar", "upper", "Foo Bar"),
        ("foo bar", "camel", "FooBar"),
    ],
)
def test_apply_dictation_postprocess_partial(raw: str, modifier_id: Optional[DictationModifierId], expected: str) -> None:
    modifiers_set = {modifier_id} if modifier_id else set()
    assert apply_dictation_postprocess_partial(raw, modifiers_set) == expected
