from typing import Optional

import pytest

from vocalance.app.events.dictation_events import DictationModifierId
from vocalance.app.services.dictation_flow.postprocess.base_postprocess import apply_base_postprocess
from vocalance.app.services.dictation_flow.postprocess.modifier_postprocess import apply_modifier_transform, modifier_display_label
from vocalance.app.services.dictation_flow.postprocess.postprocess_pipeline import (
    apply_dictation_postprocess,
    apply_dictation_postprocess_partial,
)
from vocalance.app.services.dictation_flow.postprocess.segment_text import clean_dictation_text, remove_formatting


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello  world", "hello world"),
        ("  leading and trailing  ", "leading and trailing"),
        ("", ""),
        ("unchanged", "unchanged"),
    ],
)
def test_apply_base_postprocess_normalises_whitespace(raw: str, expected: str) -> None:
    assert apply_base_postprocess(raw) == expected



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
        ("hello comma world", {"spelling"}, "Hello, world"),
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


