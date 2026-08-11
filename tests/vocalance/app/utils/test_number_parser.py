import pytest

from vocalance.app.utils.number_parser import (
    detect_digit_sequence,
    is_number,
    normalize_homophones,
    parse_number,
    parse_spoken_integer,
    remove_number_conjunctions,
    replace_spoken_numbers_in_text,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("123", True),
        ("1,234", True),
        ("10,000,000", True),
        ("12.5", True),
        ("-7", True),
        ("not a number", False),
        ("", False),
    ],
)
def test_is_number(text, expected):
    assert is_number(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("won hundred", "one hundred"),
        ("to fifty", "two fifty"),
        ("ate apples", "eight apples"),
        ("FOR the win", "four the win"),
        ("normal text", "normal text"),
    ],
)
def test_normalize_homophones_maps_and_lowercases(text, expected):
    assert normalize_homophones(text) == expected


def test_normalize_homophones_disabled_keeps_words():
    assert normalize_homophones("to fifty", apply_homophones=False) == "to fifty"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("four hundred and nine", "four hundred nine"),
        ("one and two and three", "one two three"),
        ("one and done", "one and done"),
        ("red and blue", "red and blue"),
    ],
)
def test_remove_number_conjunctions(text, expected):
    assert remove_number_conjunctions(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("four zero nine", "409"),
        ("zero zero seven", "007"),
        ("one two three four five", "12345"),
        ("seven", None),
        ("one thousand", None),
        ("four hundred nine", None),
    ],
)
def test_detect_digit_sequence(text, expected):
    assert detect_digit_sequence(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("123", 123),
        ("1,234", 1234),
        ("one", 1),
        ("twenty three", 23),
        ("twenty-three", 23),
        ("one hundred forty two", 142),
        ("four hundred and nine", 409),
        ("three thousand five hundred", 3500),
        ("ten thousand", 10000),
        ("four zero nine", 409),
    ],
)
def test_parse_spoken_integer(text, expected):
    assert parse_spoken_integer(text) == expected


@pytest.mark.parametrize("text", ["not a number", "first", "", None])
def test_parse_spoken_integer_invalid_returns_none(text):
    assert parse_spoken_integer(text) is None


def test_parse_spoken_integer_homophones_flag():
    assert parse_spoken_integer("to") == 2
    assert parse_spoken_integer("to", apply_homophones=False) is None


@pytest.mark.parametrize(
    "text,min_val,max_val,expected",
    [
        ("123", 1, 5000, 123),
        ("5000", 1, 5000, 5000),
        ("5001", 1, 5000, None),
        ("0", 1, 10, None),
        ("15", 1, 10, None),
        ("ten thousand", 1, 5000, None),
        ("ten thousand", 1, 20000, 10000),
    ],
)
def test_parse_number_enforces_range(text, min_val, max_val, expected):
    assert parse_number(text, min_value=min_val, max_value=max_val) == expected


def test_parse_number_homophones_default_for_commands():
    assert parse_number("won hundred") == 100
    assert parse_number("won hundred", apply_homophones=False) is None


def test_parse_number_coerces_numeric_types():
    assert parse_number(123) == 123
    assert parse_number(45.6) == 45


@pytest.mark.parametrize("text", ["", None, "not a number"])
def test_parse_number_invalid_returns_none(text):
    assert parse_number(text) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("call four zero nine", "call 409"),
        ("row twenty three", "row 23"),
        ("row to three", "row 23"),
        ("(four zero nine)", "(409)"),
        ("pin 1 2 3", "pin 1 2 3"),
    ],
)
def test_replace_spoken_numbers_in_text(text, expected):
    assert replace_spoken_numbers_in_text(text) == expected


@pytest.mark.parametrize(
    "text",
    ["go to the store", "waiting for you"],
)
def test_replace_spoken_numbers_keeps_homophones_when_disabled(text):
    assert replace_spoken_numbers_in_text(text, apply_homophones=False) == text


@pytest.mark.parametrize("text", ["", "   "])
def test_replace_spoken_numbers_empty_unchanged(text):
    assert replace_spoken_numbers_in_text(text) == text
