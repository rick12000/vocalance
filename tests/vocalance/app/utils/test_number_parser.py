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
        ("12.5", True),
        ("not a number", False),
        ("", False),
    ],
)
def test_is_number(text, expected):
    """Test number detection."""
    assert is_number(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("won hundred", "one hundred"),
        ("to fifty", "two fifty"),
        ("ate apples", "eight apples"),
        ("for the win", "four the win"),
        ("free pizza", "three pizza"),
        ("normal text", "normal text"),
    ],
)
def test_normalize_homophones(text, expected):
    """Homophone normalization is on by default (command-style)."""
    assert normalize_homophones(text) == expected


def test_normalize_homophones_disabled_leaves_tokens():
    """Dictation-style: no homophone rewrites."""
    assert normalize_homophones("to fifty", apply_homophones=False) == "to fifty"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("four hundred and nine", "four hundred nine"),
        ("twenty and three", "twenty three"),
        ("one and done", "one and done"),
        ("red and blue", "red and blue"),
    ],
)
def test_remove_number_conjunctions(text, expected):
    """Test removing 'and' between number words."""
    assert remove_number_conjunctions(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("four zero nine", "409"),
        ("one two three", "123"),
        ("zero zero seven", "007"),
        ("four hundred nine", None),
        ("one thousand", None),
        ("seven", None),
    ],
)
def test_detect_digit_sequence(text, expected):
    """Test detecting sequences of spoken digits."""
    assert detect_digit_sequence(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("123", 123),
        ("1,234", 1234),
        ("one", 1),
        ("twenty three", 23),
        ("one hundred", 100),
        ("four hundred nine", 409),
        ("one thousand", 1000),
        ("five thousand", 5000),
    ],
)
def test_parse_spoken_integer_basic(text, expected):
    """Spoken and numeric phrases resolve to integers."""
    assert parse_spoken_integer(text) == expected


def test_parse_spoken_integer_complex_numbers():
    """Long cardinals and mixed scales."""
    assert parse_spoken_integer("twenty three") == 23
    assert parse_spoken_integer("one hundred forty two") == 142
    assert parse_spoken_integer("three thousand five hundred") == 3500


def test_parse_spoken_integer_with_hyphens():
    """Hyphens normalize like spaces."""
    assert parse_spoken_integer("twenty-three") == 23
    assert parse_spoken_integer("forty-five") == 45


def test_parse_spoken_integer_invalid_returns_none():
    """Invalid input yields None."""
    assert parse_spoken_integer("not a number") is None
    assert parse_spoken_integer("") is None
    assert parse_spoken_integer(None) is None


def test_parse_spoken_integer_homophones_flag():
    """Single-token homophones parse by default; dictation can disable."""
    assert parse_spoken_integer("to") == 2
    assert parse_spoken_integer("won") == 1
    assert parse_spoken_integer("to", apply_homophones=False) is None


@pytest.mark.parametrize(
    "text,min_val,max_val,expected",
    [
        ("123", 1, 5000, 123),
        ("5", 1, 10, 5),
        ("0", 1, 10, None),
        ("15", 1, 10, None),
        ("one thousand", 1, 5000, 1000),
        ("five hundred", 1, 5000, 500),
    ],
)
def test_parse_number_with_range(text, min_val, max_val, expected):
    """parse_number respects min/max range."""
    assert parse_number(text, min_value=min_val, max_value=max_val) == expected


def test_parse_number_homophones_default_for_commands():
    """Command parsing uses homophones by default; optional opt-out."""
    assert parse_number("won hundred") == 100
    assert parse_number("to hundred") == 200
    assert parse_number("won hundred", apply_homophones=False) is None
    assert parse_number("to hundred", apply_homophones=False) is None


def test_parse_number_digit_sequences():
    """parse_number detects digit sequences."""
    assert parse_number("four zero nine") == 409
    assert parse_number("one two three") == 123


def test_parse_number_with_conjunctions():
    """Conjunction stripping before cardinal parse."""
    assert parse_number("four hundred and nine") == 409
    assert parse_number("twenty and three") == 23


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "not a number",
        "random text",
    ],
)
def test_parse_number_invalid_input(text):
    """parse_number returns None for invalid input."""
    assert parse_number(text) is None


def test_parse_number_default_range():
    """Default range is 1-5000."""
    assert parse_number("1") == 1
    assert parse_number("5000") == 5000
    assert parse_number("0") is None
    assert parse_number("5001") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("42", 42),
        ("one hundred", 100),
        ("nine hundred ninety nine", 999),
        ("two thousand", 2000),
    ],
)
def test_parse_number_various_formats(text, expected):
    """parse_number handles common formats."""
    assert parse_number(text) == expected


def test_parse_number_case_insensitive():
    """Input is case-insensitive after normalization."""
    assert parse_number("ONE HUNDRED") == 100
    assert parse_number("Twenty Three") == 23
    assert parse_number("FoRtY tWo") == 42


@pytest.mark.parametrize(
    "text,expected",
    [
        ("one two three four five", "12345"),
        ("zero", None),
        ("five six", "56"),
        ("nine eight seven", "987"),
    ],
)
def test_detect_digit_sequence_edge_cases(text, expected):
    """Digit sequence edge cases."""
    assert detect_digit_sequence(text) == expected


def test_parse_spoken_integer_scales():
    """Scale words beyond default parse_number cap still parse as integers."""
    assert parse_spoken_integer("five hundred") == 500
    assert parse_spoken_integer("two thousand") == 2000
    assert parse_spoken_integer("three thousand five hundred") == 3500
    assert parse_spoken_integer("ten thousand") == 10000


def test_parse_number_ten_thousand_out_of_default_range():
    """Ten thousand exceeds default command grid max."""
    assert parse_number("ten thousand") is None
    assert parse_number("ten thousand", min_value=1, max_value=20000) == 10000


def test_parse_spoken_integer_ascii_numeric_strings():
    """Plain digit strings parse."""
    assert parse_spoken_integer("123") == 123
    assert parse_spoken_integer("1234") == 1234


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5", 5),
        ("50", 50),
        ("500", 500),
        ("5000", 5000),
    ],
)
def test_parse_number_multiples_of_five(text, expected):
    """Multiples of five as digits."""
    assert parse_number(text) == expected


def test_normalize_homophones_preserves_non_homophones():
    """Non-homophone tokens unchanged."""
    assert normalize_homophones("hello world") == "hello world"
    assert normalize_homophones("test case") == "test case"


def test_remove_number_conjunctions_multiple_ands():
    """Multiple *and* between number words removed."""
    text = "one and two and three"
    result = remove_number_conjunctions(text)
    assert "and" not in result
    assert "one" in result and "two" in result and "three" in result


def test_is_number_with_commas():
    """Comma-separated numeric strings."""
    assert is_number("1,234") is True
    assert is_number("10,000,000") is True


def test_parse_number_with_commas():
    """parse_number accepts comma-separated digits."""
    assert parse_number("1,234") == 1234
    assert parse_number("5,000") == 5000


def test_parse_number_teens():
    """Teen cardinals."""
    assert parse_number("thirteen") == 13
    assert parse_number("fourteen") == 14
    assert parse_number("fifteen") == 15
    assert parse_number("sixteen") == 16
    assert parse_number("seventeen") == 17
    assert parse_number("eighteen") == 18
    assert parse_number("nineteen") == 19


def test_parse_number_tens():
    """Round tens."""
    assert parse_number("twenty") == 20
    assert parse_number("thirty") == 30
    assert parse_number("forty") == 40
    assert parse_number("fifty") == 50
    assert parse_number("sixty") == 60
    assert parse_number("seventy") == 70
    assert parse_number("eighty") == 80
    assert parse_number("ninety") == 90


def test_parse_number_compound_numbers():
    """Tens + ones."""
    assert parse_number("twenty one") == 21
    assert parse_number("thirty five") == 35
    assert parse_number("forty two") == 42
    assert parse_number("ninety nine") == 99


def test_parse_number_hundreds_with_tens():
    """Hundreds plus remainder."""
    assert parse_number("one hundred twenty three") == 123
    assert parse_number("two hundred fifty six") == 256
    assert parse_number("nine hundred ninety nine") == 999


def test_parse_spoken_integer_error_handling():
    """No exceptions on bad input."""
    assert parse_spoken_integer(None) is None
    assert parse_spoken_integer("") is None
    assert parse_spoken_integer("xyz") is None


def test_parse_number_handles_numeric_types():
    """int/float coerced via str then parsed."""
    assert parse_number(123) == 123
    assert parse_number(45.6) == 45


def test_replace_spoken_numbers_in_text_simple():
    """Default (command-style) merges spoken runs and homophones into digits."""
    assert replace_spoken_numbers_in_text("call four zero nine") == "call 409"
    assert replace_spoken_numbers_in_text("row twenty three") == "row 23"
    assert replace_spoken_numbers_in_text("row to three") == "row 23"


def test_replace_spoken_numbers_does_not_map_common_words_when_homophones_off():
    """Dictation path: pass ``apply_homophones=False`` so *to* / *for* stay words."""
    assert replace_spoken_numbers_in_text("go to the store", apply_homophones=False) == "go to the store"
    assert replace_spoken_numbers_in_text("waiting for you", apply_homophones=False) == "waiting for you"


def test_replace_spoken_numbers_in_text_preserves_punctuation():
    """Leading/trailing punctuation on edge tokens kept."""
    assert replace_spoken_numbers_in_text("(four zero nine)") == "(409)"


def test_replace_spoken_numbers_in_text_skips_existing_digits():
    """Tokens that are already numeric are not merged as spoken digits."""
    assert replace_spoken_numbers_in_text("pin 1 2 3") == "pin 1 2 3"


def test_replace_spoken_numbers_in_text_empty():
    """Empty or whitespace-only input unchanged."""
    assert replace_spoken_numbers_in_text("") == ""
    assert replace_spoken_numbers_in_text("   ") == "   "
