"""Utility functions for parsing spoken numbers to integers.

Provides robust number parsing from text, handling numeric strings ('123'),
number words ('four hundred nine'), homophones ('won' -> 'one'), digit sequences
('four zero nine' -> '409'), conjunctions, and ordinals with multi-stage pipeline
processing and range validation.
"""
import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

NUMBER_WORDS: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

SCALE_WORDS: Set[str] = {"hundred", "thousand", "million", "billion", "trillion"}
SINGLE_DIGIT_WORDS: Set[str] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}

HOMOPHONES: Dict[str, str] = {
    "won": "one",
    "to": "two",
    "too": "two",
    "free": "three",
    "for": "four",
    "fore": "four",
    "ate": "eight",
}


def is_number(text: str) -> bool:
    """Check if text represents a numeric value.

    Handles commas and decimals in numeric strings.

    Args:
        text: Text to check.

    Returns:
        True if text is numeric, False otherwise.
    """
    if isinstance(text, str):
        text = text.replace(",", "")
    try:
        float(text)
    except Exception:
        return False
    return True


def normalize_homophones(text: str) -> str:
    """Replace homophones with standard number words.

    Examples: 'won hundred' -> 'one hundred', 'to fifty' -> 'two fifty'

    Args:
        text: Text to normalize.

    Returns:
        Normalized text with homophones replaced.
    """
    if not isinstance(text, str):
        return text

    words = text.lower().split()
    normalized_words = [HOMOPHONES.get(word, word) for word in words]
    return " ".join(normalized_words)


def remove_number_conjunctions(text: str) -> str:
    """Remove 'and' when it appears between number words.

    Examples: 'four hundred and nine' -> 'four hundred nine'

    Args:
        text: Text to clean.

    Returns:
        Text with number conjunctions removed.
    """
    if not isinstance(text, str):
        return text

    words = text.lower().split()
    all_number_words = NUMBER_WORDS.keys() | SCALE_WORDS

    filtered_words = []
    for i, word in enumerate(words):
        if word == "and":
            prev_is_number = i > 0 and words[i - 1] in all_number_words
            next_is_number = i < len(words) - 1 and words[i + 1] in all_number_words
            if prev_is_number and next_is_number:
                continue
        filtered_words.append(word)

    return " ".join(filtered_words)


def detect_digit_sequence(text: str) -> Optional[str]:
    """Detect if text represents a sequence of individual digits.

    Handles cases where users say individual digits that should be
    concatenated rather than added (e.g., phone numbers, codes).

    Examples: 'four zero nine' -> '409', 'four hundred nine' -> None

    Args:
        text: Text to check.

    Returns:
        String representation of digits if detected, None otherwise.
    """
    if not isinstance(text, str):
        return None

    words = text.lower().split()

    if len(words) > 1 and all(word in SINGLE_DIGIT_WORDS for word in words) and not any(word in SCALE_WORDS for word in words):
        digit_map = {word: str(NUMBER_WORDS[word]) for word in SINGLE_DIGIT_WORDS}
        return "".join(digit_map[word] for word in words)

    return None


def text2int(textnum: str, numwords: Optional[Dict] = None) -> Optional[int]:
    """Convert text number to integer using word-based parsing.

    Handles complex number phrases like 'four hundred twenty three'.

    Args:
        textnum: Text representation of a number.
        numwords: Optional precomputed numwords dictionary.

    Returns:
        Converted integer or None if conversion fails.
    """
    if not textnum:
        return None
    if is_number(textnum):
        try:
            return int(float(textnum.replace(",", "")))
        except ValueError:
            return None

    if numwords is None:
        numwords = {}
        numwords["and"] = (1, 0)
        units = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        scales = ["hundred", "thousand", "million", "billion", "trillion"]
        for idx, word in enumerate(units):
            numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            if word:
                numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            numwords[word] = (10 ** (idx * 3 or 2), 0)

    textnum = str(textnum).replace("-", " ")
    current = result = 0
    curstring = ""
    onnumber = False
    lastunit = False
    lastscale = False

    ordinal_words: Dict[str, int] = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
        "eleventh": 11,
        "twelfth": 12,
        "thirteenth": 13,
        "twentieth": 20,
        "thirtieth": 30,
        "fortieth": 40,
        "fiftieth": 50,
    }
    ordinal_endings = [("ieth", "y"), ("th", "")]

    def is_numword(x: str) -> bool:
        if is_number(x):
            return True
        if x in numwords:
            return True
        return False

    def from_numword(x: str):
        if is_number(x):
            scale = 0
            try:
                increment = int(float(x.replace(",", "")))
                return scale, increment
            except ValueError:
                return 0, 0
        return numwords.get(x, (0, 0))

    try:
        for word in textnum.split():
            word_lower = word.lower()
            if word_lower in ordinal_words:
                scale, increment = (1, ordinal_words[word_lower])
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True
                lastunit = False
                lastscale = False
                continue

            original_word_for_numword_check = word
            for ending, replacement in ordinal_endings:
                if word_lower.endswith(ending):
                    base_word = word_lower[: -len(ending)] + replacement
                    if base_word in numwords:
                        original_word_for_numword_check = base_word
                        break

            if (not is_numword(original_word_for_numword_check)) or (original_word_for_numword_check == "and" and not lastscale):
                if onnumber:
                    curstring += str(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
                lastunit = False
                lastscale = False
            else:
                scale, increment = from_numword(original_word_for_numword_check)
                onnumber = True
                if lastunit and (original_word_for_numword_check not in scales):
                    curstring += str(result + current) + " "
                    result = current = 0
                if scale > 1:
                    current = max(1, current)
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                lastscale = False
                lastunit = False
                if original_word_for_numword_check in scales:
                    lastscale = True
                elif original_word_for_numword_check in numwords:
                    lastunit = True

        if onnumber:
            curstring += str(result + current)

        parts = curstring.strip().split()
        if parts:
            try:
                final_num_str = "".join(filter(str.isdigit, parts[-1]))
                if final_num_str:
                    return int(final_num_str)
                if curstring.strip().isdigit():
                    return int(curstring.strip())
            except ValueError:
                pass
        return None
    except Exception as e:
        logger.error(f"Error in text2int: {e} for input: '{textnum}'", exc_info=True)
        return None


def parse_number(text: str, min_value: int = 1, max_value: int = 5000) -> Optional[int]:
    """Parse text to extract a number using a multi-stage pipeline.

    Stages:
    1. Direct numeric conversion (for '123', '1,234', etc.)
    2. Homophone normalization ('won' -> 'one')
    3. Conjunction removal ('four hundred and nine' -> 'four hundred nine')
    4. Digit sequence detection ('four zero nine' -> '409')
    5. Complex number parsing ('four hundred nine' -> 409)

    Args:
        text: Input text to parse.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).

    Returns:
        Integer if successfully parsed and within range, None otherwise.
    """
    if text is None or text == "":
        logger.debug("Text is None or empty")
        return None

    if isinstance(text, (int, float)):
        text = str(text)
    elif not isinstance(text, str):
        return None

    if is_number(text):
        try:
            num = int(float(text.replace(",", "")))
            if min_value <= num <= max_value:
                return num
            else:
                return None
        except ValueError:
            logger.debug(f"ValueError parsing '{text}' as direct number")

    normalized_text = normalize_homophones(text)
    cleaned_text = remove_number_conjunctions(normalized_text)

    digit_sequence = detect_digit_sequence(cleaned_text)
    if digit_sequence:
        try:
            num = int(digit_sequence)
            if min_value <= num <= max_value:
                return num
            else:
                return None
        except ValueError:
            logger.debug(f"ValueError parsing digit sequence '{digit_sequence}'")

    try:
        num = text2int(cleaned_text)
        if num is not None:
            if min_value <= num <= max_value:
                return num
            else:
                return None
    except Exception as e:
        logger.error(f"Error parsing '{text}' with text2int: {e}", exc_info=True)

    return None
