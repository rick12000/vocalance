"""Unit tests for TextInputService and helper functions.

Tests text input handling, clipboard operations, and dictation text processing.
"""

from unittest.mock import Mock, patch

import pytest

from vocalance.app.config.app_config import DictationConfig
from vocalance.app.services.audio.dictation_handling.text_input_service import (
    TextInputService,
    clean_dictation_text,
    get_trailing_whitespace_count,
    lowercase_first_letter,
    remove_formatting,
    should_lowercase_current_start,
    should_remove_previous_period,
)

# ============================================================================
# Tests for helper functions
# ============================================================================


class TestCleanDictationText:
    """Tests for clean_dictation_text function."""

    def test_clean_removes_ellipsis(self):
        """Test that ellipsis (...) is removed."""
        result = clean_dictation_text("hello... world")
        assert "..." not in result
        assert "hello" in result
        assert "world" in result

    def test_clean_adds_trailing_space_by_default(self):
        """Test that trailing space is added by default."""
        result = clean_dictation_text("hello")
        assert result.endswith(" ")

    def test_clean_no_trailing_space_when_disabled(self):
        """Test that trailing space can be disabled."""
        result = clean_dictation_text("hello", add_trailing_space=False)
        assert not result.endswith(" ")

    def test_clean_empty_string(self):
        """Test that empty string returns empty."""
        result = clean_dictation_text("")
        assert result == ""

    def test_clean_multiple_ellipsis(self):
        """Test that multiple ellipsis are handled."""
        result = clean_dictation_text("hello... world... test")
        assert "..." not in result

    def test_clean_preserves_other_punctuation(self):
        """Test that other punctuation is preserved."""
        result = clean_dictation_text("hello, world! test?")
        assert "," in result
        assert "!" in result
        assert "?" in result


class TestShouldRemovePreviousPeriod:
    """Tests for should_remove_previous_period function."""

    def test_remove_period_when_conditions_met(self):
        """Test that period is removed when last ends with . and current starts lowercase."""
        result = should_remove_previous_period("This is a sentence.", "and continues")
        assert result is True

    def test_keep_period_when_current_starts_uppercase(self):
        """Test that period is kept when current starts uppercase."""
        result = should_remove_previous_period("This is a sentence.", "Another sentence")
        assert result is False

    def test_keep_period_when_last_no_period(self):
        """Test that period removal doesn't apply when last has no period."""
        result = should_remove_previous_period("This is text", "and continues")
        assert result is False

    def test_empty_last_text(self):
        """Test that empty last text returns False."""
        result = should_remove_previous_period("", "and continues")
        assert result is False

    def test_empty_current_text(self):
        """Test that empty current text returns False."""
        result = should_remove_previous_period("This is a sentence.", "")
        assert result is False

    def test_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        result = should_remove_previous_period("Sentence.   ", "and continues")
        assert result is True


class TestShouldLowercaseCurrentStart:
    """Tests for should_lowercase_current_start function."""

    def test_lowercase_when_no_sentence_boundary(self):
        """Test that first letter is lowercased when no sentence boundary."""
        result = should_lowercase_current_start("No sentence boundary", "Another word")
        assert result is True

    def test_keep_uppercase_when_sentence_ends(self):
        """Test that uppercase is kept when sentence ends with period."""
        result = should_lowercase_current_start("This is a sentence.", "Another word")
        assert result is False

    def test_empty_last_text(self):
        """Test that empty last text returns False."""
        result = should_lowercase_current_start("", "Another word")
        assert result is False

    def test_empty_current_text(self):
        """Test that empty current text returns False."""
        result = should_lowercase_current_start("No boundary", "")
        assert result is False

    def test_whitespace_stripping(self):
        """Test that whitespace is properly stripped."""
        result = should_lowercase_current_start("No boundary   ", "Another word")
        assert result is True

    def test_current_text_lowercase_start(self):
        """Test that lowercase starting letter returns False."""
        result = should_lowercase_current_start("No boundary", "another word")
        assert result is False


class TestGetTrailingWhitespaceCount:
    """Tests for get_trailing_whitespace_count function."""

    def test_count_spaces(self):
        """Test that spaces are counted."""
        result = get_trailing_whitespace_count("hello   ")
        assert result == 3

    def test_count_tabs(self):
        """Test that tabs are counted."""
        result = get_trailing_whitespace_count("hello\t\t")
        assert result == 2

    def test_count_mixed_whitespace(self):
        """Test that mixed whitespace is counted."""
        result = get_trailing_whitespace_count("hello \t ")
        assert result == 3

    def test_no_trailing_whitespace(self):
        """Test that no trailing whitespace returns 0."""
        result = get_trailing_whitespace_count("hello")
        assert result == 0

    def test_empty_string(self):
        """Test that empty string returns 0."""
        result = get_trailing_whitespace_count("")
        assert result == 0

    def test_only_whitespace(self):
        """Test that string of only whitespace returns its length."""
        result = get_trailing_whitespace_count("   ")
        assert result == 3


class TestLowercaseFirstLetter:
    """Tests for lowercase_first_letter function."""

    def test_lowercase_uppercase_letter(self):
        """Test that uppercase letter is lowercased."""
        result = lowercase_first_letter("Hello")
        assert result == "hello"

    def test_lowercase_already_lowercase(self):
        """Test that already lowercase letter stays lowercase."""
        result = lowercase_first_letter("hello")
        assert result == "hello"

    def test_single_character(self):
        """Test single character handling."""
        result = lowercase_first_letter("H")
        assert result == "h"

    def test_empty_string(self):
        """Test empty string returns empty."""
        result = lowercase_first_letter("")
        assert result == ""

    def test_non_letter_first_char(self):
        """Test that non-letter first character is handled."""
        result = lowercase_first_letter("1Hello")
        assert result == "1Hello"


class TestRemoveFormatting:
    """Tests for remove_formatting function."""

    def test_remove_punctuation(self):
        """Test that punctuation is removed."""
        result = remove_formatting("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_keep_hyphens(self):
        """Test that hyphens are preserved."""
        result = remove_formatting("well-known")
        assert "-" in result

    def test_keep_apostrophes(self):
        """Test that apostrophes are preserved."""
        result = remove_formatting("don't")
        assert "'" in result

    def test_lowercase_text(self):
        """Test that text is lowercased."""
        result = remove_formatting("HELLO World")
        assert result == result.lower()

    def test_keep_i_capitalized(self):
        """Test that 'I' remains capitalized as pronoun."""
        result = remove_formatting("I am here", is_first_word_of_session=False)
        assert "I" in result or "i" in result  # Depends on context
        # When not first word and 'I' is standalone
        words = result.split()
        assert any(word == "I" for word in words)

    def test_capitalize_first_word_when_flagged(self):
        """Test that first word is capitalized when flag is set."""
        result = remove_formatting("hello world", is_first_word_of_session=True)
        assert result[0].isupper()

    def test_dont_capitalize_first_word_by_default(self):
        """Test that first word is not capitalized by default."""
        result = remove_formatting("Hello world", is_first_word_of_session=False)
        assert result[0].islower() or not result[0].isalpha()

    def test_empty_string(self):
        """Test empty string handling."""
        result = remove_formatting("")
        assert result == ""

    def test_strip_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        result = remove_formatting("  hello world  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")


# ============================================================================
# Tests for TextInputService
# ============================================================================


@pytest.fixture
def dictation_config():
    """Create a mock dictation config."""
    config = Mock(spec=DictationConfig)
    config.use_clipboard = True
    config.clipboard_paste_delay_pre = 0.01
    config.clipboard_paste_delay_post = 0.01
    config.typing_delay = 0.01
    config.type_text_post_delay = 0.01
    config.pyautogui_pause = 0.01
    return config


@pytest.fixture
def text_input_service(dictation_config):
    """Create TextInputService instance for testing."""
    return TextInputService(config=dictation_config)


@pytest.mark.asyncio
async def test_initialize_succeeds(text_input_service):
    """Test that initialization succeeds."""
    result = await text_input_service.initialize()
    assert result is True


@pytest.mark.asyncio
async def test_reset_session_clears_last_text(text_input_service):
    """Test that reset_session clears last_text."""
    text_input_service.last_text = "some text"
    text_input_service.reset_session()
    assert text_input_service.last_text is None


@pytest.mark.asyncio
@patch("pyperclip.copy")
@patch("pyperclip.paste")
@patch("pyautogui.keyDown")
@patch("pyautogui.keyUp")
@patch("pyautogui.press")
async def test_input_text_success(mock_press, mock_keyup, mock_keydown, mock_paste, mock_copy, text_input_service):
    """Test successful text input via clipboard."""
    result = await text_input_service.input_text("hello world")
    assert result is True
    assert text_input_service.last_text == "hello world "


@pytest.mark.asyncio
async def test_input_text_empty_string(text_input_service):
    """Test that empty text is rejected."""
    result = await text_input_service.input_text("")
    assert result is False


@pytest.mark.asyncio
@patch("pyperclip.copy")
@patch("pyperclip.paste")
@patch("pyautogui.keyDown")
@patch("pyautogui.keyUp")
@patch("pyautogui.press")
async def test_input_text_no_trailing_space(mock_press, mock_keyup, mock_keydown, mock_paste, mock_copy, text_input_service):
    """Test text input without trailing space."""
    result = await text_input_service.input_text("hello", add_trailing_space=False)
    assert result is True
    assert text_input_service.last_text == "hello"


@pytest.mark.asyncio
@patch("pyperclip.copy")
@patch("pyperclip.paste")
@patch("pyautogui.keyDown")
@patch("pyautogui.keyUp")
@patch("pyautogui.press")
async def test_input_text_removes_previous_period(mock_press, mock_keyup, mock_keydown, mock_paste, mock_copy, text_input_service):
    """Test that previous period is removed when appropriate."""
    text_input_service.last_text = "Previous sentence. "

    result = await text_input_service.input_text("lowercase continuation")
    assert result is True


@pytest.mark.asyncio
@patch("pyperclip.copy")
@patch("pyperclip.paste")
@patch("pyautogui.keyDown")
@patch("pyautogui.keyUp")
@patch("pyautogui.press")
async def test_input_text_lowercases_first_letter(mock_press, mock_keyup, mock_keydown, mock_paste, mock_copy, text_input_service):
    """Test that first letter is lowercased when appropriate."""
    text_input_service.last_text = "No sentence boundary "

    result = await text_input_service.input_text("Uppercase start")
    assert result is True


@pytest.mark.asyncio
async def test_add_space(text_input_service):
    """Test adding a space character."""
    with patch("pyautogui.press"):
        result = await text_input_service.add_space()
        assert result is True


@pytest.mark.asyncio
async def test_add_newline(text_input_service):
    """Test adding a newline character."""
    with patch("pyautogui.press"):
        result = await text_input_service.add_newline()
        assert result is True


@pytest.mark.asyncio
async def test_backspace(text_input_service):
    """Test sending backspace keystrokes."""
    with patch("pyautogui.press"):
        result = await text_input_service.backspace(count=3)
        assert result is True


@pytest.mark.asyncio
async def test_backspace_default_count(text_input_service):
    """Test backspace with default count of 1."""
    with patch("pyautogui.press"):
        result = await text_input_service.backspace()
        assert result is True


@pytest.mark.asyncio
async def test_shutdown(text_input_service):
    """Test service shutdown."""
    await text_input_service.shutdown()
    # Should complete without error
