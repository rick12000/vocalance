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


def _setup_clipboard_mocks(mock_paste, mock_copy, expected_text):
    """Helper to configure clipboard mocks for successful paste operations.

    Args:
        mock_paste: pyperclip.paste mock
        mock_copy: pyperclip.copy mock
        expected_text: The text that should be pasted
    """
    # Sequence: read original, verify after copy, verify after paste
    mock_paste.side_effect = ["original_clipboard_content", expected_text, expected_text]
    mock_copy.return_value = None


@pytest.mark.asyncio
async def test_initialize_succeeds(text_input_service):
    """Test that initialization succeeds."""
    result = text_input_service.initialize()
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
    """Test successful text input via clipboard performs complete paste sequence."""
    _setup_clipboard_mocks(mock_paste, mock_copy, "hello world ")

    result = await text_input_service.input_text("hello world")

    # Verify return value
    assert result is True

    # Verify state tracking
    assert text_input_service.last_text == "hello world "

    # Verify clipboard operations occurred
    assert mock_copy.call_count >= 2, "Should copy text and restore original clipboard"
    assert mock_paste.call_count >= 2, "Should read original clipboard and verify paste"

    # Verify paste key sequence: Ctrl+V
    assert mock_keydown.called, "Should press Ctrl key"
    assert mock_keyup.called, "Should release Ctrl key"
    assert mock_press.called, "Should press V key"

    # Verify ctrl key was used
    keydown_args = [str(call) for call in mock_keydown.call_args_list]
    keyup_args = [str(call) for call in mock_keyup.call_args_list]
    assert any("ctrl" in str(arg).lower() for arg in keydown_args), "Should press Ctrl"
    assert any("ctrl" in str(arg).lower() for arg in keyup_args), "Should release Ctrl"

    # Verify V was pressed
    press_args = [call[0][0] if call[0] else "" for call in mock_press.call_args_list]
    assert "v" in press_args, "Should press V key for paste"


@pytest.mark.asyncio
async def test_input_text_empty_string(text_input_service):
    """Test that empty text is rejected without performing clipboard operations."""
    with patch("pyperclip.copy") as mock_copy, patch("pyperclip.paste") as mock_paste, patch(
        "pyautogui.keyDown"
    ) as mock_keydown, patch("pyautogui.keyUp") as mock_keyup, patch("pyautogui.press") as mock_press:

        result = await text_input_service.input_text("")

        assert result is False
        # Empty input should not trigger any clipboard or keyboard operations
        assert mock_copy.call_count == 0, "Should not copy for empty input"
        assert mock_paste.call_count == 0, "Should not paste for empty input"
        assert mock_press.call_count == 0, "Should not press keys for empty input"
        assert mock_keydown.call_count == 0, "Should not press Ctrl for empty input"
        assert mock_keyup.call_count == 0, "Should not release Ctrl for empty input"


@pytest.mark.asyncio
@patch("pyperclip.copy")
@patch("pyperclip.paste")
@patch("pyautogui.keyDown")
@patch("pyautogui.keyUp")
@patch("pyautogui.press")
async def test_input_text_no_trailing_space(mock_press, mock_keyup, mock_keydown, mock_paste, mock_copy, text_input_service):
    """Test text input without trailing space applies correct transformation."""
    _setup_clipboard_mocks(mock_paste, mock_copy, "hello")

    result = await text_input_service.input_text("hello", add_trailing_space=False)

    assert result is True
    # Critical: text should NOT have trailing space
    assert text_input_service.last_text == "hello"
    assert not text_input_service.last_text.endswith(" "), "Should not have trailing space"

    # Verify the exact text was copied to clipboard (without space)
    copy_calls = [call[0][0] for call in mock_copy.call_args_list if call[0]]
    assert "hello" in copy_calls, f"Should copy 'hello' without space, got {copy_calls}"
    # Ensure no variant with space was copied
    assert not any(text == "hello " for text in copy_calls), "Should not copy with trailing space"


@pytest.mark.asyncio
@patch("pyperclip.copy")
@patch("pyperclip.paste")
@patch("pyautogui.keyDown")
@patch("pyautogui.keyUp")
@patch("pyautogui.press")
async def test_input_text_removes_previous_period(mock_press, mock_keyup, mock_keydown, mock_paste, mock_copy, text_input_service):
    """Test that previous period and trailing space are deleted before pasting continuation."""
    text_input_service.last_text = "Previous sentence. "
    _setup_clipboard_mocks(mock_paste, mock_copy, " lowercase continuation ")

    result = await text_input_service.input_text("lowercase continuation")

    assert result is True
    # State should show the text that was pasted
    assert text_input_service.last_text == " lowercase continuation "

    # Verify backspace was pressed to remove period + trailing space (2 chars)
    # The period and the space after it should be deleted
    press_calls = [call[0][0] if call[0] else "" for call in mock_press.call_args_list]
    backspace_count = sum(1 for key in press_calls if key == "backspace")
    assert backspace_count == 2, f"Should backspace 2 times (period + space), got {backspace_count}"

    # Verify the pasted text starts with space (for proper sentence continuity)
    copy_calls = [call[0][0] for call in mock_copy.call_args_list if call[0]]
    pasted_texts = [text for text in copy_calls if "lowercase" in text]
    assert any(text.startswith(" ") for text in pasted_texts), "Continuation should start with space"


@pytest.mark.asyncio
@patch("pyperclip.copy")
@patch("pyperclip.paste")
@patch("pyautogui.keyDown")
@patch("pyautogui.keyUp")
@patch("pyautogui.press")
async def test_input_text_lowercases_first_letter(mock_press, mock_keyup, mock_keydown, mock_paste, mock_copy, text_input_service):
    """Test that first letter is lowercased when continuing without sentence boundary."""
    text_input_service.last_text = "No sentence boundary "
    _setup_clipboard_mocks(mock_paste, mock_copy, "uppercase start ")

    result = await text_input_service.input_text("Uppercase start")

    assert result is True
    # Critical: first letter should be lowercased for mid-sentence continuation
    assert text_input_service.last_text == "uppercase start "
    assert text_input_service.last_text[0] == "u", "First letter should be lowercase 'u', not 'U'"
    assert text_input_service.last_text[0].islower()

    # Verify the copied text has lowercase first letter (not the original uppercase)
    copy_calls = [call[0][0] for call in mock_copy.call_args_list if call[0]]
    texts_with_start = [text for text in copy_calls if "start" in text]
    assert any(text.startswith("u") for text in texts_with_start), f"Should copy with lowercase 'u', but got: {texts_with_start}"
    assert not any(
        text.startswith("U") for text in texts_with_start
    ), f"Should NOT copy with uppercase 'U', but got: {texts_with_start}"


@pytest.mark.asyncio
@patch("pyperclip.copy")
@patch("pyperclip.paste")
@patch("pyautogui.keyDown")
@patch("pyautogui.keyUp")
@patch("pyautogui.press")
async def test_input_text_skip_prose_preserves_leading_capital(
    mock_press, mock_keyup, mock_keydown, mock_paste, mock_copy, text_input_service
):
    """Identifier-style segments must not trigger mid-sentence lowercasing."""
    text_input_service.last_text = "some prior chunk "
    _setup_clipboard_mocks(mock_paste, mock_copy, "HelloWorld")

    result = await text_input_service.input_text("HelloWorld", skip_prose_segment_join_rules=True, add_trailing_space=False)

    assert result is True
    assert text_input_service.last_text == "HelloWorld"
    copy_calls = [call[0][0] for call in mock_copy.call_args_list if call[0]]
    assert "HelloWorld" in copy_calls


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
    text_input_service.shutdown()
    # Should complete without error
