import asyncio
import logging
import re
import threading
import time
from typing import Optional

import pyautogui
import pyperclip

from vocalance.app.config.app_config import DictationConfig

logger = logging.getLogger(__name__)


def clean_dictation_text(text: str, add_trailing_space: bool = True) -> str:
    """
    Clean dictation text by removing "..." and optionally adding trailing space.

    Args:
        text: Raw text to clean
        add_trailing_space: If True, add trailing space for proper concatenation

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    cleaned = re.sub(r"\.\.\.", " ", text)

    if add_trailing_space and cleaned and not cleaned[-1].isspace():
        cleaned = cleaned + " "

    return cleaned


def should_remove_previous_period(last_text: str, current_text: str) -> bool:
    """
    Determine if the period from the last segment should be removed when concatenating.

    This handles the case where:
    - Last segment ends with "."
    - Current segment starts with a lowercase letter

    Args:
        last_text: The previously pasted/typed text
        current_text: The current segment being processed

    Returns:
        True if the period should be removed, False otherwise
    """
    if not last_text or not current_text:
        return False

    return last_text.rstrip().endswith(".") and current_text.strip() and current_text.strip()[0].islower()


def should_lowercase_current_start(last_text: str, current_text: str) -> bool:
    """
    Determine if the first letter of current text should be lowercased.

    Applied when joining segments where:
    - Current segment starts with a capital letter
    - Last segment (stripped) does NOT end with a period (no sentence boundary)

    This ensures proper case for mid-sentence concatenation.

    Args:
        last_text: The previously pasted/typed text
        current_text: The current segment being processed

    Returns:
        True if current text's first letter should be lowercased, False otherwise
    """
    if not last_text or not current_text:
        return False

    last_stripped = last_text.rstrip()
    current_stripped = current_text.strip()

    return last_stripped and not last_stripped.endswith(".") and current_stripped and current_stripped[0].isupper()


def get_trailing_whitespace_count(text: str) -> int:
    """
    Calculate the number of trailing whitespace characters.

    Args:
        text: Text to measure

    Returns:
        Count of trailing whitespace characters
    """
    if not text:
        return 0
    return len(text) - len(text.rstrip())


def lowercase_first_letter(text: str) -> str:
    """
    Lowercase the first character of text.

    Used when joining text segments without a sentence boundary to maintain
    proper capitalization for mid-sentence concatenation.

    Args:
        text: Text to process

    Returns:
        Text with first character lowercased, or original text if empty
    """
    if not text:
        return text
    return text[0].lower() + text[1:] if len(text) > 1 else text[0].lower()


def remove_formatting(text: str, is_first_word_of_session: bool = False) -> str:
    """
    Remove formatting from dictation text.

    Applied when enable_dictation_formatting=False to provide clean, unformatted output:
    - Remove all punctuation except hyphens and apostrophes
    - Convert to lowercase
    - Keep 'I' capitalized when used as pronoun (standalone or in contractions)
    - Capitalize first word of session if is_first_word_of_session is True
    - Strip all leading and trailing whitespace

    Args:
        text: Text to process
        is_first_word_of_session: If True, capitalize the first word

    Returns:
        Cleaned, unformatted text
    """
    if not text:
        return ""

    # Remove all punctuation except hyphens, apostrophes, and spaces
    cleaned = re.sub(r"[^\w\s\-']", "", text)

    # Convert to lowercase
    cleaned = cleaned.lower()

    # Process words: capitalize first word if needed, and keep 'I' capitalized as pronoun
    words = cleaned.split()
    if words:
        # Capitalize first word if this is the first word of the session
        if is_first_word_of_session:
            words[0] = words[0].capitalize()

        # Keep 'I' capitalized when used as pronoun (standalone or in contractions)
        words = [word if not (word == "i" or word.startswith("i'")) else word.replace("i", "I", 1) for word in words]

        cleaned = " ".join(words)

    # Strip all leading and trailing whitespace
    return cleaned.strip()


class TextInputService:
    """Text input service with clipboard-based and typing-based input methods."""

    def __init__(self, config: DictationConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._clipboard_lock = threading.Lock()  # Prevent concurrent clipboard operations
        self.last_text: Optional[str] = None
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = config.pyautogui_pause

        logger.info("TextInputService initialized")

    async def initialize(self) -> bool:
        try:
            logger.info("TextInputService ready")
            return True
        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False

    def reset_session(self) -> None:
        """Reset session state before starting a new dictation session."""
        with self._clipboard_lock:
            self.last_text = None
        logger.debug("TextInputService session reset")

    def capture_selection_via_copy(self) -> str:
        """Copy the current foreground selection via Ctrl+C and return captured text.

        Restores the previous clipboard contents after reading. Used by amend mode
        when the target application still holds the selection.
        """
        with self._clipboard_lock:
            original = None
            original_read_ok = False
            try:
                try:
                    original = pyperclip.paste()
                    original_read_ok = True
                except (pyperclip.PyperclipException, OSError) as e:
                    logger.warning(f"Could not read clipboard before copy: {e}")

                time.sleep(self.config.clipboard_paste_delay_pre)
                pyautogui.hotkey("ctrl", "c")
                time.sleep(max(0.05, self.config.clipboard_paste_delay_post))

                captured = ""
                try:
                    captured = pyperclip.paste() or ""
                except (pyperclip.PyperclipException, OSError) as e:
                    logger.warning(f"Could not read clipboard after copy: {e}")

                if original_read_ok and original is not None:
                    try:
                        pyperclip.copy(original)
                    except (pyperclip.PyperclipException, OSError) as e:
                        logger.warning(f"Could not restore clipboard after capture: {e}")

                return captured.strip()
            except Exception as e:
                logger.error(f"capture_selection_via_copy error: {e}", exc_info=True)
                return ""

    async def input_text(
        self,
        text: str,
        add_trailing_space: bool = True,
        skip_prose_segment_join_rules: bool = False,
    ) -> bool:
        """Paste or type ``text`` after :func:`clean_dictation_text`.

        When ``skip_prose_segment_join_rules`` is False (default), applies period removal and mid-sentence
        lowercasing against ``last_text`` so consecutive dictation segments read as one sentence. Identifier
        and spelling modifiers pass ``True`` to preserve casing and avoid a trailing space.
        """
        if not text:
            return False

        try:
            cleaned_text = clean_dictation_text(text=text, add_trailing_space=add_trailing_space)
            if not cleaned_text:
                return False

            if not skip_prose_segment_join_rules:
                if self.last_text and should_remove_previous_period(self.last_text, cleaned_text):
                    trailing_whitespace_count = get_trailing_whitespace_count(self.last_text)
                    await self.backspace(1 + trailing_whitespace_count)
                    cleaned_text = " " + cleaned_text

                if self.last_text and should_lowercase_current_start(self.last_text, cleaned_text):
                    cleaned_text = lowercase_first_letter(cleaned_text)

            if self.config.use_clipboard:
                success = await asyncio.get_event_loop().run_in_executor(None, self._paste_clipboard, cleaned_text)
            else:
                success = await asyncio.get_event_loop().run_in_executor(None, self._type_text, cleaned_text)

            if success:
                logger.debug(f"Input text: '{cleaned_text[:50]}{'...' if len(cleaned_text) > 50 else ''}'")
                self.last_text = cleaned_text

            return success

        except Exception as e:
            logger.error(f"Text input error: {e}", exc_info=True)
            return False

    def _paste_clipboard(self, text: str) -> bool:
        """Paste text using clipboard with atomic operations and verification."""
        with self._clipboard_lock:
            original = None
            original_read_success = False
            try:
                try:
                    original = pyperclip.paste()
                    original_read_success = True
                except (pyperclip.PyperclipException, OSError) as e:
                    logger.warning(f"Could not read original clipboard content: {e}")
                    original = None
                    original_read_success = False

                copy_attempts = 0
                max_copy_attempts = 2
                while copy_attempts < max_copy_attempts:
                    try:
                        pyperclip.copy(text)
                        break
                    except (pyperclip.PyperclipException, OSError) as e:
                        copy_attempts += 1
                        if copy_attempts >= max_copy_attempts:
                            logger.error(
                                f"Could not copy text to clipboard after {max_copy_attempts} attempts: {e}", exc_info=True
                            )
                            return False
                        time.sleep(0.1)

                verify_attempts = 0
                max_verify_attempts = 2
                while verify_attempts < max_verify_attempts:
                    try:
                        clipboard_content = pyperclip.paste()
                        if clipboard_content == text:
                            break
                        verify_attempts += 1
                        if verify_attempts >= max_verify_attempts:
                            logger.error(
                                f"Clipboard content mismatch after {max_verify_attempts} attempts! "
                                f"Expected '{text[:50]}' but got '{clipboard_content[:50]}'"
                            )
                            return False
                        time.sleep(0.01)
                    except (pyperclip.PyperclipException, OSError) as e:
                        verify_attempts += 1
                        if verify_attempts >= max_verify_attempts:
                            logger.warning(f"Could not verify clipboard content after {max_verify_attempts} attempts: {e}")
                            return False
                        time.sleep(0.1)

                time.sleep(self.config.clipboard_paste_delay_pre)

                pyautogui.keyDown("ctrl")
                time.sleep(0.01)
                pyautogui.press("v")
                time.sleep(0.01)
                pyautogui.keyUp("ctrl")

                time.sleep(self.config.clipboard_paste_delay_post)
                time.sleep(0.05)

                if original_read_success and original is not None:
                    try:
                        pyperclip.copy(original)
                    except (pyperclip.PyperclipException, OSError) as e:
                        logger.warning(f"Could not restore clipboard: {e}")

                return True

            except Exception as e:
                logger.error(f"Clipboard paste error: {e}", exc_info=True)
                return False

    def _type_text(self, text: str) -> bool:
        """Type text character by character"""
        try:
            for char in text:
                pyautogui.write(char, interval=self.config.typing_delay)
            time.sleep(self.config.type_text_post_delay)
            return True

        except Exception as e:
            logger.error(f"Text typing error: {e}", exc_info=True)
            return False

    async def add_space(self) -> bool:
        """Add space character"""
        try:
            await asyncio.get_event_loop().run_in_executor(None, pyautogui.press, "space")
            return True
        except Exception as e:
            logger.error(f"Space input error: {e}", exc_info=True)
            return False

    async def add_newline(self) -> bool:
        """Add newline character"""
        try:
            await asyncio.get_event_loop().run_in_executor(None, pyautogui.press, "enter")
            return True
        except Exception as e:
            logger.error(f"Newline input error: {e}", exc_info=True)
            return False

    async def backspace(self, count: int = 1) -> bool:
        """Send backspace keystrokes"""
        try:
            for _ in range(count):
                await asyncio.get_event_loop().run_in_executor(None, pyautogui.press, "backspace")
            return True
        except Exception as e:
            logger.error(f"Backspace error: {e}", exc_info=True)
            return False

    async def shutdown(self) -> None:
        logger.info("TextInputService shutdown complete")


def create_text_input_service(config: DictationConfig) -> TextInputService:
    return TextInputService(config)
