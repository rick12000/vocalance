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
        """Reset session state to prevent continuation logic from applying to new sessions.
        
        When a new dictation session starts, reset last_text so that the first text pasted
        won't be incorrectly treated as a continuation of previous text, which would cause
        it to be lowercased. This preserves Whisper's native capitalization.
        """
        self.last_text = None
        logger.debug("TextInputService session reset - continuation logic will not apply to next paste")

    async def input_text(self, text: str, add_trailing_space: bool = True) -> bool:
        if not text:
            return False

        try:
            cleaned_text = clean_dictation_text(text=text, add_trailing_space=add_trailing_space)
            if not cleaned_text:
                return False

            # NOTE: If last segment ended with a period and current segment starts with a lowercase letter, remove the period from the previous paste
            # Below methods also count trailing spaces and add back leading space to current segment to maintain proper concatenation
            if self.last_text and should_remove_previous_period(self.last_text, cleaned_text):
                trailing_whitespace_count = get_trailing_whitespace_count(self.last_text)
                await self.backspace(1 + trailing_whitespace_count)
                cleaned_text = " " + cleaned_text

            # Apply capitalization rule: lowercase first letter if no sentence boundary
            # This handles mid-sentence concatenation when current text starts with capital
            if self.last_text and should_lowercase_current_start(self.last_text, cleaned_text):
                cleaned_text = lowercase_first_letter(cleaned_text)

            # Use clipboard or typing based on config
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
        """Paste using clipboard with proper error handling and key repeat prevention"""
        original = None

        try:
            # Save original clipboard content
            try:
                original = pyperclip.paste()
            except (pyperclip.PyperclipException, OSError) as e:
                logger.warning(f"Could not read clipboard: {e}")

            # Copy and paste text
            pyperclip.copy(text)
            time.sleep(self.config.clipboard_paste_delay_pre)

            # CRITICAL: Use explicit key press/release to prevent keyboard repeat
            # hotkey() can sometimes hold keys too long causing Windows autorepeat
            pyautogui.keyDown("ctrl")
            time.sleep(0.01)  # Brief delay to register modifier
            pyautogui.press("v")
            time.sleep(0.01)  # Brief delay before release
            pyautogui.keyUp("ctrl")

            time.sleep(self.config.clipboard_paste_delay_post)

            # Restore original clipboard content
            if original is not None:
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
