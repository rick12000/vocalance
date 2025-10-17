import asyncio
import logging
import re
import threading
import time
from typing import Optional

import pyautogui
import pyperclip

from iris.app.config.app_config import DictationConfig

logger = logging.getLogger(__name__)


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

    async def input_text(self, text: str) -> bool:
        if not text:
            return False

        try:
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                return False

            # Check if we need to remove period from previous paste
            if (
                self.last_text
                and self.last_text.rstrip().endswith(".")
                and cleaned_text.strip()
                and cleaned_text.strip()[0].islower()
            ):
                # Remove the period from previous paste
                await self.backspace(1)

            # Use clipboard or typing based on config
            if self.config.use_clipboard:
                success = await asyncio.get_event_loop().run_in_executor(None, self._paste_clipboard, cleaned_text)
            else:
                success = await asyncio.get_event_loop().run_in_executor(None, self._type_text, cleaned_text)

            if success:
                logger.debug(f"Input text: '{cleaned_text[:50]}{'...' if len(cleaned_text) > 50 else ''}'")
                # Update last text for next paste
                self.last_text = cleaned_text

            return success

        except Exception as e:
            logger.error(f"Text input error: {e}", exc_info=True)
            return False

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Remove "..." and replace with space (handles pause fillers)
        cleaned = re.sub(r"\.\.\.", " ", text)

        # Ensure every segment gets a trailing space for proper concatenation
        # Unlike the original logic that skipped spaces after certain punctuation,
        # we now ensure EVERY segment ends with a space for continuous dictation
        if cleaned and not cleaned[-1].isspace():
            cleaned = cleaned + " "

        return cleaned

    def _paste_clipboard(self, text: str) -> bool:
        """Paste using clipboard with proper error handling"""
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
            pyautogui.hotkey("ctrl", "v")
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
