import logging
import threading
import time

import pyautogui
import pyperclip

from vocalance.app.config.app_config import DictationConfig
from vocalance.app.services.dictation_flow.postprocess.segment_text import (
    clean_dictation_text,
    get_trailing_whitespace_count,
    lowercase_first_letter,
    should_lowercase_current_start,
    should_remove_previous_period,
)
from vocalance.app.services.keyboard_input_service import KeyboardInputService

logger = logging.getLogger(__name__)


class DictationTextInput:
    """Injects dictation text via clipboard or keyboard."""

    def __init__(
        self,
        config: DictationConfig,
        input_service: KeyboardInputService,
    ) -> None:
        self.config = config
        self.input_service = input_service
        self.clipboard_lock = threading.Lock()
        self.last_text: str | None = None
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = config.pyautogui_pause

    def initialize(self) -> bool:
        return True

    def reset_session(self) -> None:
        with self.clipboard_lock:
            self.last_text = None

    def capture_selection_via_copy(self) -> str:
        with self.clipboard_lock:
            original: str | None = None
            original_read_ok = False
            try:
                original = pyperclip.paste()
                original_read_ok = True
            except (pyperclip.PyperclipException, OSError) as e:
                logger.warning("Could not read clipboard before copy: %s", e)

            time.sleep(self.config.clipboard_paste_delay_pre)
            pyautogui.hotkey("ctrl", "c")
            time.sleep(max(0.05, self.config.clipboard_paste_delay_post))

            captured: str = ""
            try:
                captured = pyperclip.paste() or ""
            except (pyperclip.PyperclipException, OSError) as e:
                logger.warning("Could not read clipboard after copy: %s", e)

            if original_read_ok and original is not None:
                try:
                    pyperclip.copy(original)
                except (pyperclip.PyperclipException, OSError) as e:
                    logger.warning("Could not restore clipboard after capture: %s", e)

            return captured.strip()

    async def input_text(
        self,
        text: str,
        add_trailing_space: bool = True,
        skip_prose_segment_join_rules: bool = False,
    ) -> bool:
        if not text:
            return False

        cleaned_text: str = clean_dictation_text(text=text, add_trailing_space=add_trailing_space)
        if not cleaned_text:
            return False

        if not skip_prose_segment_join_rules:
            if self.last_text and should_remove_previous_period(self.last_text, cleaned_text):
                trailing_whitespace_count: int = get_trailing_whitespace_count(self.last_text)
                await self.backspace(1 + trailing_whitespace_count)
                cleaned_text = " " + cleaned_text

            if self.last_text and should_lowercase_current_start(self.last_text, cleaned_text):
                cleaned_text = lowercase_first_letter(cleaned_text)

        if self.config.use_clipboard:
            success: bool = await self.input_service.run(self.paste_clipboard, cleaned_text)
        else:
            success = await self.input_service.run(self.type_text, cleaned_text)

        if success:
            self.last_text = cleaned_text

        return success

    def paste_clipboard(self, text: str) -> bool:
        with self.clipboard_lock:
            original: str | None = None
            original_read_success = False
            try:
                original = pyperclip.paste()
                original_read_success = True
            except (pyperclip.PyperclipException, OSError) as e:
                logger.warning("Could not read original clipboard content: %s", e)
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
                            "Could not copy text to clipboard after %s attempts: %s",
                            max_copy_attempts,
                            e,
                            exc_info=True,
                        )
                        return False
                    time.sleep(0.1)

            verify_attempts = 0
            max_verify_attempts = 2
            while verify_attempts < max_verify_attempts:
                try:
                    clipboard_content: str = pyperclip.paste()
                    if clipboard_content == text:
                        break
                    verify_attempts += 1
                    if verify_attempts >= max_verify_attempts:
                        logger.error(
                            "Clipboard content mismatch after %s attempts! Expected '%s' but got '%s'",
                            max_verify_attempts,
                            text[:50],
                            clipboard_content[:50],
                        )
                        return False
                    time.sleep(0.01)
                except (pyperclip.PyperclipException, OSError) as e:
                    verify_attempts += 1
                    if verify_attempts >= max_verify_attempts:
                        logger.warning(
                            "Could not verify clipboard content after %s attempts: %s",
                            max_verify_attempts,
                            e,
                        )
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
                    logger.warning("Could not restore clipboard: %s", e)

            return True

    def type_text(self, text: str) -> bool:
        for char in text:
            pyautogui.write(char, interval=self.config.typing_delay)
        time.sleep(self.config.type_text_post_delay)
        return True

    async def add_space(self) -> bool:
        await self.input_service.run(pyautogui.press, "space")
        return True

    async def add_newline(self) -> bool:
        await self.input_service.run(pyautogui.press, "enter")
        return True

    async def backspace(self, count: int = 1) -> bool:
        for _ in range(count):
            await self.input_service.run(pyautogui.press, "backspace")
        return True

    def shutdown(self) -> None:
        pass
