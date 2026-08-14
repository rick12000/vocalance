import asyncio
import logging
import threading
import time

import pyautogui
import pyperclip

from vocalance.app.config.app_config import DictationConfig
from vocalance.app.services.dictation_flow.postprocess.segment_text import clean_dictation_text, should_add_period_before
from vocalance.app.services.keyboard_input_service import KeyboardInputService

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_TIMEOUT_SEC: float = 1.0
CTRL_KEY_PRESS_TIMING_SEC: float = 0.01
CLIPBOARD_READ_MIN_WAIT_SEC: float = 0.05


class DictationTextInput:

    def __init__(
        self,
        config: DictationConfig,
        input_service: KeyboardInputService,
    ) -> None:
        self.config = config
        self.input_service = input_service
        self.clipboard_lock = threading.Lock()
        self.last_text: str | None = None
        self.streaming_buffer: str = ""
        self.buffer_timeout_sec: float = DEFAULT_BUFFER_TIMEOUT_SEC
        self.flush_task: asyncio.Task | None = None
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = config.pyautogui_pause

    def initialize(self) -> bool:
        return True

    def set_buffer_timeout(self, timeout_sec: float) -> None:
        self.buffer_timeout_sec = timeout_sec

    def reset_session(self) -> None:
        with self.clipboard_lock:
            self.last_text = None
        self.streaming_buffer = ""
        self.cancel_flush_task()

    async def queue_streaming_text(self, text: str, add_trailing_space: bool) -> None:
        text = text.strip()
        if not text:
            return
        sep = " " if self.streaming_buffer else ""
        self.streaming_buffer += sep + text
        if len(self.streaming_buffer.split()) >= self.config.streaming_buffer_word_threshold:
            # Do not cancel the pending timeout task — if it has already woken and is
            # inside input_text, cancelling it aborts the space keypress mid-flight.
            # An empty buffer causes it to return cleanly on its next check.
            await self.paste_buffer(add_trailing_space=add_trailing_space)
            return
        if self.flush_task is None or self.flush_task.done():
            self.flush_task = asyncio.ensure_future(
                self.flush_after_timeout(add_trailing_space=add_trailing_space)
            )

    async def flush_streaming_buffer(self, add_trailing_space: bool) -> None:
        self.cancel_flush_task()
        await self.paste_buffer(add_trailing_space=add_trailing_space)

    def cancel_flush_task(self) -> None:
        if self.flush_task and not self.flush_task.done():
            self.flush_task.cancel()
        self.flush_task = None

    async def flush_after_timeout(self, add_trailing_space: bool) -> None:
        await asyncio.sleep(self.buffer_timeout_sec)
        await self.paste_buffer(add_trailing_space=add_trailing_space)

    async def paste_buffer(self, add_trailing_space: bool) -> None:
        if not self.streaming_buffer:
            return
        text = self.streaming_buffer
        self.streaming_buffer = ""
        # Shield ensures paste + space complete even if the outer task is cancelled
        # by a session-end flush call, preventing merged segments from lost space presses.
        await asyncio.shield(self.input_text(text=text, add_trailing_space=add_trailing_space))

    def capture_selection_via_copy(self) -> str:
        with self.clipboard_lock:
            original: str | None = None
            try:
                original = pyperclip.paste()
            except (pyperclip.PyperclipException, OSError) as exc:
                logger.warning("Could not read clipboard before copy: %s", exc)

            time.sleep(self.config.clipboard_paste_delay_pre)
            pyautogui.hotkey("ctrl", "c")
            time.sleep(max(CLIPBOARD_READ_MIN_WAIT_SEC, self.config.clipboard_paste_delay_post))

            captured: str = ""
            try:
                captured = pyperclip.paste() or ""
            except (pyperclip.PyperclipException, OSError) as exc:
                logger.warning("Could not read clipboard after copy: %s", exc)

            if original is not None:
                try:
                    pyperclip.copy(original)
                except (pyperclip.PyperclipException, OSError) as exc:
                    logger.warning("Could not restore clipboard after capture: %s", exc)

            return captured.strip()

    async def input_text(self, text: str, add_trailing_space: bool) -> bool:
        if not text:
            return False

        # Space is pressed as a separate keypress, not embedded in the clipboard payload,
        # because some apps (e.g. Chrome's address bar) silently strip trailing whitespace
        # from clipboard paste events, causing successive segments to merge.
        cleaned_text: str = clean_dictation_text(text=text, add_trailing_space=False)
        if not cleaned_text:
            return False

        if add_trailing_space and self.last_text and should_add_period_before(self.last_text, cleaned_text):
            await self.backspace(count=1)
            await self.input_service.run(pyautogui.write, ". ")

        if self.config.use_clipboard:
            success = await self.input_service.run(self.paste_clipboard, cleaned_text)
        else:
            success = await self.input_service.run(self.type_text, cleaned_text)

        if success:
            self.last_text = cleaned_text
            if add_trailing_space:
                await self.input_service.run(pyautogui.press, "space")

        return success

    def paste_clipboard(self, text: str) -> bool:
        with self.clipboard_lock:
            original: str | None = None
            try:
                original = pyperclip.paste()
            except (pyperclip.PyperclipException, OSError) as exc:
                logger.warning("Could not read clipboard content: %s", exc)

            try:
                pyperclip.copy(text)
            except (pyperclip.PyperclipException, OSError) as exc:
                logger.error("Could not copy text to clipboard: %s", exc, exc_info=True)
                return False

            time.sleep(self.config.clipboard_paste_delay_pre)
            pyautogui.keyDown("ctrl")
            time.sleep(CTRL_KEY_PRESS_TIMING_SEC)
            pyautogui.press("v")
            time.sleep(CTRL_KEY_PRESS_TIMING_SEC)
            pyautogui.keyUp("ctrl")
            time.sleep(self.config.clipboard_paste_delay_post)

            if original is not None:
                try:
                    pyperclip.copy(original)
                except (pyperclip.PyperclipException, OSError) as exc:
                    logger.warning("Could not restore clipboard: %s", exc)

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

    async def backspace(self, count: int) -> bool:
        for _ in range(count):
            await self.input_service.run(pyautogui.press, "backspace")
        return True

    def shutdown(self) -> None:
        pass
