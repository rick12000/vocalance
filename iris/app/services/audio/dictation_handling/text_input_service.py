"""
Streamlined Text Input Service

Simplified text input with clipboard/typing support and minimal complexity.
"""
import asyncio
import logging
import time
import threading
import pyautogui
import pyperclip
import re
from typing import Optional

from iris.app.config.dictation_config import DictationConfig

logger = logging.getLogger(__name__)

class TextInputService:
    """Streamlined text input service with clipboard and typing support"""
    
    def __init__(self, config: DictationConfig):
        self.config = config
        self._lock = threading.RLock()
        self.last_text = None

        # Configure PyAutoGUI safety
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01

        logger.info("TextInputService initialized")
    
    async def initialize(self) -> bool:
        """Initialize service"""
        try:
            logger.info("TextInputService ready")
            return True
        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False
    
    async def input_text(self, text: str) -> bool:
        """Input text at cursor position"""
        if not text:
            return False

        try:
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                return False

            # Check if we need to remove period from previous paste
            if (self.last_text and
                self.last_text.rstrip().endswith('.') and
                cleaned_text.strip() and
                cleaned_text.strip()[0].islower()):
                # Remove the period from previous paste
                await self.backspace(1)

            # Use clipboard or typing based on config
            if self.config.use_clipboard:
                success = await asyncio.get_event_loop().run_in_executor(
                    None, self._paste_clipboard, cleaned_text
                )
            else:
                success = await asyncio.get_event_loop().run_in_executor(
                    None, self._type_text, cleaned_text
                )

            if success:
                logger.debug(f"Input text: '{cleaned_text[:50]}{'...' if len(cleaned_text) > 50 else ''}'")
                # Update last text for next paste
                self.last_text = cleaned_text

            return success

        except Exception as e:
            logger.error(f"Text input error: {e}", exc_info=True)
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean text for input while preserving formatting"""
        if not text:
            return ""

        # Remove "..." and replace with space (handles pause fillers)
        cleaned = re.sub(r'\.\.\.', ' ', text)

        # Ensure every segment gets a trailing space for proper concatenation
        # Unlike the original logic that skipped spaces after certain punctuation,
        # we now ensure EVERY segment ends with a space for continuous dictation
        if cleaned and not cleaned[-1].isspace():
            cleaned = cleaned + ' '

        return cleaned
    
    def _paste_clipboard(self, text: str) -> bool:
        """Paste using clipboard"""
        try:
            # Save original clipboard
            original = None
            try:
                original = pyperclip.paste()
            except:
                pass
            
            # Paste text
            pyperclip.copy(text)
            time.sleep(0.05)
            pyautogui.hotkey('ctrl', 'v') 
            time.sleep(0.1)
            
            # Restore clipboard
            if original is not None:
                try:
                    pyperclip.copy(original)
                except:
                    pass
            
            return True
            
        except Exception as e:
            logger.error(f"Clipboard paste error: {e}", exc_info=True)
            return False
    
    def _type_text(self, text: str) -> bool:
        """Type text character by character"""
        try:
            for char in text:
                pyautogui.write(char, interval=self.config.typing_delay)
            time.sleep(0.1)
            return True
            
        except Exception as e:
            logger.error(f"Text typing error: {e}", exc_info=True)
            return False
    
    async def add_space(self) -> bool:
        """Add space character"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, pyautogui.press, 'space'
            )
            return True
        except Exception as e:
            logger.error(f"Space input error: {e}", exc_info=True)
            return False
    
    async def add_newline(self) -> bool:
        """Add newline character"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, pyautogui.press, 'enter'
            )
            return True
        except Exception as e:
            logger.error(f"Newline input error: {e}", exc_info=True)
            return False
    
    async def backspace(self, count: int = 1) -> bool:
        """Send backspace keystrokes"""
        try:
            for _ in range(count):
                await asyncio.get_event_loop().run_in_executor(
                    None, pyautogui.press, 'backspace'
                )
            return True
        except Exception as e:
            logger.error(f"Backspace error: {e}", exc_info=True)
            return False
    
    async def shutdown(self) -> None:
        """Shutdown service"""
        logger.info("TextInputService shutdown complete")

def create_text_input_service(config: DictationConfig) -> TextInputService:
    """Factory function for text input service"""
    return TextInputService(config) 