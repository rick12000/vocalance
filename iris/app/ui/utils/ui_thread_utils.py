"""
UI Thread Utilities

Thread-safe utilities for scheduling UI updates and managing UI operations across threads.
Single responsibility: Thread-safe UI operation scheduling.
"""

import tkinter as tk
import customtkinter as ctk
import threading
import logging
from typing import Callable, Any, Optional, Union

logger = logging.getLogger("UIThreadUtils")

class UIScheduler:
    """Thread-safe UI operation scheduler."""
    
    def __init__(self, root_window: Union[tk.Tk, ctk.CTk]):
        self.root_window = root_window
        self._main_thread_id = threading.get_ident()
    
    def is_main_thread(self) -> bool:
        """Check if we're running in the main thread."""
        return threading.get_ident() == self._main_thread_id
    
    def is_ui_available(self) -> bool:
        """Check if UI updates can be scheduled."""
        if not self.root_window:
            return False
        try:
            # Try a simple test to see if the main loop is running
            self.root_window.after_idle(lambda: None)
            return True
        except RuntimeError as e:
            if "main thread is not in main loop" in str(e):
                return False
            return True  # Other errors might be temporary
        except Exception:
            return False
    
    def schedule_ui_update(self, callback: Callable, *args) -> None:
        """Schedule a UI update in the main thread - thread-safe."""
        if not self.root_window:
            return
            
        def safe_wrapper():
            """Wrapper that catches exceptions during callback execution"""
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Error in UI update callback: {e}", exc_info=True)
        
        try:
            # Use after_idle for better responsiveness - processes when Tk is idle
            self.root_window.after_idle(safe_wrapper)
        except RuntimeError as e:
            if "main thread is not in main loop" in str(e) and self.is_main_thread():
                # Execute immediately if we're in the main thread but loop isn't running
                try:
                    callback(*args)
                except Exception:
                    pass  # Silently ignore errors during shutdown
        except (tk.TclError, Exception):
            pass  # Silently ignore UI errors during shutdown
    
    def schedule_ui_update_immediate(self, callback: Callable, *args) -> None:
        """Execute UI update immediately if in main thread, otherwise schedule."""
        if self.is_main_thread():
            try:
                callback(*args)
                if self.root_window:
                    try:
                        self.root_window.update_idletasks()
                    except RuntimeError as e:
                        if "main thread is not in main loop" in str(e):
                            logger.debug("Cannot update idletasks: main thread is not in main loop")
                        else:
                            logger.warning(f"Error updating idletasks: {e}")
            except Exception as e:
                logger.error(f"Error in immediate UI update: {e}", exc_info=True)
        else:
            self.schedule_ui_update(callback, *args)

# Global scheduler instance
_ui_scheduler: Optional[UIScheduler] = None

def initialize_ui_scheduler(root_window: Union[tk.Tk, ctk.CTk]) -> UIScheduler:
    """Initialize the global UI scheduler."""
    global _ui_scheduler
    _ui_scheduler = UIScheduler(root_window)
    logger.info("UI scheduler initialized")
    return _ui_scheduler

def get_ui_scheduler() -> Optional[UIScheduler]:
    """Get the global UI scheduler instance."""
    return _ui_scheduler

def schedule_ui_update(callback: Callable, *args) -> None:
    """Convenience function to schedule a UI update."""
    if _ui_scheduler:
        _ui_scheduler.schedule_ui_update(callback, *args)
    else:
        logger.warning("UI scheduler not initialized")

def schedule_ui_update_immediate(callback: Callable, *args) -> None:
    """Convenience function to execute UI update immediately or schedule."""
    if _ui_scheduler:
        _ui_scheduler.schedule_ui_update_immediate(callback, *args)
    else:
        logger.warning("UI scheduler not initialized")

def is_main_thread() -> bool:
    """Convenience function to check if we're in the main thread."""
    if _ui_scheduler:
        return _ui_scheduler.is_main_thread()
    return threading.get_ident() == threading.main_thread().ident 

def is_ui_available() -> bool:
    """Convenience function to check if UI updates can be scheduled."""
    if _ui_scheduler:
        return _ui_scheduler.is_ui_available()
    return False 