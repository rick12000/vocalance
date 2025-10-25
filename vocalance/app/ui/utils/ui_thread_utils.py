"""
UI Thread Utilities

Thread-safe utilities for scheduling UI updates and managing UI operations across threads.
Single responsibility: Thread-safe UI operation scheduling.

Thread Safety:
- UIScheduler must be initialized in the main tkinter thread
- schedule_ui_update() can be called from any thread
- All UI operations are marshalled to main thread via root.after_idle()
"""

import logging
import threading
import tkinter as tk
from typing import Callable, Optional, Union

import customtkinter as ctk

logger = logging.getLogger("UIThreadUtils")


class UIScheduler:
    """
    Thread-safe UI operation scheduler.

    MUST be initialized in the main tkinter thread.
    All methods are thread-safe and can be called from any thread.
    """

    def __init__(self, root_window: Union[tk.Tk, ctk.CTk]):
        self.root_window = root_window
        self._main_thread_id = threading.get_ident()
        logger.info(f"UIScheduler initialized in thread {self._main_thread_id}")

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


# Global scheduler instance with thread-safe initialization
_ui_scheduler: Optional[UIScheduler] = None
_ui_scheduler_lock = threading.Lock()


def initialize_ui_scheduler(root_window: Union[tk.Tk, ctk.CTk]) -> UIScheduler:
    """
    Initialize the global UI scheduler.

    MUST be called from the main tkinter thread before any UI operations.
    Thread-safe: Can be called multiple times, but only first call takes effect.
    """
    global _ui_scheduler
    with _ui_scheduler_lock:
        if _ui_scheduler is None:
            _ui_scheduler = UIScheduler(root_window)
            logger.info("UI scheduler initialized")
        else:
            logger.warning("UI scheduler already initialized, ignoring duplicate initialization")
        return _ui_scheduler


def get_ui_scheduler() -> Optional[UIScheduler]:
    """
    Get the global UI scheduler instance.

    Thread-safe: Can be called from any thread.
    Returns None if scheduler not yet initialized.
    """
    with _ui_scheduler_lock:
        return _ui_scheduler


def schedule_ui_update(callback: Callable, *args) -> None:
    """
    Convenience function to schedule a UI update.

    Thread-safe: Can be called from any thread.
    Schedules callback to run in main tkinter thread.
    """
    scheduler = get_ui_scheduler()
    if scheduler:
        scheduler.schedule_ui_update(callback, *args)
    else:
        logger.warning("UI scheduler not initialized - cannot schedule UI update")


def schedule_ui_update_immediate(callback: Callable, *args) -> None:
    """
    Convenience function to execute UI update immediately or schedule.

    Thread-safe: Can be called from any thread.
    Executes immediately if in main thread, otherwise schedules.
    """
    scheduler = get_ui_scheduler()
    if scheduler:
        scheduler.schedule_ui_update_immediate(callback, *args)
    else:
        logger.warning("UI scheduler not initialized - cannot schedule UI update")


def is_main_thread() -> bool:
    """
    Convenience function to check if we're in the main thread.

    Thread-safe: Can be called from any thread.
    """
    scheduler = get_ui_scheduler()
    if scheduler:
        return scheduler.is_main_thread()
    return threading.get_ident() == threading.main_thread().ident


def is_ui_available() -> bool:
    """
    Convenience function to check if UI updates can be scheduled.

    Thread-safe: Can be called from any thread.
    """
    scheduler = get_ui_scheduler()
    if scheduler:
        return scheduler.is_ui_available()
    return False
