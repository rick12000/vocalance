import logging
import tkinter as tk

logger = logging.getLogger(__name__)


def configure_dpi_awareness() -> None:
    """Configure DPI awareness for Windows."""
    try:
        import ctypes

        # Set DPI awareness for Windows
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        # Not on Windows or DPI awareness not supported
        pass


def initialize_windows_taskbar_icon() -> None:
    """Initialize Windows taskbar icon."""
    try:
        import ctypes

        # Set app user model ID for Windows taskbar
        myappid = "vocalance.app.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except (AttributeError, OSError):
        # Not on Windows or taskbar icon not supported
        pass


def set_window_icon_robust(window: tk.Tk) -> None:
    """Set window icon in a robust manner that handles various edge cases.

    Args:
        window: Tkinter window to set icon for
    """
    try:
        # Try to set icon from file if available
        # This is a placeholder - actual icon setting would depend on available assets
        pass
    except Exception as e:
        logger.debug(f"Could not set window icon: {e}")
