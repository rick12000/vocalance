"""
UI Icon Utilities

Simple icon management for CustomTkinter windows.
"""

import ctypes
import logging
import sys
import tkinter as tk
from pathlib import Path
from typing import Optional, Union

from vocalance.app.config.app_config import AssetPathsConfig

logger = logging.getLogger("IconUtils")

_ICON_PATH: Optional[str] = None
_APP_ID_SET: bool = False
_DPI_AWARENESS_DEACTIVATED: bool = False


def _get_icon_path() -> Optional[str]:
    """Get icon path, cached at module level."""
    global _ICON_PATH
    if _ICON_PATH is None:
        try:
            asset_paths = AssetPathsConfig()
            icon_path_str = asset_paths.icon_path
            if icon_path_str:
                icon_path = Path(icon_path_str)
                if icon_path.exists():
                    _ICON_PATH = str(icon_path.absolute())
                    logger.debug(f"Icon path loaded: {_ICON_PATH}")
        except Exception as e:
            logger.error(f"Failed to load icon path: {e}")
    return _ICON_PATH


def configure_dpi_awareness():
    """
    Configure Windows DPI awareness for icon display.

    For CustomTkinter applications, NO explicit DPI awareness gives the best
    icon quality. CustomTkinter handles its own DPI scaling internally, and
    setting explicit DPI awareness conflicts with iconbitmap() rendering.

    This function intentionally does nothing to allow Windows default behavior.
    """
    global _DPI_AWARENESS_DEACTIVATED
    if _DPI_AWARENESS_DEACTIVATED or sys.platform != "win32":
        return

    # DO NOT set any DPI awareness
    # CustomTkinter + iconbitmap() works best with default Windows DPI handling
    logger.debug("Using default Windows DPI handling for optimal icon quality")
    _DPI_AWARENESS_DEACTIVATED = True


# Backward compatibility alias
deactivate_dpi_awareness = configure_dpi_awareness


def initialize_windows_taskbar_icon():
    """
    Set Windows App User Model ID to show custom taskbar icon immediately.
    Prevents Python's default icon from appearing on taskbar.

    Also notifies Windows to refresh icon cache for this application.
    This should be called early in application startup, after configuring DPI awareness.
    """
    global _APP_ID_SET
    if _APP_ID_SET or sys.platform != "win32":
        return

    try:
        # Changed App ID to force Windows to refresh icon cache
        app_id = "Vocalance.VoiceControl.2.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        logger.debug(f"Windows App ID set: {app_id}")

        # Notify Windows that icon may have changed (forces cache refresh)
        # This helps when icon.ico has been updated
        try:
            # SHChangeNotify with SHCNE_ASSOCCHANGED tells Windows to refresh file associations and icons
            # 0x08000000 = SHCNE_ASSOCCHANGED
            # 0x0000 = SHCNF_IDLIST (not used, but required parameter)
            ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
            logger.debug("Notified Windows to refresh icon cache")
        except Exception as e:
            logger.debug(f"Could not notify icon cache refresh: {e}")

        _APP_ID_SET = True
    except Exception as e:
        logger.debug(f"Could not set App User Model ID: {e}")


def set_window_icon_robust(window: Union[tk.Tk, tk.Toplevel], is_toplevel: bool = None) -> bool:
    """
    Set window icon using direct Windows API calls for proper high-DPI support.

    Bypasses tkinter's iconbitmap() which doesn't handle high-DPI well, and instead
    uses Windows LoadImage API with LR_DEFAULTSIZE flag for proper DPI scaling.
    """
    icon_path = _get_icon_path()
    if not icon_path or sys.platform != "win32":
        logger.error("No icon path available or not Windows")
        return False

    try:
        # Get the window handle
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        if not hwnd:
            hwnd = window.winfo_id()

        # Windows constants
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040  # Use system default size
        LR_SHARED = 0x8000

        ICON_SMALL = 0
        ICON_BIG = 1
        WM_SETICON = 0x0080

        # Load icon with proper flags for DPI scaling
        hicon_small = ctypes.windll.user32.LoadImageW(
            0,  # hInst (0 for loading from file)
            icon_path,  # image path
            IMAGE_ICON,  # image type
            0,  # width (0 with LR_DEFAULTSIZE means system default)
            0,  # height
            LR_LOADFROMFILE | LR_DEFAULTSIZE | LR_SHARED,
        )

        hicon_large = ctypes.windll.user32.LoadImageW(
            0, icon_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE | LR_SHARED  # Let Windows choose size based on DPI
        )

        if hicon_small:
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)

        if hicon_large:
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_large)

        logger.info(f"Icon set via Windows API: {Path(icon_path).name}")
        return True

    except Exception as e:
        logger.error(f"Failed to set icon via Windows API: {e}", exc_info=True)

        # Fallback to standard method
        try:
            window.iconbitmap(icon_path)
            logger.info("Icon set via fallback iconbitmap")
            return True
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return False
