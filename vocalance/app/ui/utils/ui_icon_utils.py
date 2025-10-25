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


def initialize_windows_taskbar_icon():
    """
    Set Windows App User Model ID to show custom taskbar icon immediately.
    Prevents Python's default icon from appearing on taskbar.
    """
    global _APP_ID_SET
    if _APP_ID_SET or sys.platform != "win32":
        return

    try:
        app_id = "Vocalance.VoiceControl.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        logger.debug(f"Windows App ID set: {app_id}")
        _APP_ID_SET = True
    except Exception as e:
        logger.debug(f"Could not set App User Model ID: {e}")


def set_window_icon_robust(window: Union[tk.Tk, tk.Toplevel], is_toplevel: bool = None) -> bool:
    """
    Set window icon for CustomTkinter windows.

    CustomTkinter overrides icons at various timing intervals, so we set
    the icon immediately and then reinforce multiple times to ensure it sticks.
    """
    icon_path = _get_icon_path()
    if not icon_path:
        return False

    def set_icon_now():
        try:
            window.iconbitmap(icon_path)
        except Exception as e:
            logger.debug(f"Icon set failed: {e}")

    try:
        set_icon_now()
        window.after_idle(set_icon_now)
        for delay in [10, 50, 100, 200]:
            window.after(delay, set_icon_now)
        return True
    except Exception as e:
        logger.error(f"Failed to set icon: {e}")
        return False
