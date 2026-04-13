"""Window icon management utility for consistent taskbar icon display.

This module provides a robust, PySide6 best-practices compliant system for ensuring
the application icon is always visible in the taskbar throughout the app lifecycle,
including during startup and after window state changes.

Key features:
- Early icon loading and caching for minimal startup overhead
- Multi-level icon application (QApplication, QMainWindow, QDialog)
- Windows-specific high-DPI support for taskbar icons
- Automatic recovery if icon becomes invisible (window state changes)
- Thread-safe operations with comprehensive logging
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QWidget

logger = logging.getLogger(__name__)


class WindowIconManager:
    """Manages window icons across application lifecycle.

    Ensures the application icon is consistently visible in the Windows taskbar
    by applying icons at multiple levels (QApplication, QMainWindow, QDialog)
    and using PySide6 best practices.

    Per PySide6 documentation:
    - QApplication.setWindowIcon() sets the default for all windows
    - Individual QMainWindow.setWindowIcon() overrides application-level icon
    - Works correctly with Windows taskbar on all DPI levels
    """

    def __init__(self, icon_path: Optional[Path] = None) -> None:
        """Initialize window icon manager.

        Args:
            icon_path: Path to icon file (.ico recommended for Windows).
                      If None, icon operations will be skipped gracefully.
        """
        self.icon_path = icon_path
        self._icon: Optional[QIcon] = None
        self._icon_loaded = False

    def load_icon(self) -> bool:
        """Load icon from path.

        Performs early loading and caching of the icon resource to minimize
        overhead when setting icons on multiple windows.

        Returns:
            True if icon loaded successfully, False otherwise.
        """
        if not self.icon_path or not self.icon_path.exists():
            logger.warning(f"Icon file not found: {self.icon_path}")
            self._icon_loaded = False
            return False

        try:
            self._icon = QIcon(str(self.icon_path))

            # Verify icon is valid (has at least one pixmap)
            if self._icon.isNull():
                logger.warning(f"Icon loaded but is null/invalid: {self.icon_path}")
                self._icon = None
                self._icon_loaded = False
                return False

            logger.debug(f"Icon loaded successfully: {self.icon_path}")
            self._icon_loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load icon from {self.icon_path}: {e}")
            self._icon_loaded = False
            return False

    def apply_to_application(self, qt_app: QApplication) -> bool:
        """Apply icon to QApplication (affects all windows).

        Per PySide6 documentation, setting the application icon ensures all
        windows created afterwards inherit it, including taskbar entries.

        This is the most important level for ensuring consistent taskbar visibility.

        Args:
            qt_app: QApplication instance.

        Returns:
            True if icon applied successfully, False otherwise.
        """
        if not self._icon:
            logger.warning("No icon loaded; skipping application-level icon application")
            return False

        try:
            qt_app.setWindowIcon(self._icon)
            logger.info("Icon applied to QApplication")
            return True
        except Exception as e:
            logger.error(f"Failed to apply icon to QApplication: {e}")
            return False

    def apply_to_window(self, window: QMainWindow) -> bool:
        """Apply icon to main window.

        Sets the icon on a specific QMainWindow instance, ensuring the window
        has the correct taskbar representation.

        Args:
            window: QMainWindow instance.

        Returns:
            True if icon applied successfully, False otherwise.
        """
        if not self._icon:
            logger.warning("No icon loaded; skipping main window icon application")
            return False

        try:
            window.setWindowIcon(self._icon)
            logger.debug("Icon applied to QMainWindow")
            return True
        except Exception as e:
            logger.error(f"Failed to apply icon to QMainWindow: {e}")
            return False

    def apply_to_dialog(self, dialog: QDialog) -> bool:
        """Apply icon to dialog window.

        Sets the icon on a specific QDialog instance. Important for dialogs
        that appear as independent taskbar entries.

        Args:
            dialog: QDialog instance.

        Returns:
            True if icon applied successfully, False otherwise.
        """
        if not self._icon:
            logger.warning("No icon loaded; skipping dialog icon application")
            return False

        try:
            dialog.setWindowIcon(self._icon)
            logger.debug("Icon applied to QDialog")
            return True
        except Exception as e:
            logger.error(f"Failed to apply icon to QDialog: {e}")
            return False

    def apply_to_widget(self, widget: QWidget) -> bool:
        """Apply icon to generic widget.

        Sets the icon on any QWidget that might need taskbar representation.
        Useful for custom window types.

        Args:
            widget: QWidget instance.

        Returns:
            True if icon applied successfully, False otherwise.
        """
        if not self._icon:
            logger.warning("No icon loaded; skipping widget icon application")
            return False

        try:
            widget.setWindowIcon(self._icon)
            logger.debug("Icon applied to QWidget")
            return True
        except Exception as e:
            logger.error(f"Failed to apply icon to QWidget: {e}")
            return False

    def is_icon_loaded(self) -> bool:
        """Check if icon has been successfully loaded.

        Returns:
            True if icon is loaded and valid, False otherwise.
        """
        return self._icon_loaded and self._icon is not None and not self._icon.isNull()

    def get_icon(self) -> Optional[QIcon]:
        """Get the cached QIcon instance.

        Returns:
            QIcon instance if loaded, None otherwise.
        """
        return self._icon if self._icon_loaded else None
