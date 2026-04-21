import logging
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QWidget

logger = logging.getLogger(__name__)


class WindowIconManager:
    """Caches one ``QIcon`` and applies it to the app shell and dialogs."""

    def __init__(self, icon_path: Optional[Path] = None) -> None:
        self.icon_path = icon_path
        self._window_icon: Optional[QIcon] = None
        self._has_cached_icon: bool = False

    def load_icon(self) -> bool:
        """Load ``icon_path`` into an internal ``QIcon`` cache."""
        if not self.icon_path or not self.icon_path.exists():
            logger.warning("Icon file not found: %s", self.icon_path)
            self._has_cached_icon = False
            return False

        try:
            self._window_icon = QIcon(str(self.icon_path))

            if self._window_icon.isNull():
                logger.warning("Icon invalid or empty: %s", self.icon_path)
                self._window_icon = None
                self._has_cached_icon = False
                return False

            logger.debug("Icon loaded: %s", self.icon_path)
            self._has_cached_icon = True
            return True

        except OSError as exc:
            logger.error("Failed to load icon %s: %s", self.icon_path, exc)
            self._has_cached_icon = False
            return False

    def apply_to_application(self, qt_app: QApplication) -> bool:
        """Set ``qt_app`` window icon from the cache."""
        if not self._window_icon:
            logger.debug("No icon cached; skip QApplication icon")
            return False

        try:
            qt_app.setWindowIcon(self._window_icon)
            logger.debug("Icon applied to QApplication")
            return True
        except RuntimeError as exc:
            logger.error("Failed to apply icon to QApplication: %s", exc)
            return False

    def apply_to_window(self, window: QMainWindow) -> bool:
        """Set the main window icon from the cache."""
        if not self._window_icon:
            logger.debug("No icon cached; skip QMainWindow icon")
            return False

        try:
            window.setWindowIcon(self._window_icon)
            return True
        except RuntimeError as exc:
            logger.error("Failed to apply icon to QMainWindow: %s", exc)
            return False

    def apply_to_dialog(self, dialog: QDialog) -> bool:
        """Set a dialog window icon from the cache."""
        if not self._window_icon:
            logger.debug("No icon cached; skip QDialog icon")
            return False

        try:
            dialog.setWindowIcon(self._window_icon)
            return True
        except RuntimeError as exc:
            logger.error("Failed to apply icon to QDialog: %s", exc)
            return False

    def apply_to_widget(self, widget: QWidget) -> bool:
        """Set ``widget`` window icon from the cache."""
        if not self._window_icon:
            logger.debug("No icon cached; skip QWidget icon")
            return False

        try:
            widget.setWindowIcon(self._window_icon)
            return True
        except RuntimeError as exc:
            logger.error("Failed to apply icon to QWidget: %s", exc)
            return False

    def is_icon_loaded(self) -> bool:
        """True when a non-null icon is ready to apply."""
        return self._has_cached_icon and self._window_icon is not None and not self._window_icon.isNull()

    def get_icon(self) -> Optional[QIcon]:
        """Return the cached icon, if any."""
        return self._window_icon if self._has_cached_icon else None
