from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QWidget


class VocalanceWidget(QWidget):
    """Base QWidget with consistent auto-fill and stylesheet inheritance policy."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def apply_role_palette(self, palette: QPalette) -> None:
        """Apply a fully-built palette (e.g. from ``theme.get_palette``)."""
        self.setPalette(palette)
        self.setAutoFillBackground(True)
