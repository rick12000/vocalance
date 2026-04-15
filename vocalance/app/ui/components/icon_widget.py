from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget


class IconWidget(QWidget):
    """Widget that renders an icon pixmap with high-quality scaling.

    Used to replace QLabel for icons to ensure strict size constraints
    and high-quality rendering regardless of source pixmap DPI.
    """

    def __init__(self, pixmap: Optional[QPixmap], size: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self.setFixedSize(size, size)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Update the displayed pixmap."""
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        """Paint the pixmap scaled to the widget's fixed size."""
        if not self._pixmap or self._pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(self.rect(), self._pixmap)
        painter.end()
