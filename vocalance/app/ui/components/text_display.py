from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPalette
from PySide6.QtWidgets import QPlainTextEdit, QSizePolicy, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme


class TransparentTextEdit(QPlainTextEdit):
    """Plain text edit with transparent background and no selection highlighting.

    Features:
    - Transparent background (text only, no highlighting)
    - No border or padding
    - No focus rectangle
    - No selection background color
    - Designed for use inside a dark rounded container
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize transparent text edit widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setReadOnly(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("transparent"))
        palette.setColor(QPalette.ColorRole.Window, QColor("transparent"))
        self.setPalette(palette)

        self.setStyleSheet(
            """
            TransparentTextEdit {{
                background-color: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
                selection-background-color: transparent;
            }}
            TransparentTextEdit:focus {{
                background-color: transparent;
                border: none;
                selection-background-color: transparent;
            }}
        """
        )


class TextDisplayContainer(QWidget):
    """Container with dark rounded background for text display.

    Features:
    - Dark rounded background box with custom painting
    - Contains a TransparentTextEdit
    - 8px padding inside the container
    - Clean, minimal design
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize text display container.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setMinimumHeight(100)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("transparent"))
        palette.setColor(QPalette.ColorRole.Base, QColor("transparent"))
        self.setPalette(palette)

        self._bg_color = QColor(theme.config.shapes.dark)
        self._border_radius = theme.config.radius.medium

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self.text_edit = TransparentTextEdit()
        layout.addWidget(self.text_edit)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        """Paint the rounded background.

        Args:
            paint_event: The paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(self.rect(), self._border_radius, self._border_radius)

        painter.fillPath(path, self._bg_color)

        super().paintEvent(paint_event)
