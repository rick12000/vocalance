from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath
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

    def __init__(self, parent: QWidget = None):
        """Initialize transparent text edit widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setReadOnly(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Use palette to ensure truly transparent background
        from PySide6.QtGui import QPalette

        palette = self.palette()
        # Set Base color to transparent directly - don't inherit from parent which might be opaque/medium colored
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

    def __init__(self, parent: QWidget = None):
        """Initialize text display container.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Set minimum size to ensure visibility
        self.setMinimumHeight(100)

        # Set size policy to expand and fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Make background truly transparent - all styling via paintEvent
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        # Clear palette colors to ensure transparency
        from PySide6.QtGui import QPalette

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("transparent"))
        palette.setColor(QPalette.ColorRole.Base, QColor("transparent"))
        self.setPalette(palette)

        # Store the background color
        self._bg_color = QColor(theme.config.shapes.dark)
        self._border_radius = theme.config.radius.medium

        # Create layout with proper margins
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        # Create text edit widget
        self.text_edit = TransparentTextEdit()
        layout.addWidget(self.text_edit)

    def paintEvent(self, event) -> None:
        """Paint the rounded background.

        Args:
            event: The paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(self.rect(), self._border_radius, self._border_radius)

        # Fill with background color
        painter.fillPath(path, self._bg_color)

        # Call parent paintEvent to draw children
        super().paintEvent(event)
