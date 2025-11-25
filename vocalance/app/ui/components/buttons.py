"""Button component subclasses with inheritance-based styling.

Each button class inherits from QPushButton and applies its own styling.
Primary and Danger buttons use custom paintEvent for complex rendering.
GhostButton uses stylesheet for simple transparent styling.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QPushButton, QWidget

from vocalance.app.ui.qt_theme import theme


class PrimaryButton(QPushButton):
    """Primary action button with blue_1 background and blue_2 text.

    Uses custom paintEvent for consistent pill-shaped rendering.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        command=None,
    ):
        super().__init__(text, parent)

        # Set cursor
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Set font
        self.setFont(theme.get_font("small", "semibold"))

        # Set height
        height = theme.config.components.button_height
        self.setFixedHeight(height)

        # Calculate border radius for pill shape
        self._border_radius = height // 2

        # State tracking
        self._is_hovered = False
        self._is_pressed = False

        # Enable hover tracking
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        # Use flat style - we handle painting ourselves
        self.setFlat(True)
        self.setStyleSheet("")  # Clear any inherited stylesheet

        # Connect command if provided
        if command:
            self.clicked.connect(command)

    def enterEvent(self, event):
        """Handle mouse enter."""
        self._is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave."""
        self._is_hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        self._is_pressed = True
        self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self._is_pressed = False
        self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        """Custom paint for primary button appearance."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self._border_radius, self._border_radius)

        # Determine background color based on state
        if self._is_pressed:
            bg_color = QColor(theme.config.blue.blue_1).darker(120)
        elif self._is_hovered:
            bg_color = QColor(theme.config.blue.blue_1).lighter(120)
        else:
            bg_color = QColor(theme.config.blue.blue_1)

        # Fill background
        painter.fillPath(path, bg_color)

        # Draw text with blue_2 color
        painter.setPen(QColor(theme.config.blue.blue_2))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


class DangerButton(QPushButton):
    """Danger/secondary button with transparent background and blue_1 border.

    Uses custom paintEvent for border rendering.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        command=None,
    ):
        super().__init__(text, parent)

        # Set cursor
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Set font
        self.setFont(theme.get_font("small", "semibold"))

        # Set height
        height = theme.config.components.button_height
        self.setFixedHeight(height)

        # Calculate border radius for pill shape
        self._border_radius = height // 2

        # State tracking
        self._is_hovered = False
        self._is_pressed = False

        # Enable hover tracking
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        # Use flat style - we handle painting ourselves
        self.setFlat(True)
        self.setStyleSheet("")  # Clear any inherited stylesheet

        # Connect command if provided
        if command:
            self.clicked.connect(command)

    def enterEvent(self, event):
        """Handle mouse enter."""
        self._is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave."""
        self._is_hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        self._is_pressed = True
        self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self._is_pressed = False
        self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        """Custom paint for danger button appearance."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create rounded rectangle path for background
        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, self.width(), self.height(), self._border_radius, self._border_radius)

        # Determine fill based on state
        if self._is_pressed:
            painter.fillPath(bg_path, QColor(theme.config.shapes.medium))
        elif self._is_hovered:
            painter.fillPath(bg_path, QColor(theme.config.shapes.light))
        else:
            painter.fillPath(bg_path, Qt.GlobalColor.transparent)

        # Draw 1px border with blue_1 color
        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, self.width() - 1, self.height() - 1, self._border_radius, self._border_radius)

        pen = QPen(QColor(theme.config.blue.blue_1), 1.0)
        painter.setPen(pen)
        painter.drawPath(border_path)

        # Draw text with blue_2 color
        painter.setPen(QColor(theme.config.blue.blue_2))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


class GhostButton(QPushButton):
    """Ghost button with transparent background and light text.

    Uses stylesheet for simple styling with hover effects.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        command=None,
    ):
        super().__init__(text, parent)

        # Set cursor
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Set font
        self.setFont(theme.get_font("small", "semibold"))

        # Set height
        height = theme.config.components.button_height
        self.setFixedHeight(height)

        # Calculate border radius for pill shape
        border_radius = height // 2

        # Apply stylesheet - only for GhostButton
        self.setStyleSheet(
            f"""
            GhostButton {{
                background-color: transparent;
                color: {theme.config.text.light};
                border: none;
                border-radius: {border_radius}px;
                padding: 2px 16px;
            }}
            GhostButton:hover {{
                background-color: {theme.config.shapes.medium};
            }}
            GhostButton:pressed {{
                background-color: {theme.config.shapes.dark};
            }}
            GhostButton:disabled {{
                color: {theme.config.shapes.light};
            }}
        """
        )

        # Connect command if provided
        if command:
            self.clicked.connect(command)
