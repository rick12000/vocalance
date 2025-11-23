"""Primitive UI components with pure programmatic styling.

All components use theme tokens via programmatic Qt APIs.
NO STYLESHEETS - only QPalette, setFont(), geometry setters, etc.
These are the atomic building blocks for all other components.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette
from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QPushButton, QWidget

from vocalance.app.ui.qt_theme import theme


class PrimitiveLabel(QLabel):
    """Base label with programmatic styling."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        font_size: int = None,
        font_weight: str = "regular",
        color: str = None,
    ):
        super().__init__(text, parent)

        # Set font
        if font_size is None:
            font_size = theme.config.fonts.medium
        self.setFont(theme.get_font(size=font_size, weight=font_weight))

        # Set color via palette
        if color:
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.WindowText, QColor(color))
            self.setPalette(palette)

        # Transparent background
        self.setAutoFillBackground(False)


class PrimitiveButton(QPushButton):
    """Base button with programmatic styling."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        bg_color: str = None,
        text_color: str = None,
        height: int = None,
    ):
        super().__init__(text, parent)

        # Set cursor
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Set font - smaller for button text
        self.setFont(theme.get_font("small", "semibold"))

        # Set height
        if height is None:
            height = theme.config.components.button_height
        self.setFixedHeight(height)

        # Initialize state flags first
        self._is_hovered = False
        self._is_pressed = False

        # Calculate border radius for perfect pill shape (height / 2)
        self._border_radius = height // 2

        # Store colors for state changes
        self._bg_color = bg_color or theme.config.shapes.accent
        self._text_color = text_color or theme.config.shapes.darkest
        self._hover_bg_color = theme.config.shapes.lightest
        self._pressed_bg_color = theme.config.shapes.accent_minus
        self._disabled_bg_color = theme.config.shapes.medium
        self._disabled_text_color = theme.config.shapes.light

        # Enable hover tracking
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        # Set flat style and apply stylesheet for pill-shaped button
        self.setFlat(True)
        self._apply_stylesheet()

    def enterEvent(self, event):
        """Handle mouse enter."""
        self._is_hovered = True
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave."""
        self._is_hovered = False
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        self._is_pressed = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self._is_pressed = False
        super().mouseReleaseEvent(event)

    def changeEvent(self, event):
        """Handle state changes."""
        super().changeEvent(event)

    def _apply_stylesheet(self):
        """Apply stylesheet for rounded corners (pill shape)."""
        bg_color = self._bg_color
        text_color = self._text_color
        hover_bg = self._hover_bg_color
        pressed_bg = self._pressed_bg_color
        disabled_bg = self._disabled_bg_color
        disabled_text = self._disabled_text_color

        stylesheet = f"""
        QPushButton {{
            background-color: {bg_color};
            color: {text_color};
            border: none;
            border-radius: {self._border_radius}px;
            padding: 2px 16px;
            font-weight: bold;
            outline: none;
        }}

        QPushButton:hover {{
            background-color: {hover_bg};
        }}

        QPushButton:pressed {{
            background-color: {pressed_bg};
        }}

        QPushButton:disabled {{
            background-color: {disabled_bg};
            color: {disabled_text};
        }}
        """
        self.setStyleSheet(stylesheet)

    def _update_colors(self):
        """Update button colors based on state and apply stylesheet."""
        # Apply stylesheet which handles all visual states
        self._apply_stylesheet()


class PrimitiveInput(QLineEdit):
    """Base input with pure programmatic styling.

    Uses QPalette and direct property setters only - NO STYLESHEETS.
    Follows the design principle that Primitives use programmatic styling only.
    """

    def __init__(self, placeholder: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setPlaceholderText(placeholder)

        # Set font FIRST before any other styling
        self.setFont(theme.get_font("medium"))

        # Set minimum height
        self.setMinimumHeight(theme.config.components.input_height)

        # Set padding via text margins
        margins = self.textMargins()
        margins.setLeft(theme.config.components.input_padding_horizontal)
        margins.setRight(theme.config.components.input_padding_horizontal)
        margins.setTop(theme.config.components.input_padding_vertical)
        margins.setBottom(theme.config.components.input_padding_vertical)
        self.setTextMargins(margins)

        # Apply colors via palette
        self._update_palette()

    def _update_palette(self):
        """Update colors using QPalette - programmatic only."""
        palette = self.palette()

        palette.setColor(QPalette.ColorRole.Base, QColor(theme.config.shapes.darkest))
        palette.setColor(QPalette.ColorRole.Text, QColor(theme.config.text.light))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.config.text.medium))

        self.setPalette(palette)
        self.setAutoFillBackground(True)


class PrimitiveCheckbox(QCheckBox):
    """Base checkbox with programmatic styling."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)

        self.setFont(theme.get_font("medium"))

        # Set text color
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.light))
        self.setPalette(palette)

        # Store colors for custom painting
        self._indicator_size = 18
        self._indicator_border_color = theme.config.shapes.light
        self._indicator_bg_color = theme.config.shapes.darkest
        self._indicator_checked_bg_color = theme.config.shapes.accent
        self._indicator_checked_border_color = theme.config.shapes.accent
        self._indicator_hover_border_color = theme.config.shapes.lightest
        self._indicator_border_radius = theme.config.radius.small // 2

        # Enable hover tracking
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._is_hovered = False

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

    def paintEvent(self, event):
        """Custom paint for checkbox indicator."""
        # Let QCheckBox handle text
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate indicator position
        indicator_rect = self.style().subElementRect(
            self.style().SubElement.SE_CheckBoxIndicator, self.style().itemOptionFromWidget(self), self
        )

        # Determine colors based on state
        if self.isChecked():
            bg_color = self._indicator_checked_bg_color
            border_color = self._indicator_checked_border_color
        else:
            bg_color = self._indicator_bg_color
            border_color = self._indicator_hover_border_color if self._is_hovered else self._indicator_border_color

        # Draw indicator background
        path = QPainterPath()
        path.addRoundedRect(
            indicator_rect.x(),
            indicator_rect.y(),
            self._indicator_size,
            self._indicator_size,
            self._indicator_border_radius,
            self._indicator_border_radius,
        )
        painter.fillPath(path, QColor(bg_color))

        # Draw border
        painter.setPen(QColor(border_color))
        painter.drawRoundedRect(
            indicator_rect.x(),
            indicator_rect.y(),
            self._indicator_size - 1,
            self._indicator_size - 1,
            self._indicator_border_radius,
            self._indicator_border_radius,
        )
