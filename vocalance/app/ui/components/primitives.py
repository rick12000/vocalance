"""Primitive UI components with pure programmatic styling.

All components use theme tokens via programmatic Qt APIs.
NO STYLESHEETS - only QPalette, setFont(), geometry setters, etc.
These are the atomic building blocks for all other components.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPalette, QPen, QRadialGradient
from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QPushButton, QWidget

from vocalance.app.ui.qt_theme import theme


def _create_gradient(width: int, colors: list) -> QLinearGradient:
    """Create smooth linear gradient with blended transitions."""
    gradient = QLinearGradient(0, 0, width, 0)
    if not colors:
        return gradient

    num = len(colors)
    if num == 1:
        gradient.setColorAt(0, QColor(colors[0]))
        return gradient

    # For 2+ colors, add intermediate stops for smoother blending
    # Each color gets placed and repeated at close intervals for smooth transition
    stops_per_color = 5  # More stops = smoother blend

    for i, color in enumerate(colors):
        pos = i / (num - 1)

        # Add main color stop
        gradient.setColorAt(pos, QColor(color))

        # Add intermediate stops for blend smoothing between colors
        if i < num - 1:
            next_color = QColor(colors[i + 1])
            current = QColor(color)

            # Create blend steps between this color and next
            for step in range(1, stops_per_color):
                blend_pos = pos + (1 / (num - 1)) * (step / stops_per_color)
                blend_ratio = step / stops_per_color

                # Interpolate between current and next color
                blended = QColor(
                    int(current.red() * (1 - blend_ratio) + next_color.red() * blend_ratio),
                    int(current.green() * (1 - blend_ratio) + next_color.green() * blend_ratio),
                    int(current.blue() * (1 - blend_ratio) + next_color.blue() * blend_ratio),
                )
                gradient.setColorAt(blend_pos, blended)

    return gradient


def _create_radial_gradient(width: int, height: int, colors: list) -> QRadialGradient:
    """Create smooth radial gradient from bottom-left corner with glow effect."""
    # Center at bottom-left, radius extends to cover entire button
    center_x = 0
    center_y = height
    radius = (width**2 + height**2) ** 0.5  # Diagonal distance covers button

    gradient = QRadialGradient(center_x, center_y, radius)
    if not colors:
        return gradient

    num = len(colors)
    if num == 1:
        gradient.setColorAt(0, QColor(colors[0]))
        return gradient

    # Add smooth blended stops
    stops_per_color = 5

    for i, color in enumerate(colors):
        pos = i / (num - 1)

        # Add main color stop
        gradient.setColorAt(pos, QColor(color))

        # Add intermediate stops for blend smoothing
        if i < num - 1:
            next_color = QColor(colors[i + 1])
            current = QColor(color)

            # Create blend steps
            for step in range(1, stops_per_color):
                blend_pos = pos + (1 / (num - 1)) * (step / stops_per_color)
                blend_ratio = step / stops_per_color

                blended = QColor(
                    int(current.red() * (1 - blend_ratio) + next_color.red() * blend_ratio),
                    int(current.green() * (1 - blend_ratio) + next_color.green() * blend_ratio),
                    int(current.blue() * (1 - blend_ratio) + next_color.blue() * blend_ratio),
                )
                gradient.setColorAt(blend_pos, blended)

    return gradient


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
        is_primary: bool = False,
        is_danger: bool = False,
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
        self._is_primary = is_primary
        self._is_danger = is_danger

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

        # If primary or danger button, use custom painting instead of stylesheet
        if self._is_primary or self._is_danger:
            self.setStyleSheet("")  # Clear stylesheet for custom painting
        else:
            self._apply_stylesheet()

    def enterEvent(self, event):
        """Handle mouse enter."""
        self._is_hovered = True
        if self._is_primary or self._is_danger:
            self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave."""
        self._is_hovered = False
        if self._is_primary or self._is_danger:
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        self._is_pressed = True
        if self._is_primary or self._is_danger:
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self._is_pressed = False
        if self._is_primary or self._is_danger:
            self.update()
        super().mouseReleaseEvent(event)

    def changeEvent(self, event):
        """Handle state changes."""
        super().changeEvent(event)

    def _apply_stylesheet(self):
        """Apply stylesheet for rounded corners (pill shape).

        Uses PrimitiveButton selector to avoid affecting other QPushButton subclasses.
        """
        bg_color = self._bg_color
        text_color = self._text_color
        hover_bg = self._hover_bg_color
        pressed_bg = self._pressed_bg_color
        disabled_bg = self._disabled_bg_color
        disabled_text = self._disabled_text_color

        stylesheet = f"""
        PrimitiveButton {{
            background-color: {bg_color};
            color: {text_color};
            border: none;
            border-radius: {self._border_radius}px;
            padding: 2px 16px;
            font-weight: bold;
            outline: none;
        }}

        PrimitiveButton:hover {{
            background-color: {hover_bg};
        }}

        PrimitiveButton:pressed {{
            background-color: {pressed_bg};
        }}

        PrimitiveButton:disabled {{
            background-color: {disabled_bg};
            color: {disabled_text};
        }}
        """
        self.setStyleSheet(stylesheet)

    def _update_colors(self):
        """Update button colors based on state and apply stylesheet."""
        # Apply stylesheet which handles all visual states
        self._apply_stylesheet()

    def paintEvent(self, event):
        """Custom paint event for primary and danger buttons."""
        if self._is_primary:
            # Primary button: solid blue background, no border
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Create rounded rectangle path
            path = QPainterPath()
            path.addRoundedRect(0, 0, self.width(), self.height(), self._border_radius, self._border_radius)

            # Fill with blue_1 color
            painter.fillPath(path, QColor(theme.config.blue.blue_1))

            # Draw text with blue_2 color
            painter.setPen(QColor(theme.config.blue.blue_2))
            painter.setFont(self.font())
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())

        elif self._is_danger:
            # Danger button: transparent background with blue_1 border
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Draw transparent background
            bg_path = QPainterPath()
            bg_path.addRoundedRect(0, 0, self.width(), self.height(), self._border_radius, self._border_radius)
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

        else:
            # For other buttons, use default painting
            super().paintEvent(event)


class PrimitiveInput(QLineEdit):
    """Base input with programmatic styling.

    Uses QPalette, direct property setters, and focused stylesheet for the input element itself.
    Stylesheet targets only this specific custom component to avoid broad selector issues.
    """

    def __init__(self, placeholder: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setPlaceholderText(placeholder)

        # Set font to small size
        self.setFont(theme.get_font("small"))

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

        # Apply focused stylesheet for border and border radius
        self._apply_stylesheet()

    def _update_palette(self):
        """Update colors using QPalette - programmatic only."""
        palette = self.palette()

        palette.setColor(QPalette.ColorRole.Base, QColor(theme.config.shapes.darkest))
        palette.setColor(QPalette.ColorRole.Text, QColor(theme.config.text.light))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.config.text.medium))

        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def _apply_stylesheet(self):
        """Apply stylesheet targeting ONLY this specific component class.

        Uses specific selector to avoid affecting parent or sibling classes.
        IMPORTANT: Must explicitly set all visual properties when using stylesheets,
        as stylesheets override the entire style system. Unset properties fall back
        to palette colors which may become visible in unexpected ways.
        """
        border_color = theme.config.shapes.light
        border_radius = theme.config.radius.small

        # Focused stylesheet with specific component selector
        # NOTE: background-color MUST be explicit to override palette behavior
        stylesheet = f"""
        PrimitiveInput {{
            background-color: transparent;
            border: 1px solid {border_color};
            border-radius: {border_radius}px;
            padding: 0px;
        }}
        """
        self.setStyleSheet(stylesheet)


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
