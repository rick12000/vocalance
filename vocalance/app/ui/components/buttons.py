"""Button component subclasses with inheritance-based styling.

Each button class inherits from QPushButton and applies its own styling.
Primary and Danger buttons use custom paintEvent for complex rendering.
GhostButton uses stylesheet for simple transparent styling.
ChangeButton and DeleteButton are circular icon-based variants.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtSvg import QSvgRenderer
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

        # Set font (use Alata display font for button text)
        self.setFont(theme.get_font("medium", "semibold", display=True))

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
        """Custom paint for primary button appearance with gradient border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        border_width = 1

        # Create gradient for border
        gradient_colors = theme.config.text.gradient_colors
        gradient = QLinearGradient(QPointF(0, 0), QPointF(rect.width(), rect.height()))
        gradient.setColorAt(0, QColor(gradient_colors[0]))
        gradient.setColorAt(1, QColor(gradient_colors[1]))

        # Draw 1px gradient border using pen
        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, rect.width() - 1, rect.height() - 1, self._border_radius, self._border_radius)

        pen = QPen(gradient, border_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.GlobalColor.transparent)
        painter.drawPath(border_path)

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

        # Set font (use Alata display font for button text)
        self.setFont(theme.get_font("medium", "semibold", display=True))

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
            painter.fillPath(bg_path, Qt.GlobalColor.transparent)
        elif self._is_hovered:
            painter.fillPath(bg_path, Qt.GlobalColor.transparent)
        else:
            painter.fillPath(bg_path, Qt.GlobalColor.transparent)

        # Draw 1px border with blue_1 color
        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, self.width() - 1, self.height() - 1, self._border_radius, self._border_radius)

        pen = QPen(QColor(theme.config.shapes.lightest), 1.0)
        painter.setPen(pen)
        painter.drawPath(border_path)

        # Draw text with blue_2 color
        painter.setPen(QColor(theme.config.shapes.accent))
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

        # Set font (use Alata display font for button text)
        self.setFont(theme.get_font("medium", "semibold", display=True))

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


def _create_recolored_renderer(svg_path: str, color: str) -> QSvgRenderer:
    """Load an SVG icon, recolor it, and return a renderer.

    This allows rendering the SVG directly to the painter for maximum crispness
    at any DPI or scale, avoiding rasterization artifacts.

    Args:
        svg_path: Path to the SVG file
        color: Hex color string (e.g., "#a8c7fa")

    Returns:
        QSvgRenderer initialized with recolored SVG data
    """
    from PySide6.QtCore import QByteArray

    # Read SVG file
    svg_file = Path(svg_path)
    if not svg_file.exists():
        return QSvgRenderer()

    svg_content = svg_file.read_text(encoding="utf-8")

    # Replace fill color in SVG
    svg_content = svg_content.replace('fill="#e3e3e3"', f'fill="{color}"')
    svg_content = svg_content.replace('fill="#E3E3E3"', f'fill="{color}"')

    # Create renderer with modified content
    renderer = QSvgRenderer(QByteArray(svg_content.encode("utf-8")))
    return renderer


class ChangeButton(PrimaryButton):
    """Circular icon button with add icon, inherits styling from PrimaryButton.

    Uses the add icon and renders as a perfect circle instead of pill shape.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        command=None,
    ):
        # Initialize with empty text
        super().__init__("", parent, command)

        # Calculate size: same height as primary button (24px)
        button_size = theme.config.components.button_height
        self.setFixedSize(button_size, button_size)

        # CRITICAL: Remove all content margins to ensure proper centering
        self.setContentsMargins(0, 0, 0, 0)

        # Override border radius to make it circular (half of button size)
        self._border_radius = button_size // 2

        # Set up icon
        assets_dir = Path(__file__).parent.parent.parent / "assets" / "icons"
        icon_path = assets_dir / "add_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.svg"

        # Create vector renderer
        self._renderer = _create_recolored_renderer(str(icon_path), theme.config.text.light)

    def paintEvent(self, event):
        """Custom paint for circular button with icon."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Use exact dimensions from self.rect() to match the widget geometry
        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # Create circular path
        path = QPainterPath()
        path.addRoundedRect(0, 0, width, height, self._border_radius, self._border_radius)

        # Determine background color based on state
        if self._is_pressed:
            bg_color = QColor(theme.config.shapes.medium)
        elif self._is_hovered:
            bg_color = QColor(theme.config.shapes.medium)
        else:
            bg_color = QColor(theme.config.shapes.medium)

        # Fill background
        painter.fillPath(path, bg_color)

        # Draw icon centered using vector renderer
        if self._renderer.isValid():
            # Calculate target icon size (60% of button)
            icon_dim = int(width * 0.6)

            # Calculate centered rect
            x_pos = (width - icon_dim) / 2.0
            y_pos = (height - icon_dim) / 2.0

            from PySide6.QtCore import QRectF

            target_rect = QRectF(x_pos, y_pos, icon_dim, icon_dim)

            self._renderer.render(painter, target_rect)


class DeleteButton(DangerButton):
    """Circular icon button with delete icon, inherits styling from DangerButton.

    Uses the delete icon and renders as a perfect circle instead of pill shape.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        command=None,
    ):
        # Initialize with empty text
        super().__init__("", parent, command)

        # Calculate size: same height as danger button (24px)
        button_size = theme.config.components.button_height
        self.setFixedSize(button_size, button_size)

        # CRITICAL: Remove all content margins to ensure proper centering
        self.setContentsMargins(0, 0, 0, 0)

        # Override border radius to make it circular (half of button size)
        self._border_radius = button_size // 2

        # Set up icon
        assets_dir = Path(__file__).parent.parent.parent / "assets" / "icons"
        icon_path = assets_dir / "delete_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.svg"

        # Create vector renderer
        self._renderer = _create_recolored_renderer(str(icon_path), theme.config.shapes.accent)

    def paintEvent(self, event):
        """Custom paint for circular danger button with icon."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Use exact dimensions from self.rect()
        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # Create circular path for background using explicit coordinates
        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, width, height, self._border_radius, self._border_radius)

        # Background is always transparent
        painter.fillPath(bg_path, Qt.GlobalColor.transparent)

        # Draw 1px border with lightest color
        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, width - 1, height - 1, self._border_radius, self._border_radius)

        pen = QPen(QColor(theme.config.shapes.lightest), 1.0)
        painter.setPen(pen)
        painter.drawPath(border_path)

        # Draw icon centered using vector renderer
        if self._renderer.isValid():
            # Calculate target icon size (60% of button)
            icon_dim = int(width * 0.6)

            # Calculate centered rect
            x_pos = (width - icon_dim) / 2.0
            y_pos = (height - icon_dim) / 2.0

            from PySide6.QtCore import QRectF

            target_rect = QRectF(x_pos, y_pos, icon_dim, icon_dim)

            self._renderer.render(painter, target_rect)


class ExpandButton(PrimaryButton):
    """Circular icon button with right arrow icon for expanding sections.

    Uses the keyboard_arrow_right icon and renders as a perfect circle instead of pill shape.
    Icon colored with shapes light, background transparent with shapes light border.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        command=None,
    ):
        # Initialize with empty text
        super().__init__("", parent, command)

        # Calculate size: same height as primary button (24px)
        button_size = theme.config.components.button_height
        self.setFixedSize(button_size, button_size)

        # CRITICAL: Remove all content margins to ensure proper centering
        self.setContentsMargins(0, 0, 0, 0)

        # Override border radius to make it circular (half of button size)
        self._border_radius = button_size // 2

        # Set up icon
        assets_dir = Path(__file__).parent.parent.parent / "assets" / "icons"
        icon_path = assets_dir / "keyboard_arrow_right_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.svg"

        # Create vector renderer with shapes light color
        self._renderer = _create_recolored_renderer(str(icon_path), theme.config.shapes.light)

    def paintEvent(self, event):
        """Custom paint for circular button with icon."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Use exact dimensions from self.rect() to match the widget geometry
        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # Create circular path for background
        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, width, height, self._border_radius, self._border_radius)

        # Fill background with transparent
        painter.fillPath(bg_path, Qt.GlobalColor.transparent)

        # Draw 1px border with shapes light color
        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, width - 1, height - 1, self._border_radius, self._border_radius)

        pen = QPen(QColor(theme.config.shapes.light), 1.0)
        painter.setPen(pen)
        painter.drawPath(border_path)

        # Draw icon centered using vector renderer
        if self._renderer.isValid():
            # Calculate target icon size (60% of button)
            icon_dim = int(width * 0.6)

            # Calculate centered rect
            x_pos = (width - icon_dim) / 2.0
            y_pos = (height - icon_dim) / 2.0

            from PySide6.QtCore import QRectF

            target_rect = QRectF(x_pos, y_pos, icon_dim, icon_dim)

            self._renderer.render(painter, target_rect)


class CollapseButton(PrimaryButton):
    """Circular icon button with down arrow icon for collapsing sections.

    Uses the keyboard_arrow_down icon and renders as a perfect circle instead of pill shape.
    Icon colored with shapes light, background transparent with shapes light border.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        command=None,
    ):
        # Initialize with empty text
        super().__init__("", parent, command)

        # Calculate size: same height as primary button (24px)
        button_size = theme.config.components.button_height
        self.setFixedSize(button_size, button_size)

        # CRITICAL: Remove all content margins to ensure proper centering
        self.setContentsMargins(0, 0, 0, 0)

        # Override border radius to make it circular (half of button size)
        self._border_radius = button_size // 2

        # Set up icon
        assets_dir = Path(__file__).parent.parent.parent / "assets" / "icons"
        icon_path = assets_dir / "keyboard_arrow_down_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.svg"

        # Create vector renderer with shapes light color
        self._renderer = _create_recolored_renderer(str(icon_path), theme.config.shapes.light)

    def paintEvent(self, event):
        """Custom paint for circular button with icon."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Use exact dimensions from self.rect() to match the widget geometry
        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # Create circular path for background
        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, width, height, self._border_radius, self._border_radius)

        # Fill background with transparent
        painter.fillPath(bg_path, Qt.GlobalColor.transparent)

        # Draw 1px border with shapes light color
        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, width - 1, height - 1, self._border_radius, self._border_radius)

        pen = QPen(QColor(theme.config.shapes.light), 1.0)
        painter.setPen(pen)
        painter.drawPath(border_path)

        # Draw icon centered using vector renderer
        if self._renderer.isValid():
            # Calculate target icon size (60% of button)
            icon_dim = int(width * 0.6)

            # Calculate centered rect
            x_pos = (width - icon_dim) / 2.0
            y_pos = (height - icon_dim) / 2.0

            from PySide6.QtCore import QRectF

            target_rect = QRectF(x_pos, y_pos, icon_dim, icon_dim)

            self._renderer.render(painter, target_rect)
