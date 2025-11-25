"""Complex UI components built from new component subclasses.

All components use systematic spacing from theme.
Uses new label, button, and input subclasses.
"""

from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BodyLabel, GroupHeaderLabel, SmallLabel
from vocalance.app.ui.components.layouts import BaseContainer
from vocalance.app.ui.qt_theme import theme


class FormGroup(QWidget):
    """Container for a label and an input field.

    Note: Consider using FormField from layouts.py instead for new code.
    """

    def __init__(
        self,
        label: str,
        input_widget: QWidget,
        parent: Optional[QWidget] = None,
        description: Optional[str] = None,
    ):
        super().__init__(parent)

        # Transparent background
        self.setAutoFillBackground(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.config.spacing.tiny)

        # Label
        self.label = SmallLabel(label, color=theme.config.text.light)
        layout.addWidget(self.label)

        # Input
        layout.addWidget(input_widget)

        # Optional description
        if description:
            desc = SmallLabel(description, color=theme.config.text.medium)
            layout.addWidget(desc)

    @staticmethod
    def create_text(
        label: str,
        placeholder: str = "",
        default: str = "",
        parent: Optional[QWidget] = None,
    ) -> Tuple["FormGroup", TextInput]:
        """Factory to create a text input form group."""
        inp = TextInput(placeholder)
        if default:
            inp.setText(str(default))
        group = FormGroup(label, inp, parent)
        return group, inp


class Tile(BaseContainer):
    """Tile component for instructions or info cards with rounded corners."""

    def __init__(self, title: str, content: str, parent: Optional[QWidget] = None):
        super().__init__(
            parent=parent,
            layout="vertical",
            bg_color=theme.config.shapes.transparent,
            border_color=theme.config.blue.blue_1,
            border_radius=theme.config.radius.rounded,
        )

        padding = theme.config.spacing.medium
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(theme.config.spacing.small)

        # Title - blue_2 color
        title_label = GroupHeaderLabel(title, align="center", color=theme.config.blue.blue_2)
        self.add(title_label)

        # Content - small font, blue_2 color
        content_label = SmallLabel(content, align="center", color=theme.config.blue.blue_2)
        # Make font even smaller
        font = content_label.font()
        font.setPointSize(theme.config.fonts.small - 1)
        content_label.setFont(font)
        content_label.setWordWrap(True)
        self.add(content_label)


class SidebarButton(QWidget):
    """Sidebar navigation button with icon and text.

    Handles selection state and hover effects programmatically.
    Uses fixed icon positioning for smooth sidebar expansion animation.
    """

    clicked = Signal()

    def __init__(
        self,
        text: str,
        icon_pixmap: Optional[QPixmap] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._text_content = text
        self._selected = False
        self._expanded = False
        self._hovered = False
        self._default_icon = icon_pixmap

        # Colors
        self._bg_color_default = "transparent"
        self._bg_color_hover = theme.config.shapes.accent
        self._bg_color_selected = theme.config.shapes.accent
        self._text_color_default = theme.config.shapes.accent
        self._text_color_hover = theme.config.text.light_blue_accent
        self._text_color_selected = theme.config.text.light_blue_accent

        # Store border radius for custom painting
        self._border_radius = theme.config.radius.small

        # Calculate icon position (centered in collapsed width)
        self._icon_area_width = theme.config.sidebar.collapsed_width
        self._button_padding = 4

        # Ensure no background fill - we handle all painting in paintEvent
        self.setAutoFillBackground(False)

        # Setup UI
        self._setup_ui(icon_pixmap)

        # Enable mouse tracking for hover
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        # Set cursor
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Set minimum height
        self.setMinimumHeight(theme.config.sidebar.button_min_height)

        # Apply initial styling
        self._update_appearance()

    def _setup_ui(self, icon_pixmap: Optional[QPixmap]):
        """Setup button UI with fixed icon positioning.

        Design: Icon area has fixed width matching collapsed sidebar.
        Text appears next to icon area when expanded.
        """
        from PySide6.QtWidgets import QLabel

        layout = QHBoxLayout(self)
        layout.setContentsMargins(self._button_padding, self._button_padding, self._button_padding, self._button_padding)
        layout.setSpacing(0)
        # CRITICAL: Anchor content to the left to prevent centering jolt when text is hidden
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Icon area: fixed width container that matches collapsed sidebar width
        self.icon_area = QWidget()
        self.icon_area.setFixedWidth(self._icon_area_width - (2 * self._button_padding))
        icon_area_layout = QHBoxLayout(self.icon_area)
        icon_area_layout.setContentsMargins(0, 0, 0, 0)
        icon_area_layout.setSpacing(0)

        if icon_pixmap:
            # Use QLabel directly for icon display
            self.icon_label = QLabel("")
            self.icon_label.setPixmap(
                icon_pixmap.scaled(
                    theme.config.sidebar.button_icon_size,
                    theme.config.sidebar.button_icon_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            # Center icon within the fixed icon area
            icon_area_layout.addStretch()
            icon_area_layout.addWidget(self.icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
            icon_area_layout.addStretch()
        else:
            self.icon_label = None

        layout.addWidget(self.icon_area)

        # Spacer widget (toggleable) - replaces addSpacing so we can hide it
        self.spacer = QWidget()
        self.spacer.setFixedWidth(theme.config.spacing.medium)
        self.spacer.setVisible(False)  # Initially hidden
        layout.addWidget(self.spacer)

        # Text label (initially hidden)
        self.text_label = BodyLabel(self._text_content)
        font = self.text_label.font()
        font.setWeight(QFont.Weight.DemiBold)
        self.text_label.setFont(font)
        self.text_label.setVisible(False)
        # Use explicit alignment for text instead of stretch to avoid layout shifts
        layout.addWidget(self.text_label)

        # Add a final stretch to consume any remaining space on the right
        # This reinforces the left alignment of the icon
        layout.addStretch(1)

    def _update_appearance(self):
        """Update button appearance based on state."""
        # Determine text color: blue_2 on hover/select, shapes.accent by default
        if self._hovered or self._selected:
            text_color = theme.config.blue.blue_2
        else:
            text_color = theme.config.shapes.accent

        # Update text color
        if self.text_label:
            palette = self.text_label.palette()
            palette.setColor(QPalette.ColorRole.WindowText, QColor(text_color))
            self.text_label.setPalette(palette)

        # Update icon color based on state
        if self.icon_label and self._default_icon:
            if self._hovered or self._selected:
                # Use blue_2 color on hover/select
                icon_color = theme.config.blue.blue_2
                self.icon_label.setPixmap(
                    self._color_pixmap(self._default_icon, icon_color).scaled(
                        theme.config.sidebar.button_icon_size,
                        theme.config.sidebar.button_icon_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                # Use default icon (shapes.accent color)
                self.icon_label.setPixmap(
                    self._default_icon.scaled(
                        theme.config.sidebar.button_icon_size,
                        theme.config.sidebar.button_icon_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

        self.update()

    def _color_pixmap(self, pixmap: QPixmap, color: str) -> QPixmap:
        """Color a pixmap by replacing white pixels with the given color."""
        result = pixmap.copy()

        # Create a painter to recolor the pixmap
        painter = QPainter(result)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(result.rect(), QColor(color))
        painter.end()

        return result

    def set_selected(self, selected: bool):
        """Set selection state."""
        self._selected = selected
        self._update_appearance()

    def set_expanded(self, expanded: bool):
        """Set expansion state."""
        self._expanded = expanded
        if self.text_label:
            self.text_label.setVisible(expanded)
        if hasattr(self, "spacer"):
            self.spacer.setVisible(expanded)

    def enterEvent(self, event):
        """Handle mouse enter."""
        self._hovered = True
        self._update_appearance()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave."""
        self._hovered = False
        self._update_appearance()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press - emit clicked signal."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        """No custom painting - rely on theme colors for text and icon highlighting."""
        # Remove the blue background highlight - just use default painting
        super().paintEvent(event)
