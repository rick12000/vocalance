"""Complex UI components built from simple components.

All components use systematic spacing from theme.
NO STYLESHEETS - built from simple_components and layouts.
"""

from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from vocalance.app.ui.components.layouts import BaseContainer
from vocalance.app.ui.components.simple_components import Input, Label
from vocalance.app.ui.qt_theme import theme


class FormGroup(QWidget):
    """Container for a label and an input field."""

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
        self.label = Label(label, variant="small", color=theme.config.text.light)
        layout.addWidget(self.label)

        # Input
        layout.addWidget(input_widget)

        # Optional description
        if description:
            desc = Label(description, variant="small", color=theme.config.text.medium)
            layout.addWidget(desc)

    @staticmethod
    def create_text(
        label: str,
        placeholder: str = "",
        default: str = "",
        parent: Optional[QWidget] = None,
    ) -> Tuple["FormGroup", Input]:
        """Factory to create a text input form group."""
        inp = Input(placeholder)
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
            bg_color=theme.config.shapes.medium,
            border_color=None,
            border_radius=theme.config.radius.large,  # Rounded corners
        )

        padding = theme.config.spacing.medium
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(theme.config.spacing.small)

        # Title
        title_label = Label(title, variant="group_header", align="center")
        self.add(title_label)

        # Content - use consistent small font
        content_label = Label(content, variant="small", align="center")
        content_label.setWordWrap(True)
        self.add(content_label)


class ListItem(QWidget):
    """Standard list item with systematic spacing and transparent background."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Transparent background
        self.setAutoFillBackground(False)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("transparent"))
        self.setPalette(palette)

        self._layout = QHBoxLayout(self)

        # Use list item padding from theme
        v_pad = theme.config.container.list_item_padding_vertical
        h_pad = theme.config.container.list_item_padding_horizontal
        self._layout.setContentsMargins(h_pad, v_pad, h_pad, v_pad)
        self._layout.setSpacing(theme.config.spacing.small)

    def add(self, widget: QWidget, stretch: int = 0):
        """Add widget to list item."""
        self._layout.addWidget(widget, stretch)


class GroupHeader(QWidget):
    """Group header for lists with systematic spacing."""

    def __init__(self, text: str, is_first: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Transparent background
        self.setAutoFillBackground(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top margin - first group has less
        top_margin = (
            theme.config.container.group_header_first_margin_top if is_first else theme.config.container.group_header_margin_top
        )
        if top_margin > 0:
            layout.addSpacing(top_margin)

        # Header label
        header_label = Label(text, variant="group_header")
        layout.addWidget(header_label)

        # Add bottom spacing
        layout.addSpacing(theme.config.container.group_header_margin_bottom)

        # Divider
        divider = QWidget()
        divider.setFixedHeight(1)
        palette = divider.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.medium))
        divider.setPalette(palette)
        divider.setAutoFillBackground(True)
        layout.addWidget(divider)

        # Bottom spacing after divider
        bottom_margin = theme.config.container.divider_margin_bottom
        if bottom_margin > 0:
            layout.addSpacing(bottom_margin)


class SidebarButton(QWidget):
    """Sidebar navigation button with icon and text.

    Handles selection state and hover effects programmatically.
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
        """Setup button UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(theme.config.spacing.small)

        # Icon label
        if icon_pixmap:
            self.icon_label = Label("")
            self.icon_label.setPixmap(
                icon_pixmap.scaled(
                    theme.config.sidebar.button_icon_size,
                    theme.config.sidebar.button_icon_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            layout.addWidget(self.icon_label)
        else:
            self.icon_label = None

        # Text label (initially hidden in collapsed state)
        self.text_label = Label(self._text_content, variant="body")
        self.text_label.setVisible(False)
        layout.addWidget(self.text_label, stretch=1)

        layout.addStretch()

    def _update_appearance(self):
        """Update button appearance based on state."""
        # Determine colors
        if self._selected:
            bg_color = self._bg_color_selected
            text_color = self._text_color_selected
        elif self._hovered:
            bg_color = self._bg_color_hover
            text_color = self._text_color_hover
        else:
            bg_color = self._bg_color_default
            text_color = self._text_color_default

        # Apply background
        if bg_color == "transparent":
            self.setAutoFillBackground(False)
        else:
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor(bg_color))
            self.setPalette(palette)
            self.setAutoFillBackground(True)

        # Update text color
        if self.text_label:
            palette = self.text_label.palette()
            palette.setColor(QPalette.ColorRole.WindowText, QColor(text_color))
            self.text_label.setPalette(palette)

        self.update()

    def set_selected(self, selected: bool):
        """Set selection state."""
        self._selected = selected
        self._update_appearance()

    def set_expanded(self, expanded: bool):
        """Set expansion state."""
        self._expanded = expanded
        if self.text_label:
            self.text_label.setVisible(expanded)

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
