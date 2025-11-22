"""Complex UI components built from atoms and containers.

All components use systematic spacing from theme.
NO MAGIC NUMBERS. NO STYLESHEET OVERRIDES.
"""

from typing import Optional

from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme

from .atoms import Label
from .containers import Box, ContentArea


class Tile(QFrame):
    """Tile component for instructions or info cards."""

    def __init__(self, title: str, content: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("variant", "card")
        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        padding = theme.config.spacing.medium
        layout.setContentsMargins(padding, padding, padding, padding)
        layout.setSpacing(theme.config.spacing.small)

        # Title
        title_label = Label(title, variant="subtitle", align="center")
        layout.addWidget(title_label)

        # Content
        content_label = Label(content, variant="small", align="center")
        content_label.setWordWrap(True)
        layout.addWidget(content_label)


class TwoColumnLayout(QWidget):
    """Two column layout with systematic spacing.

    Structure:
    - Main widget has two Box containers side by side
    - Each Box has: border (1px) + padding (box_padding)
    - Title and content both respect the box padding
    - Content uses ContentArea for proper spacing
    """

    def __init__(self, left_title: str = "", right_title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")

        # Main layout - space between boxes
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.config.container.box_spacing_between)

        # Left Box - uses Box padding system
        self.left_box = Box(layout="vertical")
        if left_title:
            title_label = Label(left_title, variant="title")
            self.left_box.add(title_label)

        # Right Box - uses Box padding system
        self.right_box = Box(layout="vertical")
        if right_title:
            title_label = Label(right_title, variant="title")
            self.right_box.add(title_label)

        # Content areas - transparent widgets with proper spacing
        self.left_content = ContentArea()
        self.left_box.add(self.left_content, stretch=1)

        self.right_content = ContentArea()
        self.right_box.add(self.right_content, stretch=1)

        layout.addWidget(self.left_box, stretch=1)
        layout.addWidget(self.right_box, stretch=1)


class ListItem(QWidget):
    """Standard list item with systematic spacing."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")

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
        self.setStyleSheet("background: transparent; border: none;")

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
        header_label.setStyleSheet(
            f"""
            color: {theme.config.text.medium};
            background: transparent;
            border: none;
            padding: 0px;
            margin: 0px;
            margin-bottom: {theme.config.container.group_header_margin_bottom}px;
        """
        )
        layout.addWidget(header_label)

        # Divider
        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {theme.config.shapes.medium}; border: none;")
        layout.addWidget(divider)

        # Bottom spacing after divider
        bottom_margin = theme.config.container.divider_margin_bottom
        if bottom_margin > 0:
            layout.addSpacing(bottom_margin)
