"""Layout container components with programmatic styling.

All containers use theme tokens and programmatic styling.
NO STYLESHEETS - only QPalette, geometry, and custom painting.
"""

from typing import Literal, Optional

from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLayout, QScrollArea, QVBoxLayout, QWidget

from vocalance.app.ui.components.simple_components import Label
from vocalance.app.ui.qt_theme import theme


class BaseContainer(QFrame):
    """Base container with programmatic styling.

    Uses theme.config.container for all spacing.
    Provides foundation for Box, Panel, Card variants.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        layout: Literal["vertical", "horizontal"] = "vertical",
        bg_color: str = None,
        border_color: str = None,
        border_radius: int = None,
    ):
        super().__init__(parent)

        # Store styling attributes
        self._bg_color = bg_color or theme.config.shapes.darkest
        self._border_color = border_color
        self._border_radius = border_radius or 0

        # Apply background color
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(self._bg_color))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Remove default frame styling
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Create layout
        if layout == "vertical":
            self._layout = QVBoxLayout(self)
        elif layout == "horizontal":
            self._layout = QHBoxLayout(self)
        else:
            self._layout = QVBoxLayout(self)

        # Base container has no margins/spacing - subclasses define based on role
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def add(self, widget: QWidget, stretch: int = 0):
        """Add widget to layout."""
        self._layout.addWidget(widget, stretch)

    def add_layout(self, layout: QLayout, stretch: int = 0):
        """Add sublayout to layout."""
        self._layout.addLayout(layout, stretch)

    def add_stretch(self, stretch: int = 1):
        """Add stretch to layout."""
        self._layout.addStretch(stretch)

    def layout(self) -> QLayout:
        """Return the layout."""
        return self._layout

    def paintEvent(self, event):
        """Custom paint for border and rounded corners."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        bg_color = self.palette().color(QPalette.ColorRole.Window)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self._border_radius, self._border_radius)
        painter.fillPath(path, bg_color)

        # Draw border if specified
        if self._border_color:
            painter.setPen(QColor(self._border_color))
            painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, self._border_radius, self._border_radius)


class Box(BaseContainer):
    """Primary content box with border and padding.

    Spacing: border (1px) + padding (box_padding) from theme.
    """

    def __init__(self, parent: Optional[QWidget] = None, layout: Literal["vertical", "horizontal"] = "vertical"):
        super().__init__(
            parent=parent,
            layout=layout,
            bg_color=theme.config.shapes.darkest,
            border_color=theme.config.shapes.medium,
            border_radius=theme.config.radius.rounded,
        )

        # Box padding - space from border to content
        padding = theme.config.container.box_padding
        self._layout.setContentsMargins(padding, padding, padding, padding)

        # Spacing between items inside box
        self._layout.setSpacing(theme.config.container.content_vertical_spacing)


class Panel(BaseContainer):
    """Secondary panel container."""

    def __init__(self, parent: Optional[QWidget] = None, layout: Literal["vertical", "horizontal"] = "vertical"):
        super().__init__(
            parent=parent,
            layout=layout,
            bg_color=theme.config.shapes.dark,
            border_color=theme.config.shapes.medium,
            border_radius=theme.config.radius.medium,
        )

        padding = theme.config.container.box_padding
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(theme.config.container.content_vertical_spacing)


class Card(BaseContainer):
    """Card container for grouped content."""

    def __init__(self, parent: Optional[QWidget] = None, layout: Literal["vertical", "horizontal"] = "vertical"):
        super().__init__(
            parent=parent,
            layout=layout,
            bg_color=theme.config.shapes.medium,
            border_color=None,
            border_radius=theme.config.radius.medium,
        )

        padding = theme.config.spacing.medium
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(theme.config.spacing.small)


class TransparentBox(BaseContainer):
    """Transparent container - inherits parent styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        layout: Literal["vertical", "horizontal"] = "vertical",
        spacing: Optional[int] = None,
    ):
        super().__init__(
            parent=parent,
            layout=layout,
            bg_color="transparent",
            border_color=None,
            border_radius=0,
        )

        # Transparent - no background fill
        self.setAutoFillBackground(False)

        # Transparent boxes have no margins, optional spacing
        if spacing is not None:
            self._layout.setSpacing(spacing)


class ContentArea(QWidget):
    """Content area widget for inside containers.

    Proper way to add content to a Box:
    - Box provides border + box_padding
    - ContentArea provides layout for items with proper spacing
    """

    def __init__(self, parent: Optional[QWidget] = None, layout: Literal["vertical", "horizontal"] = "vertical"):
        super().__init__(parent)

        # Transparent background
        self.setAutoFillBackground(False)

        if layout == "vertical":
            self._layout = QVBoxLayout(self)
        elif layout == "horizontal":
            self._layout = QHBoxLayout(self)
        else:
            self._layout = QVBoxLayout(self)

        # No margins - parent box already has padding
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Spacing between content items
        self._layout.setSpacing(theme.config.container.content_vertical_spacing)

    def add(self, widget: QWidget, stretch: int = 0):
        """Add widget to content area."""
        self._layout.addWidget(widget, stretch)

    def add_layout(self, layout: QLayout, stretch: int = 0):
        """Add sublayout."""
        self._layout.addLayout(layout, stretch)

    def add_stretch(self, stretch: int = 1):
        """Add stretch."""
        self._layout.addStretch(stretch)

    def layout(self) -> QLayout:
        """Return the layout."""
        return self._layout


class ScrollableContainer(QFrame):
    """Scrollable container with programmatic styling."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Transparent frame
        self.setAutoFillBackground(False)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        # Make scroll area transparent
        palette = self.scroll_area.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("transparent"))
        self.scroll_area.setPalette(palette)
        self.scroll_area.setAutoFillBackground(False)

        # Style scrollbar programmatically
        self._style_scrollbar()

        # Content widget with transparent background
        self.content_widget = QWidget()
        self.content_widget.setAutoFillBackground(False)
        content_palette = self.content_widget.palette()
        content_palette.setColor(QPalette.ColorRole.Window, QColor("transparent"))
        self.content_widget.setPalette(content_palette)

        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(theme.config.container.content_vertical_spacing)

        self.scroll_area.setWidget(self.content_widget)
        main_layout.addWidget(self.scroll_area)

    def _style_scrollbar(self):
        """Apply scrollbar styling programmatically."""
        # Note: QScrollBar styling is complex to do purely programmatically
        # We'll use minimal stylesheet just for scrollbar as it's internal to QScrollArea
        c = theme.config
        scrollbar_style = f"""
        QScrollBar:vertical {{
            background: {c.shapes.dark};
            width: 10px;
            margin: 0;
            border-radius: 5px;
            border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {c.shapes.light};
            min-height: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {c.shapes.lightest};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}
        """
        self.scroll_area.setStyleSheet(scrollbar_style)

    def add(self, widget: QWidget, stretch: int = 0):
        """Add widget to scrollable content."""
        self.content_layout.addWidget(widget, stretch)

    def add_stretch(self, stretch: int = 1):
        """Add stretch to content."""
        self.content_layout.addStretch(stretch)


class TwoColumnLayout(QWidget):
    """Two column layout with systematic spacing.

    Structure:
    - Main widget has two Box containers side by side
    - Each Box has: border (1px) + padding (box_padding)
    - Title and content both respect the box padding
    - Content uses ContentArea for proper spacing
    """

    def __init__(
        self,
        left_title: str = "",
        right_title: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        # Transparent background
        self.setAutoFillBackground(False)

        # Main layout - space between boxes
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.config.container.box_spacing_between)

        # Left Box
        self.left_box = Box(layout="vertical")
        if left_title:
            title_label = Label(left_title, variant="box_title")
            self.left_box.add(title_label)
            # Add extra spacing after title
            title_spacing = QWidget()
            title_spacing.setFixedHeight(theme.config.spacing.small)
            title_spacing.setAutoFillBackground(False)
            self.left_box.add(title_spacing)

        # Right Box
        self.right_box = Box(layout="vertical")
        if right_title:
            title_label = Label(right_title, variant="box_title")
            self.right_box.add(title_label)
            # Add extra spacing after title
            title_spacing = QWidget()
            title_spacing.setFixedHeight(theme.config.spacing.small)
            title_spacing.setAutoFillBackground(False)
            self.right_box.add(title_spacing)

        # Content areas
        self.left_content = ContentArea()
        self.left_box.add(self.left_content, stretch=1)

        self.right_content = ContentArea()
        self.right_box.add(self.right_content, stretch=1)

        layout.addWidget(self.left_box, stretch=1)
        layout.addWidget(self.right_box, stretch=1)
