"""Container components with systematic spacing hierarchy.

SPACING SYSTEM:
1. Container has border (1px) + padding (box_padding)
2. Content inside has margin (content_horizontal_margin)
3. Items within content have spacing (content_vertical_spacing)

All spacing comes from theme.config.container - NO MAGIC NUMBERS.
"""

from typing import Optional

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLayout, QScrollArea, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme


def _get_container_stylesheet() -> str:
    """Generate stylesheet for container components.

    Containers are styled only via stylesheet - never override programmatically.
    """
    c = theme.config
    return f"""
    /* Box variant - primary content container with border */
    QFrame[variant="box"] {{
        background-color: {c.shapes.darkest};
        border: 1px solid {c.shapes.medium};
        border-radius: {c.radius.rounded}px;
    }}

    /* Panel variant - secondary container */
    QFrame[variant="panel"] {{
        background-color: {c.shapes.dark};
        border: 1px solid {c.shapes.medium};
        border-radius: {c.radius.medium}px;
    }}

    /* Card variant - elevated content */
    QFrame[variant="card"] {{
        background-color: {c.shapes.medium};
        border-radius: {c.radius.medium}px;
        border: none;
    }}

    /* Transparent variant - no styling */
    QFrame[variant="transparent"] {{
        background: transparent;
        border: none;
    }}

    /* Scrollbar styling - consistent across all scroll areas */
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


class BaseContainer(QFrame):
    """Base container with systematic spacing.

    Enforces spacing hierarchy:
    - Uses theme.config.container for all spacing
    - Never accepts magic number overrides
    - Child classes specify their role in hierarchy
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        layout: str = "vertical",
        variant: str = "default",
    ):
        super().__init__(parent)
        self.setProperty("variant", variant)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(_get_container_stylesheet())

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


class Box(BaseContainer):
    """Primary content box with border and padding.

    Spacing: border (1px) + padding (box_padding) from theme.
    Content inside should use content margins for alignment.
    """

    def __init__(self, parent: Optional[QWidget] = None, layout: str = "vertical"):
        super().__init__(parent, layout, variant="box")

        # Box padding - space from border to content
        padding = theme.config.container.box_padding
        self._layout.setContentsMargins(padding, padding, padding, padding)

        # Spacing between items inside box
        self._layout.setSpacing(theme.config.container.content_vertical_spacing)


class Panel(BaseContainer):
    """Secondary panel container."""

    def __init__(self, parent: Optional[QWidget] = None, layout: str = "vertical"):
        super().__init__(parent, layout, variant="panel")

        padding = theme.config.container.box_padding
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(theme.config.container.content_vertical_spacing)


class Card(BaseContainer):
    """Card container for grouped content."""

    def __init__(self, parent: Optional[QWidget] = None, layout: str = "vertical"):
        super().__init__(parent, layout, variant="card")

        padding = theme.config.spacing.medium
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(theme.config.spacing.small)


class TransparentBox(BaseContainer):
    """Transparent container - inherits parent spacing."""

    def __init__(self, parent: Optional[QWidget] = None, layout: str = "vertical", spacing: Optional[int] = None):
        super().__init__(parent, layout, variant="transparent")

        # Transparent boxes have no margins, optional spacing
        if spacing is not None:
            self._layout.setSpacing(spacing)


class ContentArea(QWidget):
    """Content area widget for inside containers.

    This is the proper way to add content to a Box:
    - Box provides border + box_padding
    - ContentArea provides layout for items with proper spacing
    - No margin needed as box_padding already provides space
    """

    def __init__(self, parent: Optional[QWidget] = None, layout: str = "vertical"):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")

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


class ScrollableContainer(QFrame):
    """Scrollable container with systematic spacing."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet(_get_container_stylesheet())

        # Content widget uses ContentArea pattern
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background: transparent; border: none;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(theme.config.container.content_vertical_spacing)

        self.scroll_area.setWidget(self.content_widget)
        self.layout().addWidget(self.scroll_area)

    def add(self, widget: QWidget, stretch: int = 0):
        """Add widget to scrollable content."""
        self.content_layout.addWidget(widget, stretch)

    def add_stretch(self, stretch: int = 1):
        """Add stretch to content."""
        self.content_layout.addStretch(stretch)
