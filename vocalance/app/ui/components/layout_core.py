from typing import Literal, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPalette
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLayout, QScrollArea, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme


class TransparentWidget(QWidget):
    """A QWidget that guarantees transparent background with no stylesheet interference."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Disable auto fill background - we'll handle it ourselves
        self.setAutoFillBackground(False)
        # Set widget to not receive stylesheet styling
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        # Ensure palette is transparent on all roles
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("transparent"))
        palette.setColor(QPalette.ColorRole.Base, QColor("transparent"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("transparent"))
        self.setPalette(palette)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        """Explicitly paint transparent background."""
        # Do nothing - let parent show through


class TransparentViewport(QWidget):
    """A viewport widget that guarantees transparent background."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        # Ensure palette is transparent on all background roles
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("transparent"))
        palette.setColor(QPalette.ColorRole.Base, QColor("transparent"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("transparent"))
        self.setPalette(palette)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        """Don't paint any background."""


class BaseContainer(QFrame):
    """Base container with programmatic styling.

    Uses theme.config.container for all spacing.
    Provides foundation for Box, Panel, Card variants.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        layout: Literal["vertical", "horizontal"] = "vertical",
        bg_color: Optional[str] = None,
        border_color: Optional[str] = None,
        border_radius: Optional[int] = None,
    ) -> None:
        super().__init__(parent)

        # Store styling attributes
        self._bg_color = bg_color or theme.config.shapes.darkest
        self._border_color = border_color
        self._border_radius = border_radius or 0

        # Apply background color
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(self._bg_color))
        self.setPalette(palette)
        # Don't auto-fill - we handle all painting in paintEvent
        self.setAutoFillBackground(False)

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

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        """Add widget to layout."""
        self._layout.addWidget(widget, stretch)

    def add_layout(self, layout: QLayout, stretch: int = 0) -> None:
        """Add sublayout to layout."""
        self._layout.addLayout(layout, stretch)

    def add_stretch(self, stretch: int = 1) -> None:
        """Add stretch to layout."""
        self._layout.addStretch(stretch)

    def layout(self) -> QLayout:
        """Return the layout."""
        return self._layout

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        """Custom paint for border and rounded corners."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Draw background with rounded corners
        bg_color = self.palette().color(QPalette.ColorRole.Window)
        path = QPainterPath()
        path.addRoundedRect(0.5, 0.5, self.width() - 1, self.height() - 1, self._border_radius, self._border_radius)

        # Fill the background
        painter.fillPath(path, bg_color)

        # Draw border if specified
        if self._border_color:
            painter.setPen(QColor(self._border_color))
            painter.drawPath(path)

        # Let parent handle child widget painting
        painter.end()


class Box(BaseContainer):
    """Primary content box with border and padding.

    Spacing: border (1px) + padding (box_padding) from theme.
    """

    def __init__(self, parent: Optional[QWidget] = None, layout: Literal["vertical", "horizontal"] = "vertical") -> None:
        super().__init__(
            parent=parent,
            layout=layout,
            bg_color=theme.config.shapes.dark,
            border_color=None,
            border_radius=theme.config.radius.rounded,
        )

        # Box padding - space from border to content
        padding = theme.config.container.box_padding
        self._layout.setContentsMargins(padding, padding, padding, padding)

        # Spacing between items inside box
        self._layout.setSpacing(theme.config.container.content_vertical_spacing)


class Panel(BaseContainer):
    """Secondary panel container."""

    def __init__(self, parent: Optional[QWidget] = None, layout: Literal["vertical", "horizontal"] = "vertical") -> None:
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

    def __init__(self, parent: Optional[QWidget] = None, layout: Literal["vertical", "horizontal"] = "vertical") -> None:
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
    """Transparent container - inherits parent styling.

    Prevents stylesheet and palette inheritance to ensure transparency.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        layout: Literal["vertical", "horizontal"] = "vertical",
        spacing: Optional[int] = None,
    ) -> None:
        super().__init__(
            parent=parent,
            layout=layout,
            bg_color="transparent",
            border_color=None,
            border_radius=0,
        )

        # Transparent - no background fill
        self.setAutoFillBackground(False)
        # Prevent stylesheet styling of background
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        # Transparent boxes have no margins, optional spacing
        if spacing is not None:
            self._layout.setSpacing(spacing)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        """Override to not paint anything - let parent show through."""
        # Don't call parent paintEvent - that would draw the background
        # Just let the transparent background work


class ContentArea(QWidget):
    """Content area widget for inside containers.

    Proper way to add content to a Box:
    - Box provides border + box_padding
    - ContentArea provides layout for items with proper spacing
    """

    def __init__(self, parent: Optional[QWidget] = None, layout: Literal["vertical", "horizontal"] = "vertical") -> None:
        super().__init__(parent)

        # Transparent background
        self.setAutoFillBackground(False)
        # Prevent stylesheet styling
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        # Ensure palette is transparent
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("transparent"))
        palette.setColor(QPalette.ColorRole.Base, QColor("transparent"))
        self.setPalette(palette)

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

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        """Add widget to content area."""
        self._layout.addWidget(widget, stretch)

    def add_layout(self, layout: QLayout, stretch: int = 0) -> None:
        """Add sublayout."""
        self._layout.addLayout(layout, stretch)

    def add_stretch(self, stretch: int = 1) -> None:
        """Add stretch."""
        self._layout.addStretch(stretch)

    def layout(self) -> QLayout:
        """Return the layout."""
        return self._layout


class ScrollableContainer(QFrame):
    """Scrollable container with programmatic styling.

    Uses centralized QSS for scrollbar styling.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Transparent frame
        self.setAutoFillBackground(False)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("VocalanceScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        # Replace the viewport with a transparent one BEFORE adding content
        transparent_viewport = TransparentViewport()
        self.scroll_area.setViewport(transparent_viewport)

        # Content widget with guaranteed transparent background
        self.content_widget = TransparentWidget()

        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(theme.config.container.content_vertical_spacing)

        # Set widget AFTER viewport is set
        self.scroll_area.setWidget(self.content_widget)

        main_layout.addWidget(self.scroll_area)

    def clear_content(self) -> None:
        """Replace the scroll body with an empty widget and layout.

        Rebuilding by only ``takeAt`` + ``deleteLater`` leaves former children parented to
        the old content widget until the event loop runs, so new siblings overlap them.
        ``QScrollArea.setWidget`` removes the previous widget from the scene graph and
        destroys it; do not call ``deleteLater`` on that widget or it would be freed twice.
        """

        self.content_widget = TransparentWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(theme.config.container.content_vertical_spacing)
        self.scroll_area.setWidget(self.content_widget)

    def add(self, widget: QWidget, stretch: int = 0):
        """Add widget to scrollable content."""
        self.content_layout.addWidget(widget, stretch)

    def add_stretch(self, stretch: int = 1):
        """Add stretch to content."""
        self.content_layout.addStretch(stretch)
