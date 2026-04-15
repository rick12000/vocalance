from typing import Optional

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from vocalance.app.ui.components.layout_core import Box, ContentArea, ScrollableContainer, TransparentWidget
from vocalance.app.ui.qt_theme import theme


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
    ) -> None:
        super().__init__(parent)

        # Import here to avoid circular imports
        from vocalance.app.ui.components.labels import BoxTitleLabel

        # Transparent background
        self.setAutoFillBackground(False)

        # Main layout - space between boxes
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.config.container.box_spacing_between)

        # Left Box
        self.left_box = Box(layout="vertical")
        if left_title:
            title_label = BoxTitleLabel(left_title)
            self.left_box.add(title_label)
            # Add extra spacing after title
            self.left_box.layout().addSpacing(theme.config.container.box_title_spacing)

        # Right Box
        self.right_box = Box(layout="vertical")
        if right_title:
            title_label = BoxTitleLabel(right_title)
            self.right_box.add(title_label)
            # Add extra spacing after title
            self.right_box.layout().addSpacing(theme.config.container.box_title_spacing)

        # Content areas
        self.left_content = ContentArea()
        self.left_box.add(self.left_content, stretch=1)

        self.right_content = ContentArea()
        self.right_box.add(self.right_content, stretch=1)

        layout.addWidget(self.left_box, stretch=1)
        layout.addWidget(self.right_box, stretch=1)


# =============================================================================
# High-Level Layout Orchestration Components
# =============================================================================


class FormField(QWidget):
    """Label + Input pairing for forms.

    Provides consistent spacing between label and input widget.
    """

    def __init__(
        self,
        label: str,
        input_widget: QWidget,
        parent: Optional[QWidget] = None,
        description: Optional[str] = None,
    ) -> None:
        super().__init__(parent)

        # Import here to avoid circular imports
        from vocalance.app.ui.components.labels import SmallLabel

        # Transparent background
        self.setAutoFillBackground(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.config.spacing.tiny)

        # Label
        self.label = SmallLabel(label, color=theme.config.text.light)
        layout.addWidget(self.label)

        # Input widget
        self.input_widget = input_widget
        layout.addWidget(input_widget)

        # Optional description
        if description:
            desc = SmallLabel(description, color=theme.config.text.medium)
            layout.addWidget(desc)


class ListItem(QWidget):
    """Standard list item with systematic spacing and transparent background."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
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

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        """Add widget to list item."""
        self._layout.addWidget(widget, stretch)


class GroupHeader(QWidget):
    """Group header for lists with systematic spacing."""

    def __init__(self, text: str, is_first: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Import here to avoid circular imports
        from vocalance.app.ui.components.labels import GroupHeaderLabel

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
        header_label = GroupHeaderLabel(text)
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


class CollapsibleSection(QWidget):
    """Collapsible section with header and content area.

    Features:
    - Circular expand/collapse button with arrow icons (right when collapsed, down when expanded)
    - Content area that shows/hides based on state
    - Divider appears only when content is expanded
    - Configurable to start expanded or collapsed
    """

    def __init__(self, title: str, is_first: bool = False, start_expanded: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        from vocalance.app.ui.components.buttons import CollapseButton, ExpandButton

        self.is_expanded = start_expanded

        # Transparent background
        self.setAutoFillBackground(False)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top margin - first group has less
        top_margin = (
            theme.config.container.group_header_first_margin_top if is_first else theme.config.container.group_header_margin_top
        )
        if top_margin > 0:
            main_layout.addSpacing(top_margin)

        # Header container
        header_widget = QWidget()
        header_widget.setAutoFillBackground(False)

        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(theme.config.spacing.small)

        # Arrow button - show expand or collapse based on state
        if self.is_expanded:
            self.arrow_button = CollapseButton()
        else:
            self.arrow_button = ExpandButton()

        self.arrow_button.clicked.connect(self._toggle_expanded)
        header_layout.addWidget(self.arrow_button)

        # Title label - use medium text color
        from vocalance.app.ui.components.labels import GroupHeaderLabel

        self.title_label = GroupHeaderLabel(title, color=theme.config.text.medium)
        header_layout.addWidget(self.title_label, stretch=1)

        main_layout.addWidget(header_widget)

        # Add bottom spacing after header
        main_layout.addSpacing(theme.config.container.group_header_margin_bottom)

        # Divider - only show when expanded
        self.divider = QWidget()
        self.divider.setFixedHeight(1)
        divider_palette = self.divider.palette()
        divider_palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.medium))
        self.divider.setPalette(divider_palette)
        self.divider.setAutoFillBackground(True)
        main_layout.addWidget(self.divider)
        self.divider.setVisible(self.is_expanded)

        # Bottom spacing after divider
        bottom_margin = theme.config.container.divider_margin_bottom
        if bottom_margin > 0:
            self.divider_spacing = bottom_margin
            main_layout.addSpacing(bottom_margin)
        else:
            self.divider_spacing = 0

        self.content_widget = TransparentWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        _indent = theme.config.components.button_height + theme.config.spacing.small
        self.content_layout.setContentsMargins(_indent, 0, 0, 0)
        self.content_layout.setSpacing(theme.config.container.list_item_spacing)

        main_layout.addWidget(self.content_widget)

        # Set initial visibility
        self.content_widget.setVisible(self.is_expanded)

    def _toggle_expanded(self) -> None:
        """Toggle the expanded/collapsed state."""
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        self.divider.setVisible(self.is_expanded)

        # Replace arrow button with the opposite state
        from vocalance.app.ui.components.buttons import CollapseButton, ExpandButton

        # Get the parent layout of the arrow button
        parent_layout = self.arrow_button.parent().layout()

        # Remove old button
        parent_layout.removeWidget(self.arrow_button)
        self.arrow_button.deleteLater()

        # Create new button
        if self.is_expanded:
            self.arrow_button = CollapseButton()
        else:
            self.arrow_button = ExpandButton()

        self.arrow_button.clicked.connect(self._toggle_expanded)
        parent_layout.insertWidget(0, self.arrow_button)

    def add_item(self, widget: QWidget) -> None:
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)


class ListForm(QWidget):
    """High-level component for scrollable list with items and group headers.

    Manages a scrollable list area with methods to add items, headers,
    and clear the list. Useful for views displaying lists of commands,
    prompts, sounds, etc.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Transparent background
        self.setAutoFillBackground(False)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scrollable container
        self._scroll_container = ScrollableContainer()
        layout.addWidget(self._scroll_container)

        # Internal list widget for items
        self._list_widget = TransparentWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(theme.config.container.list_item_spacing)
        self._list_layout.setContentsMargins(0, 0, theme.config.spacing.small, 0)

        self._scroll_container.content_layout.addWidget(self._list_widget)

    def add_item(self, widget: QWidget) -> None:
        """Add an item widget to the list."""
        self._list_layout.addWidget(widget)

    def add_header(self, text: str, is_first: bool = False) -> None:
        """Add a group header to the list."""
        header = GroupHeader(text, is_first=is_first)
        self._list_layout.addWidget(header)

    def add_stretch(self) -> None:
        """Add stretch at the end of the list."""
        self._list_layout.addStretch()

    def clear_items(self) -> None:
        """Clear all items from the list."""
        while self._list_layout.count() > 0:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    @property
    def item_count(self) -> int:
        """Return the number of items in the list."""
        return self._list_layout.count()
