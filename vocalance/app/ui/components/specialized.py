"""Specialized UI components for specific purposes.

Complex widgets like expandable sidebar with animations.
NO STYLESHEETS - all styling done programmatically.
"""

from typing import List, Optional

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

from vocalance.app.ui.components.complex_components import SidebarButton
from vocalance.app.ui.qt_theme import theme


class SidebarButtonManager:
    """Manages selection state of sidebar buttons."""

    def __init__(self):
        self._buttons: List[SidebarButton] = []
        self._selected: Optional[SidebarButton] = None

    def add(self, button: SidebarButton):
        """Add button to manager."""
        self._buttons.append(button)
        button.clicked.connect(lambda: self.select(button))

    def select(self, button: SidebarButton):
        """Select a button, deselecting others."""
        if self._selected:
            self._selected.set_selected(False)

        button.set_selected(True)
        self._selected = button

    def set_expanded(self, expanded: bool):
        """Set expanded state for all buttons."""
        for btn in self._buttons:
            btn.set_expanded(expanded)


class ExpandableSidebar(QFrame):
    """Sidebar that expands on hover with animation."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Transparent background
        self.setAutoFillBackground(False)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Width configuration
        self.collapsed_width = theme.config.sidebar.collapsed_width
        self.expanded_width = theme.config.sidebar.expanded_width
        self.setFixedWidth(self.collapsed_width)

        # Enable mouse tracking
        self.setMouseTracking(True)

        # Layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, theme.config.sidebar.padding_top, 0, 0)
        self._layout.setSpacing(0)

        # Manager for button selection
        self.manager = SidebarButtonManager()

        # Setup animations
        self._anim_min = QPropertyAnimation(self, b"minimumWidth")
        self._anim_max = QPropertyAnimation(self, b"maximumWidth")
        for anim in (self._anim_min, self._anim_max):
            anim.setDuration(theme.config.sidebar.animation_duration)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def add_widget(self, widget: QWidget):
        """Add widget to sidebar layout."""
        self._layout.addWidget(widget)

    def add_stretch(self):
        """Add stretch to sidebar layout."""
        self._layout.addStretch()

    def enterEvent(self, event):
        """Handle mouse enter - expand sidebar."""
        self._animate(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave - collapse sidebar."""
        self._animate(False)
        super().leaveEvent(event)

    def _animate(self, expand: bool):
        """Animate sidebar width.

        Args:
            expand: True to expand, False to collapse
        """
        target_width = self.expanded_width if expand else self.collapsed_width

        self._anim_min.setEndValue(target_width)
        self._anim_max.setEndValue(target_width)

        self._anim_min.start()
        self._anim_max.start()

        # Update button text visibility
        self.manager.set_expanded(expand)
