from typing import Any, List, Optional

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

from vocalance.app.ui.components.complex_components import SidebarButton
from vocalance.app.ui.qt_theme import theme


class SidebarButtonManager:
    """Manages selection state of sidebar buttons."""

    def __init__(self) -> None:
        self._buttons: List[SidebarButton] = []
        self._selected: Optional[SidebarButton] = None

    def add(self, button: SidebarButton) -> None:
        """Add button to manager."""
        self._buttons.append(button)
        button.clicked.connect(lambda: self.select(button))

    def select(self, button: SidebarButton) -> None:
        """Select a button, deselecting others."""
        if self._selected:
            self._selected.set_selected(False)

        button.set_selected(True)
        self._selected = button

    def set_expanded(self, expanded: bool) -> None:
        """Set expanded state for all buttons."""
        for btn in self._buttons:
            btn.set_expanded(expanded)

    def set_text_opacity(self, opacity: float) -> None:
        """Set text opacity for all buttons."""
        for btn in self._buttons:
            btn.set_text_opacity(opacity)


class ExpandableSidebar(QFrame):
    """Sidebar that expands on hover with animation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
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
            anim.setEasingCurve(QEasingCurve.Type.OutQuart)

        # Track animation progress
        self._anim_min.valueChanged.connect(self._on_width_changed)
        self._anim_min.finished.connect(self._on_animation_finished)
        self._expanding = False

    def add_widget(self, widget: QWidget) -> None:
        """Add widget to sidebar layout."""
        self._layout.addWidget(widget)

    def add_stretch(self) -> None:
        """Add stretch to sidebar layout."""
        self._layout.addStretch()

    def enterEvent(self, enter_event: QEvent) -> None:
        """Handle mouse enter - expand sidebar."""
        self._animate(True)
        super().enterEvent(enter_event)

    def leaveEvent(self, leave_event: QEvent) -> None:
        """Handle mouse leave - collapse sidebar."""
        self._animate(False)
        super().leaveEvent(leave_event)

    def _animate(self, expand: bool) -> None:
        """Animate sidebar width.

        Args:
            expand: True to expand, False to collapse
        """
        self._expanding = expand
        target_width = self.expanded_width if expand else self.collapsed_width

        self._anim_min.setEndValue(target_width)
        self._anim_max.setEndValue(target_width)

        if expand:
            # Show text immediately but transparent
            self.manager.set_expanded(True)
            self.manager.set_text_opacity(0.0)

        self._anim_min.start()
        self._anim_max.start()

        # Update button text visibility is handled in _on_animation_finished for collapse

    def _on_width_changed(self, value: Any) -> None:
        """Handle width animation progress."""
        current_width = int(value)

        # Calculate progress
        total_delta = self.expanded_width - self.collapsed_width
        if total_delta <= 0:
            return

        progress = (current_width - self.collapsed_width) / total_delta
        # Clamp progress
        progress = max(0.0, min(1.0, progress))

        # Update opacity
        self.manager.set_text_opacity(progress)

    def _on_animation_finished(self) -> None:
        """Handle animation completion."""
        if not self._expanding:
            # If collapsed, hide text completely
            self.manager.set_expanded(False)
            self.manager.set_text_opacity(0.0)
        else:
            # Ensure fully visible
            self.manager.set_text_opacity(1.0)
