from typing import List, Optional

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFrame, QPushButton, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme


def _get_sidebar_button_stylesheet() -> str:
    """Generate stylesheet for sidebar buttons."""
    c = theme.config
    return f"""
    QPushButton[buttonType="sidebar"] {{
        background-color: transparent;
        color: {c.shapes.accent};
        border-radius: {c.radius.small}px;
        text-align: left;
        padding: 4px;
        font-weight: 600;
        border: none;
        outline: none;
    }}

    QPushButton[buttonType="sidebar"]:hover {{
        color: {c.text.light_blue_accent};
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {c.gradients.blue_rose_start}, stop:1 {c.gradients.blue_rose_end});
    }}

    QPushButton[buttonType="sidebar"][selected="true"] {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {c.gradients.blue_rose_start}, stop:1 {c.gradients.blue_rose_end});
        color: {c.text.light_blue_accent};
    }}
    """


class SidebarButton(QPushButton):
    """Sidebar navigation button with icon and text."""

    def __init__(self, text: str, icon_pixmap: Optional[QPixmap] = None, parent: Optional[QWidget] = None):
        super().__init__("", parent)
        self.setProperty("buttonType", "sidebar")
        self.setStyleSheet(_get_sidebar_button_stylesheet())
        self._text_content = text
        self._selected = False
        self._expanded = False
        self._default_icon = icon_pixmap
        self._hover_icon = None

        if icon_pixmap:
            self.setIcon(QIcon(icon_pixmap))
            self.setIconSize(QSize(theme.config.sidebar_layout.button_icon_size, theme.config.sidebar_layout.button_icon_size))
            self._generate_hover_icon()

        self.setMinimumHeight(50)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _generate_hover_icon(self):
        """Generate hover state icon."""
        if not self._default_icon:
            return

        try:
            # Simple approach: create a tinted version by manipulating the QPixmap
            # For now, just use the original icon as fallback to avoid PIL conversion issues
            # The gradient styling in CSS will handle the visual effect
            self._hover_icon = self._default_icon

        except Exception as e:
            print(f"Error generating hover icon: {e}")
            self._hover_icon = self._default_icon

    def set_selected(self, selected: bool):
        self._selected = selected
        self.setProperty("selected", "true" if selected else "false")

        # Update icon
        icon = self._hover_icon if (selected and self._hover_icon) else self._default_icon
        if icon:
            self.setIcon(QIcon(icon))

        self.style().unpolish(self)
        self.style().polish(self)

    def set_expanded(self, expanded: bool):
        self._expanded = expanded
        self.setText(self._text_content if expanded else "")

    def enterEvent(self, event):
        if self._hover_icon:
            self.setIcon(QIcon(self._hover_icon))
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self._selected and self._default_icon:
            self.setIcon(QIcon(self._default_icon))
        elif self._selected and self._hover_icon:
            self.setIcon(QIcon(self._hover_icon))
        super().leaveEvent(event)


class SidebarButtonManager:
    """Manages selection state of sidebar buttons."""

    def __init__(self):
        self._buttons: List[SidebarButton] = []
        self._selected: Optional[SidebarButton] = None

    def add(self, button: SidebarButton):
        self._buttons.append(button)
        button.clicked.connect(lambda: self.select(button))

    def select(self, button: SidebarButton):
        if self._selected:
            self._selected.set_selected(False)

        button.set_selected(True)
        self._selected = button

    def set_expanded(self, expanded: bool):
        for btn in self._buttons:
            btn.set_expanded(expanded)


class ExpandableSidebar(QFrame):
    """Sidebar that expands on hover."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("frameType", "sidebar")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            """
            QFrame[frameType="sidebar"] {{
                background: transparent;
                border: none;
            }}
        """
        )

        self.collapsed_width = theme.config.sidebar_layout.collapsed_width
        self.expanded_width = theme.config.sidebar_layout.expanded_width
        self.setFixedWidth(self.collapsed_width)
        self.setMouseTracking(True)

        # Layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, theme.config.sidebar_layout.top_spacing, 0, 0)
        self._layout.setSpacing(0)

        # Manager
        self.manager = SidebarButtonManager()

        # Animations
        self._anim_min = QPropertyAnimation(self, b"minimumWidth")
        self._anim_max = QPropertyAnimation(self, b"maximumWidth")
        for anim in (self._anim_min, self._anim_max):
            anim.setDuration(theme.config.sidebar_layout.animation_duration)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def add_widget(self, widget: QWidget):
        self._layout.addWidget(widget)

    def add_stretch(self):
        self._layout.addStretch()

    def enterEvent(self, event):
        self._animate(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate(False)
        super().leaveEvent(event)

    def _animate(self, expand: bool):
        w = self.expanded_width if expand else self.collapsed_width

        self._anim_min.setEndValue(w)
        self._anim_max.setEndValue(w)

        self._anim_min.start()
        self._anim_max.start()

        self.manager.set_expanded(expand)
