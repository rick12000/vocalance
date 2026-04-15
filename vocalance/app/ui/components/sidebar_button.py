from typing import Optional

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPalette, QPixmap
from PySide6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QWidget

from vocalance.app.ui.components.icon_widget import IconWidget
from vocalance.app.ui.components.labels import BodyLabel
from vocalance.app.ui.qt_theme import theme


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
    ) -> None:
        super().__init__(parent)

        self._text_content = text
        self._selected = False
        self._expanded = False
        self._hovered = False
        self._default_icon = icon_pixmap

        self._bg_color_default = "transparent"
        self._bg_color_hover = theme.config.shapes.accent
        self._bg_color_selected = theme.config.shapes.accent
        self._text_color_default = theme.config.shapes.accent
        self._text_color_hover = theme.config.blue.blue_1
        self._text_color_selected = theme.config.blue.blue_1

        self._border_radius = theme.config.radius.small

        self._icon_area_width = theme.config.sidebar.collapsed_width
        self._button_padding_v = 3

        self.setAutoFillBackground(False)

        self._setup_ui(icon_pixmap)

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.setMinimumHeight(theme.config.sidebar.button_min_height)

        self._update_appearance()

    def _setup_ui(self, icon_pixmap: Optional[QPixmap]) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, self._button_padding_v, 0, self._button_padding_v)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.icon_area = QWidget()
        self.icon_area.setFixedWidth(self._icon_area_width)
        icon_area_layout = QHBoxLayout(self.icon_area)
        icon_area_layout.setContentsMargins(0, 0, 0, 0)
        icon_area_layout.setSpacing(0)

        if icon_pixmap:
            self.icon_widget = IconWidget(icon_pixmap, theme.config.sidebar.button_icon_size)
            icon_area_layout.addStretch()
            icon_area_layout.addWidget(self.icon_widget, alignment=Qt.AlignmentFlag.AlignCenter)
            icon_area_layout.addStretch()
        else:
            self.icon_widget = None

        layout.addWidget(self.icon_area)

        self.spacer = QWidget()
        self.spacer.setFixedWidth(theme.config.sidebar.button_icon_text_spacing)
        self.spacer.setVisible(False)
        layout.addWidget(self.spacer)

        self.text_label = BodyLabel(self._text_content)
        font = theme.get_font(size="medium", weight="semibold", display=True)
        self.text_label.setFont(font)

        self.opacity_effect = QGraphicsOpacityEffect(self.text_label)
        self.opacity_effect.setOpacity(1.0)
        self.text_label.setGraphicsEffect(self.opacity_effect)

        self.text_label.setVisible(False)
        layout.addWidget(self.text_label)

        layout.addStretch(1)

    def _update_appearance(self) -> None:
        if self._hovered or self._selected:
            text_color = theme.config.blue.blue_2
        else:
            text_color = theme.config.shapes.accent

        if self.text_label:
            palette = self.text_label.palette()
            palette.setColor(QPalette.ColorRole.WindowText, QColor(text_color))
            self.text_label.setPalette(palette)

        if self.icon_widget and self._default_icon:
            if self._hovered or self._selected:
                icon_color = theme.config.blue.blue_2
                colored_pixmap = self._color_pixmap(self._default_icon, icon_color)
                self.icon_widget.set_pixmap(colored_pixmap)
            else:
                self.icon_widget.set_pixmap(self._default_icon)

        self.update()

    def _color_pixmap(self, pixmap: QPixmap, color: str) -> QPixmap:
        result = pixmap.copy()

        painter = QPainter(result)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(result.rect(), QColor(color))
        painter.end()

        return result

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._update_appearance()

    def set_text_opacity(self, opacity: float) -> None:
        if hasattr(self, "opacity_effect"):
            self.opacity_effect.setOpacity(opacity)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        if self.text_label:
            self.text_label.setVisible(expanded)
        if hasattr(self, "spacer"):
            self.spacer.setVisible(expanded)

    def enterEvent(self, enter_event: QEvent) -> None:
        self._hovered = True
        self._update_appearance()
        super().enterEvent(enter_event)

    def leaveEvent(self, leave_event: QEvent) -> None:
        self._hovered = False
        self._update_appearance()
        super().leaveEvent(leave_event)

    def mousePressEvent(self, press_event: QMouseEvent) -> None:
        if press_event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(press_event)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        super().paintEvent(paint_event)
