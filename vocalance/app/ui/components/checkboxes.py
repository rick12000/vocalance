from typing import Callable, Optional

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPalette
from PySide6.QtWidgets import QCheckBox, QWidget

from vocalance.app.ui.qt_theme import theme


class Checkbox(QCheckBox):
    """Themed checkbox with custom painted indicator.

    Uses custom paintEvent for consistent indicator appearance.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        checked: bool = False,
        command: Optional[Callable[[int], None]] = None,
    ) -> None:
        super().__init__(text, parent)

        self.setFont(theme.get_font("medium"))

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.light))
        self.setPalette(palette)

        self._indicator_size = 18
        self._indicator_border_color = theme.config.shapes.light
        self._indicator_bg_color = theme.config.shapes.darkest
        self._indicator_checked_bg_color = theme.config.shapes.accent
        self._indicator_checked_border_color = theme.config.shapes.accent
        self._indicator_hover_border_color = theme.config.shapes.lightest
        self._indicator_border_radius = theme.config.radius.small // 2

        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._is_hovered = False

        self.setChecked(checked)

        if command:
            self.stateChanged.connect(command)

    def enterEvent(self, enter_event: QEvent) -> None:
        """Handle mouse enter."""
        self._is_hovered = True
        self.update()
        super().enterEvent(enter_event)

    def leaveEvent(self, leave_event: QEvent) -> None:
        """Handle mouse leave."""
        self._is_hovered = False
        self.update()
        super().leaveEvent(leave_event)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        """Custom paint for checkbox indicator."""
        super().paintEvent(paint_event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        indicator_rect_left = self.contentsRect().left()
        indicator_rect_top = (self.contentsRect().height() - self._indicator_size) // 2
        indicator_rect = self.contentsRect()
        indicator_rect.setLeft(indicator_rect_left)
        indicator_rect.setTop(indicator_rect_top)
        indicator_rect.setWidth(self._indicator_size)
        indicator_rect.setHeight(self._indicator_size)

        if self.isChecked():
            bg_color = self._indicator_checked_bg_color
            border_color = self._indicator_checked_border_color
        else:
            bg_color = self._indicator_bg_color
            border_color = self._indicator_hover_border_color if self._is_hovered else self._indicator_border_color

        path = QPainterPath()
        path.addRoundedRect(
            indicator_rect.x(),
            indicator_rect.y(),
            self._indicator_size,
            self._indicator_size,
            self._indicator_border_radius,
            self._indicator_border_radius,
        )
        painter.fillPath(path, QColor(bg_color))

        painter.setPen(QColor(border_color))
        painter.drawRoundedRect(
            indicator_rect.x(),
            indicator_rect.y(),
            self._indicator_size - 1,
            self._indicator_size - 1,
            self._indicator_border_radius,
            self._indicator_border_radius,
        )

        if self.isChecked():
            painter.setPen(QColor(theme.config.shapes.darkest))
            check_x = indicator_rect.x() + 4
            check_y = indicator_rect.y() + 9
            painter.drawLine(check_x, check_y, check_x + 3, check_y + 3)
            painter.drawLine(check_x + 3, check_y + 3, check_x + 10, check_y - 4)
