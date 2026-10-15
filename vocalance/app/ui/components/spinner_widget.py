import math
from typing import Optional

from PySide6.QtCore import QMetaObject, QRectF, Qt, QTimer, Slot
from PySide6.QtGui import QBrush, QColor, QConicalGradient, QHideEvent, QPainter, QPaintEvent, QPen, QShowEvent
from PySide6.QtWidgets import QWidget

from vocalance.app.ui.qt_theme import theme


class SpinnerWidget(QWidget):
    """Themed bouncing-dot spinner driven by a frame timer."""

    def __init__(self, parent: Optional[QWidget] = None, size: int = 24) -> None:
        """``size`` is the widget height; width follows dot count and spacing."""
        super().__init__(parent)

        self._dot_count = 4
        self._dot_size = 6
        self._dot_spacing = 4
        self._bounce_height = 8

        total_width = (self._dot_count * self._dot_size) + ((self._dot_count - 1) * self._dot_spacing)
        self.setFixedSize(total_width, size)

        self._phase = 0.0
        self._is_animating = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        self._timer.setInterval(16)

        self.setVisible(False)

    def _update_animation(self) -> None:
        """Advance phase and request repaint."""
        self._phase = (self._phase + 0.15) % (2 * math.pi)
        self.update()

    def start(self) -> None:
        """Start animation (queued onto the Qt thread)."""
        QMetaObject.invokeMethod(self, "_do_start", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_start(self) -> None:
        """Run on the Qt GUI thread."""
        if not self._is_animating:
            self._is_animating = True
            self._phase = 0.0
            self.setVisible(True)
            self._timer.start()

    def stop(self) -> None:
        """Stop animation (queued onto the Qt thread)."""
        QMetaObject.invokeMethod(self, "_do_stop", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_stop(self) -> None:
        """Run on the Qt GUI thread."""
        if self._is_animating:
            self._is_animating = False
            self._timer.stop()
            self.setVisible(False)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        """Draw the bouncing dots with gradient colors."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        gradient_colors = theme.config.text.gradient_colors
        color_start = QColor(gradient_colors[0])
        color_end = QColor(gradient_colors[1])

        total_width = (self._dot_count * self._dot_size) + ((self._dot_count - 1) * self._dot_spacing)
        start_x = (self.width() - total_width) / 2
        center_y = self.height() / 2

        for i in range(self._dot_count):
            dot_phase = self._phase - (i * 0.5)

            bounce = math.sin(dot_phase)
            if bounce > 0:
                y_offset = -bounce * self._bounce_height
            else:
                y_offset = 0

            x = start_x + (i * (self._dot_size + self._dot_spacing))
            y = center_y + y_offset

            t = i / (self._dot_count - 1) if self._dot_count > 1 else 0
            dot_color = QColor(
                int(color_start.red() + t * (color_end.red() - color_start.red())),
                int(color_start.green() + t * (color_end.green() - color_start.green())),
                int(color_start.blue() + t * (color_end.blue() - color_start.blue())),
            )

            scale = 1.0 + (max(0, -y_offset / self._bounce_height) * 0.2)
            scaled_size = self._dot_size * scale

            painter.setBrush(QBrush(dot_color))
            painter.setPen(Qt.PenStyle.NoPen)

            rect = QRectF(x - (scaled_size - self._dot_size) / 2, y - scaled_size / 2, scaled_size, scaled_size)
            painter.drawEllipse(rect)

    def showEvent(self, show_event: QShowEvent) -> None:
        """Restart the timer if visible while supposed to be animating."""
        super().showEvent(show_event)
        if self._is_animating and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, hide_event: QHideEvent) -> None:
        """Stop the timer while hidden."""
        super().hideEvent(hide_event)
        if self._timer.isActive():
            self._timer.stop()


class OrangeArcSpinner(QWidget):
    """Compact rotating arc in the processing-border orange gradient."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._angle = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._advance)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

    def start_now(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop_now(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    def _advance(self) -> None:
        self._angle = (self._angle + 6.0) % 360.0
        self.update()

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen_width = max(2.0, min(self.width(), self.height()) * 0.12)
        inset = (pen_width / 2) + 1
        diameter = min(self.width(), self.height()) - (inset * 2)
        if diameter < 4:
            return
        left = (self.width() - diameter) / 2
        top = (self.height() - diameter) / 2
        center_x = left + diameter / 2
        center_y = top + diameter / 2
        orange = QColor(theme.config.shapes.orange)
        highlight = QColor(orange)
        highlight.setRed(min(255, orange.red() + 50))
        highlight.setGreen(min(255, orange.green() + 40))
        shadow = QColor(orange).darker(140)
        gradient = QConicalGradient(center_x, center_y, -self._angle)
        gradient.setColorAt(0.0, highlight)
        gradient.setColorAt(0.45, orange)
        gradient.setColorAt(1.0, shadow)
        pen = QPen(QBrush(gradient), max(2.0, diameter * 0.14))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(QRectF(left, top, diameter, diameter), int(self._angle * 16), 110 * 16)

    def hideEvent(self, hide_event: QHideEvent) -> None:
        super().hideEvent(hide_event)
        self.stop_now()
