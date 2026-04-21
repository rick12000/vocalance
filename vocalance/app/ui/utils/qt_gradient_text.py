from typing import Any, List, Optional, Union

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPaintEvent
from PySide6.QtWidgets import QLabel


class GradientDirection:
    """String tokens for ``GradientTextMixin`` gradient axes."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL = "diagonal"


class GradientTextMixin:
    """Optional gradient fill for ``QLabel`` text via ``paintEvent`` override."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gradient_enabled = False
        self._gradient_colors: List[str] = []
        self._gradient_direction: str = GradientDirection.HORIZONTAL

    def enable_gradient(self, colors: List[str], direction: Union[str, Qt.Orientation] = GradientDirection.HORIZONTAL) -> None:
        """Enable gradient painting; ``colors`` needs at least two entries."""
        if len(colors) < 2:
            raise ValueError("Gradient requires at least 2 colors")

        self._gradient_enabled = True
        self._gradient_colors = colors

        if isinstance(direction, Qt.Orientation):
            self._gradient_direction = (
                GradientDirection.HORIZONTAL if direction == Qt.Orientation.Horizontal else GradientDirection.VERTICAL
            )
        else:
            self._gradient_direction = direction

        if isinstance(self, QLabel):
            self.update()

    def disable_gradient(self) -> None:
        """Restore default ``QLabel`` painting."""
        self._gradient_enabled = False
        if isinstance(self, QLabel):
            self.update()

    def is_gradient_enabled(self) -> bool:
        """Whether custom gradient painting is active."""
        return self._gradient_enabled

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        if not self._gradient_enabled or not self._gradient_colors:
            super().paintEvent(paint_event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        text = self.text()
        if not text:
            painter.end()
            return

        painter.setFont(self.font())

        content_rect = self.contentsRect()
        if content_rect.isEmpty():
            content_rect = self.rect()

        text_rect = self._label_content_rect(content_rect)
        gradient = self._linear_gradient_for_text(painter, text, text_rect)

        path = QPainterPath()
        path.addText(text_rect.x(), text_rect.y() + painter.fontMetrics().ascent(), self.font(), text)

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

        painter.end()

    def _linear_gradient_for_text(self, painter: QPainter, text: str, text_rect: QRect) -> QLinearGradient:
        font_metrics = painter.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text)
        text_height = font_metrics.height()

        alignment = self.alignment()

        if alignment & Qt.AlignmentFlag.AlignRight:
            text_x = text_rect.right() - text_width
        elif alignment & Qt.AlignmentFlag.AlignHCenter or alignment & Qt.AlignmentFlag.AlignCenter:
            text_x = text_rect.x() + (text_rect.width() - text_width) / 2
        else:
            text_x = text_rect.x()

        if alignment & Qt.AlignmentFlag.AlignBottom:
            text_y = text_rect.bottom() - text_height
        elif alignment & Qt.AlignmentFlag.AlignVCenter or alignment & Qt.AlignmentFlag.AlignCenter:
            text_y = text_rect.y() + (text_rect.height() - text_height) / 2
        else:
            text_y = text_rect.y()

        if self._gradient_direction == GradientDirection.HORIZONTAL or self._gradient_direction == Qt.Orientation.Horizontal:
            gradient = QLinearGradient(text_x, text_y, text_x + text_width, text_y)
        elif self._gradient_direction == GradientDirection.VERTICAL or self._gradient_direction == Qt.Orientation.Vertical:
            gradient = QLinearGradient(text_x, text_y, text_x, text_y + text_height)
        elif self._gradient_direction == GradientDirection.DIAGONAL:
            gradient = QLinearGradient(text_x, text_y, text_x + text_width, text_y + text_height)
        else:
            gradient = QLinearGradient(text_x, text_y, text_x + text_width, text_y)

        num_colors = len(self._gradient_colors)
        for i, color in enumerate(self._gradient_colors):
            position = i / (num_colors - 1) if num_colors > 1 else 0
            gradient.setColorAt(position, QColor(color))

        return gradient

    def _label_content_rect(self, content_rect: QRect) -> QRect:
        return content_rect


def create_gradient_label(
    text: str = "",
    gradient_colors: Optional[List[str]] = None,
    direction: Qt.Orientation = Qt.Orientation.Horizontal,
    parent=None,
) -> QLabel:
    """Return a ``QLabel`` subclass that optionally enables a gradient."""

    class GradientLabel(GradientTextMixin, QLabel):
        pass

    label = GradientLabel(text, parent)

    if gradient_colors:
        label.enable_gradient(gradient_colors, direction)

    return label
