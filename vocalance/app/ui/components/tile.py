from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPaintEvent, QPalette
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from vocalance.app.ui.components.layouts import BaseContainer
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.qt_gradient_text import GradientTextMixin


class Tile(BaseContainer):
    """Tile component for instructions or info cards with rounded corners and gradient title text."""

    def __init__(self, title: str, content: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            parent=parent,
            layout="vertical",
            bg_color="transparent",
            border_color=theme.config.shapes.light,
            border_radius=theme.config.radius.rounded,
        )

        padding = theme.config.spacing.medium
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(theme.config.spacing.small)

        class CenteredGradientLabel(GradientTextMixin, QLabel):
            """Gradient label with proper center alignment support."""

            def paintEvent(self, paint_event: QPaintEvent) -> None:
                """Override paintEvent to render centered text with gradient."""
                if not self._gradient_enabled or not self._gradient_colors:
                    QLabel.paintEvent(self, paint_event)
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

                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                alignment = self.alignment()

                if alignment & Qt.AlignmentFlag.AlignRight:
                    text_x = content_rect.right() - text_width
                elif alignment & Qt.AlignmentFlag.AlignHCenter or alignment & Qt.AlignmentFlag.AlignCenter:
                    text_x = content_rect.x() + (content_rect.width() - text_width) / 2
                else:
                    text_x = content_rect.x()

                if alignment & Qt.AlignmentFlag.AlignBottom:
                    text_y = content_rect.bottom() - font_metrics.height()
                elif alignment & Qt.AlignmentFlag.AlignVCenter or alignment & Qt.AlignmentFlag.AlignCenter:
                    text_y = content_rect.y() + (content_rect.height() - font_metrics.height()) / 2
                else:
                    text_y = content_rect.y()

                gradient = QLinearGradient(text_x, text_y, text_x + text_width, text_y)
                num_colors = len(self._gradient_colors)
                for i, color in enumerate(self._gradient_colors):
                    position = i / (num_colors - 1) if num_colors > 1 else 0
                    gradient.setColorAt(position, QColor(color))

                path = QPainterPath()
                path.addText(text_x, text_y + font_metrics.ascent(), self.font(), text)

                brush = QBrush(gradient)
                painter.setBrush(brush)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPath(path)

                painter.end()

        title_label = CenteredGradientLabel(title)
        title_label.setFont(theme.get_font(size="moderate", weight="semibold", display=True))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setAutoFillBackground(False)
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        title_label.enable_gradient(colors=theme.config.text.gradient_colors, direction=Qt.Orientation.Horizontal)
        self.add(title_label)

        content_label = QLabel(content)
        content_label.setFont(theme.get_font(size="small", weight="regular"))
        content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        palette = content_label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.medium))
        content_label.setPalette(palette)
        content_label.setAutoFillBackground(False)
        content_label.setWordWrap(True)
        self.add(content_label, stretch=1)
