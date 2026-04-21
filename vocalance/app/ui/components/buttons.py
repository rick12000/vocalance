from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QByteArray, QEvent, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QPushButton, QWidget

from vocalance.app.ui.qt_theme import theme

_ICONS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "icons"


class _ThemedFlatPushButton(QPushButton):
    """Shared sizing, hover tracking, and flat styling for custom-painted push buttons."""

    def __init__(
        self,
        text: str,
        parent: Optional[QWidget],
        command: Optional[Callable[[], None]],
    ) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(theme.get_font("medium", "semibold", display=True))
        height = theme.config.components.button_height
        self.setFixedHeight(height)
        self._border_radius = height // 2
        self._is_hovered = False
        self._is_pressed = False
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFlat(True)
        self.setStyleSheet("")
        if command:
            self.clicked.connect(command)

    def enterEvent(self, enter_event: QEvent) -> None:
        self._is_hovered = True
        self.update()
        super().enterEvent(enter_event)

    def leaveEvent(self, leave_event: QEvent) -> None:
        self._is_hovered = False
        self.update()
        super().leaveEvent(leave_event)

    def mousePressEvent(self, press_event: QMouseEvent) -> None:
        self._is_pressed = True
        self.update()
        super().mousePressEvent(press_event)

    def mouseReleaseEvent(self, release_event: QMouseEvent) -> None:
        self._is_pressed = False
        self.update()
        super().mouseReleaseEvent(release_event)


class PrimaryButton(_ThemedFlatPushButton):
    """Primary action button with custom pill-shaped rendering."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        command: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(text, parent, command)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        border_width = 1

        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, rect.width() - 1, rect.height() - 1, self._border_radius, self._border_radius)

        accent = QColor(theme.config.blue.blue_2)
        accent.setAlpha(130)
        pen = QPen(accent, border_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.GlobalColor.transparent)
        painter.drawPath(border_path)

        painter.setPen(QColor(theme.config.blue.blue_2))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


class DangerButton(_ThemedFlatPushButton):
    """Danger-style button with transparent fill and light border."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        command: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(text, parent, command)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, self.width(), self.height(), self._border_radius, self._border_radius)
        painter.fillPath(bg_path, Qt.GlobalColor.transparent)

        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, self.width() - 1, self.height() - 1, self._border_radius, self._border_radius)

        border = QColor(theme.config.shapes.light)
        border.setAlpha(200)
        painter.setPen(QPen(border, 1.0))
        painter.drawPath(border_path)

        painter.setPen(QColor(theme.config.text.light))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


class GhostButton(QPushButton):
    """Ghost button using stylesheet hover states."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        command: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(text, parent)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(theme.get_font("medium", "semibold", display=True))
        height = theme.config.components.button_height
        self.setFixedHeight(height)
        border_radius = height // 2

        self.setStyleSheet(
            f"""
            GhostButton {{
                background-color: transparent;
                color: {theme.config.text.light};
                border: none;
                border-radius: {border_radius}px;
                padding: 2px 16px;
            }}
            GhostButton:hover {{
                background-color: {theme.config.shapes.light};
            }}
            GhostButton:pressed {{
                background-color: {theme.config.shapes.medium};
            }}
            GhostButton:disabled {{
                color: {theme.config.text.medium};
            }}
        """
        )

        if command:
            self.clicked.connect(command)


def _create_recolored_renderer(svg_path: str, color: str) -> QSvgRenderer:
    """Load SVG from ``svg_path``, replace default fill with ``color``, return a renderer."""
    svg_file = Path(svg_path)
    if not svg_file.exists():
        return QSvgRenderer()

    svg_content = svg_file.read_text(encoding="utf-8")
    svg_content = svg_content.replace('fill="#e3e3e3"', f'fill="{color}"')
    svg_content = svg_content.replace('fill="#E3E3E3"', f'fill="{color}"')

    return QSvgRenderer(QByteArray(svg_content.encode("utf-8")))


def _centered_icon_target_rect(width: int, height: int) -> QRectF:
    icon_dim = int(width * 0.6)
    x_pos = (width - icon_dim) / 2.0
    y_pos = (height - icon_dim) / 2.0
    return QRectF(x_pos, y_pos, icon_dim, icon_dim)


def _render_centered_icon(painter: QPainter, renderer: QSvgRenderer, width: int, height: int) -> None:
    if renderer.isValid():
        renderer.render(painter, _centered_icon_target_rect(width, height))


class ChangeButton(PrimaryButton):
    """Circular icon button with add glyph."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        command: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__("", parent, command)

        button_size = theme.config.components.button_height
        self.setFixedSize(button_size, button_size)
        self.setContentsMargins(0, 0, 0, 0)
        self._border_radius = button_size // 2

        icon_path = _ICONS_DIR / "add_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.svg"
        self._renderer = _create_recolored_renderer(str(icon_path), theme.config.text.light)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        width = rect.width()
        height = rect.height()

        path = QPainterPath()
        path.addRoundedRect(0, 0, width, height, self._border_radius, self._border_radius)

        if self._is_pressed:
            bg_color = QColor(theme.config.shapes.medium)
        elif self._is_hovered:
            bg_color = QColor(theme.config.shapes.light)
        else:
            bg_color = QColor(theme.config.shapes.medium)

        painter.fillPath(path, bg_color)
        _render_centered_icon(painter, self._renderer, width, height)


class DeleteButton(DangerButton):
    """Circular icon button with delete glyph."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        command: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__("", parent, command)

        button_size = theme.config.components.button_height
        self.setFixedSize(button_size, button_size)
        self.setContentsMargins(0, 0, 0, 0)
        self._border_radius = button_size // 2

        icon_path = _ICONS_DIR / "delete_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.svg"
        self._renderer = _create_recolored_renderer(str(icon_path), theme.config.shapes.accent)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        width = rect.width()
        height = rect.height()

        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, width, height, self._border_radius, self._border_radius)
        painter.fillPath(bg_path, Qt.GlobalColor.transparent)

        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, width - 1, height - 1, self._border_radius, self._border_radius)

        b = QColor(theme.config.shapes.light)
        b.setAlpha(200)
        painter.setPen(QPen(b, 1.0))
        painter.drawPath(border_path)

        _render_centered_icon(painter, self._renderer, width, height)


class ExpandButton(PrimaryButton):
    """Circular expand (chevron right) control."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        command: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__("", parent, command)

        button_size = theme.config.components.button_height
        self.setFixedSize(button_size, button_size)
        self.setContentsMargins(0, 0, 0, 0)
        self._border_radius = button_size // 2

        icon_path = _ICONS_DIR / "keyboard_arrow_right_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.svg"
        self._renderer = _create_recolored_renderer(str(icon_path), theme.config.shapes.light)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        width = rect.width()
        height = rect.height()

        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, width, height, self._border_radius, self._border_radius)
        painter.fillPath(bg_path, Qt.GlobalColor.transparent)

        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, width - 1, height - 1, self._border_radius, self._border_radius)

        e = QColor(theme.config.shapes.light)
        e.setAlpha(180)
        painter.setPen(QPen(e, 1.0))
        painter.drawPath(border_path)

        _render_centered_icon(painter, self._renderer, width, height)


class CollapseButton(PrimaryButton):
    """Circular collapse (chevron down) control."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        command: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__("", parent, command)

        button_size = theme.config.components.button_height
        self.setFixedSize(button_size, button_size)
        self.setContentsMargins(0, 0, 0, 0)
        self._border_radius = button_size // 2

        icon_path = _ICONS_DIR / "keyboard_arrow_down_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.svg"
        self._renderer = _create_recolored_renderer(str(icon_path), theme.config.shapes.light)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        width = rect.width()
        height = rect.height()

        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, width, height, self._border_radius, self._border_radius)
        painter.fillPath(bg_path, Qt.GlobalColor.transparent)

        border_path = QPainterPath()
        border_path.addRoundedRect(0.5, 0.5, width - 1, height - 1, self._border_radius, self._border_radius)

        c = QColor(theme.config.shapes.light)
        c.setAlpha(180)
        painter.setPen(QPen(c, 1.0))
        painter.drawPath(border_path)

        _render_centered_icon(painter, self._renderer, width, height)
