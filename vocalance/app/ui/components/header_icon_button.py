from typing import Optional

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPalette, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QWidget

from vocalance.app.ui.components.icon_widget import IconWidget, tint_pixmap
from vocalance.app.ui.components.labels import BodyLabel
from vocalance.app.ui.qt_theme import theme


class HeaderIconButton(QWidget):
    """Icon button with text that expands left on hover.

    Similar to sidebar buttons but text expands right-to-left.
    No background, icon and text in blue_1 color.
    """

    clicked = Signal()

    def __init__(
        self,
        text: str,
        icon_pixmap: Optional[QPixmap] = None,
        text_icon_spacing: Optional[int] = None,
        icon_size: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._text_content = text
        self._default_icon = icon_pixmap
        self._text_icon_spacing = text_icon_spacing if text_icon_spacing is not None else theme.config.spacing.medium

        self._icon_color = theme.config.blue.blue_2
        self._text_color = theme.config.blue.blue_2

        self._icon_size = icon_size if icon_size is not None else 40
        self._button_padding = 8

        self.setAutoFillBackground(False)

        self._setup_ui(icon_pixmap)

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._setup_animation()

    def _setup_ui(self, icon_pixmap: Optional[QPixmap]) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(self._button_padding, self._button_padding, self._button_padding, self._button_padding)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.text_label = BodyLabel(self._text_content)
        font = theme.get_font(size="medium", weight="regular", display=True)
        self.text_label.setFont(font)

        palette = self.text_label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(self._text_color))
        self.text_label.setPalette(palette)

        self.text_label.setMaximumWidth(0)
        layout.addWidget(self.text_label)

        self.spacer = QWidget()
        self.spacer.setMinimumWidth(0)
        self.spacer.setMaximumWidth(0)
        self.spacer.setFixedHeight(1)
        layout.addWidget(self.spacer)

        if icon_pixmap:
            colored_icon = tint_pixmap(icon_pixmap, self._icon_color)

            self.icon_widget = IconWidget(colored_icon, self._icon_size)
            layout.addWidget(self.icon_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            self.icon_widget = None

    def _setup_animation(self) -> None:
        self._text_anim = QPropertyAnimation(self.text_label, b"maximumWidth")
        self._text_anim.setDuration(260)
        self._text_anim.setEasingCurve(QEasingCurve.Type.OutQuart)

        self._spacer_anim = QPropertyAnimation(self.spacer, b"maximumWidth")
        self._spacer_anim.setDuration(260)
        self._spacer_anim.setEasingCurve(QEasingCurve.Type.OutQuart)

    def _animate_expansion(self, expand: bool) -> None:
        if expand:
            fm = self.text_label.fontMetrics()
            text_width = fm.horizontalAdvance(self._text_content) + 10

            self._text_anim.setStartValue(self.text_label.maximumWidth())
            self._text_anim.setEndValue(text_width)
            self._text_anim.start()

            self._spacer_anim.setStartValue(self.spacer.width())
            self._spacer_anim.setEndValue(self._text_icon_spacing)
            self._spacer_anim.start()
        else:
            self._text_anim.setStartValue(self.text_label.maximumWidth())
            self._text_anim.setEndValue(0)
            self._text_anim.start()

            self._spacer_anim.setStartValue(self.spacer.width())
            self._spacer_anim.setEndValue(0)
            self._spacer_anim.start()

    def enterEvent(self, enter_event: QEvent) -> None:
        self._animate_expansion(True)
        super().enterEvent(enter_event)

    def leaveEvent(self, leave_event: QEvent) -> None:
        self._animate_expansion(False)
        super().leaveEvent(leave_event)

    def mousePressEvent(self, press_event: QMouseEvent) -> None:
        if press_event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(press_event)
