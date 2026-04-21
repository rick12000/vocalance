import logging
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPoint, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QWidget

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.events.mark_events import MarkData
from vocalance.app.ui.qt_theme import theme


class QtMarkView(QWidget):
    """Fullscreen translucent overlay that draws mark positions and labels."""

    show_requested = Signal()
    hide_requested = Signal()

    def __init__(self, config: GlobalAppConfig) -> None:
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        self.marks: Dict[str, Tuple[int, int]] = {}
        self.overlay_active: bool = False

        self.focus_timers: List[QTimer] = []

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.background_color = QColor(theme.config.shapes.dark)
        self.background_color.setAlpha(204)

        self.mark_fill_color = QColor(theme.config.shapes.accent)
        self.mark_fill_color.setAlpha(255)
        self.mark_border_color = QColor(theme.config.blue.blue_2)
        self.mark_border_color.setAlpha(255)
        self.text_color = QColor(theme.config.text.medium)
        self.font = QFont(theme.config.font_family_primary, theme.config.fonts.small, QFont.Weight.DemiBold)

        self.marks_controller: Optional[object] = None

        self.show_requested.connect(self.on_show_requested)
        self.hide_requested.connect(self.on_hide_requested)

        self.logger.info("QtMarkView initialized")

    def bind_controller(self, controller) -> None:
        """Attach the marks controller for overlay lifecycle callbacks."""
        self.marks_controller = controller

    @Slot()
    def on_show_requested(self) -> None:
        self.logger.info("Show requested signal received on Qt main thread")
        self.do_show_direct()

    @Slot()
    def on_hide_requested(self) -> None:
        self.logger.info("Hide requested signal received on Qt main thread")
        self.do_hide_direct()

    def do_show_direct(self) -> None:
        if self.overlay_active:
            self.logger.warning("Overlay already active")
            return

        self.logger.info("Direct show: %s marks ready", len(self.marks))

        primary = QApplication.primaryScreen()
        if primary:
            geometry = primary.geometry()
            self.setGeometry(geometry)
            self.logger.info(
                "Overlay geometry set to primary screen %sx%s at (%s,%s)",
                geometry.width(),
                geometry.height(),
                geometry.x(),
                geometry.y(),
            )
        else:
            self.logger.warning("No primary screen found, using fallback geometry")
            self.setGeometry(0, 0, 1920, 1080)

        super().show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

        self.overlay_active = True
        self.update()

        self.schedule_robust_focus()

        ctrl = self.marks_controller
        if ctrl is not None:
            QTimer.singleShot(0, ctrl.request_show_overlay)

        self.logger.info("Overlay displayed, marks: %s", len(self.marks))

    def schedule_robust_focus(self) -> None:
        """Queue several focus attempts to work around OS focus stealing."""
        self.cancel_focus_timers()

        delays_ms: Tuple[int, ...] = (10, 50, 100, 200)

        for delay in delays_ms:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.ensure_focus)
            timer.start(delay)
            self.focus_timers.append(timer)

        self.logger.debug("Scheduled %s focus attempts: %s ms", len(delays_ms), delays_ms)

    def cancel_focus_timers(self) -> None:
        for timer in self.focus_timers:
            if timer.isActive():
                timer.stop()
            timer.deleteLater()
        self.focus_timers.clear()

    def ensure_focus(self) -> None:
        if self.overlay_active and not self.isHidden():
            if not self.hasFocus():
                self.raise_()
                self.activateWindow()
                self.setFocus()
                self.logger.debug("Focus asserted on overlay")
            else:
                self.logger.debug("Overlay already has focus")

    def do_hide_direct(self) -> None:
        if not self.overlay_active:
            return

        self.logger.info("Direct hide")

        self.cancel_focus_timers()

        self.clearFocus()
        super().hide()
        self.overlay_active = False

        ctrl = self.marks_controller
        if ctrl is not None:
            QTimer.singleShot(0, ctrl.request_hide_overlay)

        self.logger.info("Overlay hidden")

    def update_marks(self, marks_list: List[MarkData]) -> None:
        QTimer.singleShot(0, lambda: self.do_update_marks(marks_list))

    def do_update_marks(self, marks_list: List[MarkData]) -> None:
        self.marks = {mark.name: (mark.x, mark.y) for mark in marks_list}
        self.logger.info("Updated %s marks in view", len(self.marks))
        if self.overlay_active:
            self.update()

    def update_marks_dict(self, marks: Dict[str, Tuple[int, int]]) -> None:
        QTimer.singleShot(0, lambda: self.do_update_marks_dict(marks))

    def do_update_marks_dict(self, marks: Dict[str, Tuple[int, int]]) -> None:
        self.marks = marks.copy()
        self.logger.info("Updated %s marks from dict", len(self.marks))
        if self.overlay_active:
            self.update()

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        del paint_event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), self.background_color)

        if len(self.marks) == 0:
            return

        primary = QApplication.primaryScreen()
        device_pixel_ratio: float = float(primary.devicePixelRatio()) if primary else 1.0

        overlay_geometry = self.geometry()
        overlay_x: int = overlay_geometry.x()
        overlay_y: int = overlay_geometry.y()

        for label, (x, y) in self.marks.items():
            logical_x: float = x / device_pixel_ratio
            logical_y: float = y / device_pixel_ratio
            relative_x: float = logical_x - overlay_x
            relative_y: float = logical_y - overlay_y

            radius: int = 4
            border_width: int = 2

            painter.setPen(QPen(self.mark_border_color, border_width))
            painter.setBrush(self.mark_fill_color)
            painter.drawEllipse(QPoint(int(relative_x), int(relative_y)), radius, radius)

            painter.setPen(QPen(self.text_color))
            painter.setFont(self.font)
            painter.drawText(int(relative_x) + 10, int(relative_y) - 10, label)

    def keyPressEvent(self, key_event: QKeyEvent) -> None:
        if key_event.key() == Qt.Key.Key_Escape:
            QTimer.singleShot(0, self.do_hide_direct)
            key_event.accept()
        else:
            super().keyPressEvent(key_event)

    def show(self) -> None:
        self.show_requested.emit()

    def hide(self) -> None:
        self.hide_requested.emit()

    def toggle(self) -> None:
        if self.overlay_active:
            self.hide()
        else:
            self.show()

    def is_active(self) -> bool:
        return self.overlay_active and self.isVisible()

    def cleanup(self) -> None:
        self.cancel_focus_timers()
        self.hide()
        self.marks.clear()
        self.logger.debug("QtMarkView cleanup completed")
