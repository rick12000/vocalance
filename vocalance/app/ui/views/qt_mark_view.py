import logging
from typing import Dict, List, Tuple

from PySide6.QtCore import QPoint, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.events.mark_events import MarkData
from vocalance.app.ui.qt_theme import theme


class QtMarkView(QWidget):
    """Thread-safe mark visualization overlay for PySide6.

    Key principles:
    - All GUI operations marshalled to main Qt thread via signals
    - Simple keyboard handling without complex event filters
    - Window state managed explicitly
    - Deferred focus to avoid blocking
    """

    # Signals for marshalling from async threads
    show_requested = Signal()
    hide_requested = Signal()

    def __init__(self, mark_service, config: GlobalAppConfig):
        """Initialize mark view.

        Args:
            mark_service: Mark service instance.
            config: Global app configuration.
        """
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.mark_service = mark_service
        self.config = config

        # Mark data
        self.marks: Dict[str, Tuple[int, int]] = {}
        self._is_active = False

        # Setup window as frameless overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Don't use setWindowOpacity - it affects all content including circles
        # Instead, we'll paint a semi-transparent background in paintEvent

        # Visual properties
        # Background - semi-transparent for overlay effect
        self.background_color = QColor(theme.config.shapes.dark)
        self.background_color.setAlpha(204)  # 80% opacity (204/255)

        # Mark elements - fully opaque for consistent appearance
        self.mark_fill_color = QColor(theme.config.shapes.darkest)
        self.mark_fill_color.setAlpha(255)  # 100% opacity
        self.mark_border_color = QColor(theme.config.blue.blue_2)
        self.mark_border_color.setAlpha(255)  # 100% opacity
        self.text_color = QColor(theme.config.text.medium)
        self.font = QFont(theme.config.font_family_primary, theme.config.fonts.small, QFont.Weight.DemiBold)

        # Controller callback
        self.controller_callback = None

        # For deferred focus setting
        self._focus_timer = None

        # Connect signals to slots for thread-safe operations
        self.show_requested.connect(self._on_show_requested)
        self.hide_requested.connect(self._on_hide_requested)

        self.logger.info("QtMarkView initialized")

    def set_controller_callback(self, callback) -> None:
        """Set the controller callback."""
        self.controller_callback = callback

    @Slot()
    def _on_show_requested(self) -> None:
        """Handle show request from signal - runs on Qt main thread."""
        self.logger.info("Show requested signal received on Qt main thread")
        self._do_show_direct()

    @Slot()
    def _on_hide_requested(self) -> None:
        """Handle hide request from signal - runs on Qt main thread."""
        self.logger.info("Hide requested signal received on Qt main thread")
        self._do_hide_direct()

    def _do_show_direct(self) -> None:
        """Show the overlay directly - controller calls this after ensuring marks are loaded."""
        if self._is_active:
            self.logger.warning("Overlay already active")
            return

        try:
            self.logger.info(f"Direct show: {len(self.marks)} marks ready")
            self.logger.info(f"  self.marks dict content: {self.marks}")

            # CRITICAL: If marks are still empty, defer showing briefly to allow
            # controller to set them. This handles race condition in signal delivery.
            if not self.marks:
                self.logger.warning(
                    "Direct show called with empty marks dict, deferring 50ms to allow controller to populate marks"
                )
                QTimer.singleShot(50, self._do_show_direct)
                return

            # Get PRIMARY screen geometry with detailed diagnostics
            primary = QApplication.primaryScreen()
            if primary:
                geometry = primary.geometry()
                available = primary.availableGeometry()
                virtual = primary.virtualGeometry()

                self.logger.info(
                    f"  Primary screen geometry: ({geometry.x()}, {geometry.y()}) {geometry.width()}x{geometry.height()}"
                )
                self.logger.info(
                    f"  Primary available geometry: ({available.x()}, {available.y()}) {available.width()}x{available.height()}"
                )
                self.logger.info(
                    f"  Primary virtual geometry: ({virtual.x()}, {virtual.y()}) {virtual.width()}x{virtual.height()}"
                )

                # Use full geometry (not just available, which excludes taskbar)
                self.setGeometry(geometry)
                self.logger.info(f"  Set overlay to: ({geometry.x()}, {geometry.y()}) {geometry.width()}x{geometry.height()}")
            else:
                # Fallback geometry
                self.logger.warning("  No primary screen found, using fallback geometry")
                self.setGeometry(0, 0, 1920, 1080)

            # Show and configure
            super().show()
            self.raise_()
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.activateWindow()  # Activate window to bring to foreground
            self.setFocus()  # Set focus IMMEDIATELY on main thread

            self._is_active = True
            self.logger.info(f"  Calling update() to trigger paintEvent with {len(self.marks)} marks...")
            self.update()  # Trigger repaint with marks

            # Schedule another focus attempt shortly after to ensure it sticks
            QTimer.singleShot(10, self._ensure_focus)

            # Notify controller asynchronously
            if self.controller_callback:
                QTimer.singleShot(0, lambda: self.controller_callback.on_mark_visualization_shown())

            self.logger.info(f"Overlay displayed with focus set immediately, marks: {self.marks}")

        except Exception as e:
            self.logger.error(f"Error showing overlay: {e}", exc_info=True)
            if self.controller_callback:
                error_msg = str(e)
                QTimer.singleShot(0, lambda: self.controller_callback.on_mark_visualization_failed(error_msg))

    def _ensure_focus(self) -> None:
        """Ensure focus is maintained after show."""
        if self._is_active and not self.isHidden():
            self.setFocus()
            self.logger.debug("Focus re-asserted on overlay")

    def _do_hide_direct(self) -> None:
        """Hide the overlay directly."""
        if not self._is_active:
            return

        try:
            self.logger.info("Direct hide")

            # Cancel any pending focus timer
            if self._focus_timer:
                self._focus_timer.stop()
                self._focus_timer = None

            self.clearFocus()
            super().hide()
            self._is_active = False

            # Notify controller asynchronously
            if self.controller_callback:
                QTimer.singleShot(0, lambda: self.controller_callback.on_mark_visualization_hidden())

            self.logger.info("Overlay hidden")

        except Exception as e:
            self.logger.error(f"Error hiding overlay: {e}", exc_info=True)

    def update_marks(self, marks_list: List[MarkData]) -> None:
        """Update marks data - thread-safe."""
        QTimer.singleShot(0, lambda: self._do_update_marks(marks_list))

    def _do_update_marks(self, marks_list: List[MarkData]) -> None:
        """Internal update marks - MUST run on main Qt thread."""
        self.logger.info(f"_do_update_marks called with {len(marks_list)} MarkData objects")
        for idx, mark in enumerate(marks_list):
            self.logger.info(f"  marks_list[{idx}]: name='{mark.name}', x={mark.x}, y={mark.y}")

        self.marks = {mark.name: (mark.x, mark.y) for mark in marks_list}
        self.logger.info(f"Updated {len(self.marks)} marks in view: {self.marks}")
        for name, (x, y) in self.marks.items():
            self.logger.debug(f"  Mark '{name}' at ({x}, {y})")
        if self._is_active:
            self.update()

    def update_marks_dict(self, marks: Dict[str, Tuple[int, int]]) -> None:
        """Update marks from dictionary - thread-safe."""
        QTimer.singleShot(0, lambda: self._do_update_marks_dict(marks))

    def _do_update_marks_dict(self, marks: Dict[str, Tuple[int, int]]) -> None:
        """Internal update marks dict - MUST run on main Qt thread."""
        self.marks = marks.copy()
        self.logger.info(f"Updated {len(self.marks)} marks from dict")
        if self._is_active:
            self.update()

    def paintEvent(self, event) -> None:
        """Paint marks on screen with semi-transparent background and fully opaque marks."""
        self.logger.info(f"paintEvent called: active={self._is_active}, marks_count={len(self.marks)}")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw semi-transparent background first
        painter.fillRect(self.rect(), self.background_color)

        if len(self.marks) == 0:
            self.logger.warning(f"paintEvent: No marks to paint (self.marks={self.marks})")
            return

        self.logger.info(f"paintEvent: Drawing {len(self.marks)} marks")
        self.logger.info(f"  self.marks content: {self.marks}")

        # Get device pixel ratio for DPI scaling conversion
        primary = QApplication.primaryScreen()
        device_pixel_ratio = primary.devicePixelRatio() if primary else 1.0

        # Get overlay position to offset coordinates (in logical pixels)
        overlay_geometry = self.geometry()
        overlay_x = overlay_geometry.x()
        overlay_y = overlay_geometry.y()
        self.logger.info(f"  Overlay position (logical): ({overlay_x}, {overlay_y}), size: ({self.width()}, {self.height()})")
        self.logger.info(f"  Device pixel ratio: {device_pixel_ratio}")

        # Draw each mark - following legacy approach: draw ALL marks at their absolute coordinates
        # Mark coordinates are in PHYSICAL pixels (from pyautogui), convert to LOGICAL pixels for Qt painting
        for label, (x, y) in self.marks.items():
            try:
                # Convert physical pixels to logical pixels
                logical_x = x / device_pixel_ratio
                logical_y = y / device_pixel_ratio

                # Convert absolute logical screen coordinates to overlay-relative logical coordinates
                relative_x = logical_x - overlay_x
                relative_y = logical_y - overlay_y

                # Draw mark circle with solid border and fill
                radius = 4
                border_width = 2

                # Draw circle with solid border (blue_2) and solid fill (darkest)
                painter.setPen(QPen(self.mark_border_color, border_width))
                painter.setBrush(self.mark_fill_color)
                painter.drawEllipse(QPoint(int(relative_x), int(relative_y)), radius, radius)

                # Draw mark label
                painter.setPen(QPen(self.text_color))
                painter.setFont(self.font)
                painter.drawText(int(relative_x) + 10, int(relative_y) - 10, label)

                self.logger.info(
                    f"  Drew mark '{label}': physical=({x}, {y}) -> logical=({logical_x:.1f}, {logical_y:.1f}) -> relative=({relative_x:.1f}, {relative_y:.1f})"
                )
            except Exception as e:
                self.logger.error(f"  Error drawing mark '{label}': {e}", exc_info=True)

        self.logger.info(f"Completed painting {len(self.marks)} marks")

    def keyPressEvent(self, event) -> None:
        """Handle Escape key to close overlay."""
        if event.key() == Qt.Key.Key_Escape:
            # Schedule hide on next event loop iteration to avoid blocking
            QTimer.singleShot(0, self._do_hide_direct)
            event.accept()
        else:
            super().keyPressEvent(event)

    def show(self) -> None:
        """Show the overlay - calls signal for thread safety."""
        self.show_requested.emit()

    def hide(self) -> None:
        """Hide the overlay - calls signal for thread safety."""
        self.hide_requested.emit()

    def toggle(self) -> None:
        """Toggle mark overlay visibility."""
        if self._is_active:
            self.hide()
        else:
            self.show()

    def is_active(self) -> bool:
        """Check if overlay is active."""
        return self._is_active and self.isVisible()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.hide()
        self.marks.clear()
        if self._focus_timer:
            self._focus_timer.stop()
            self._focus_timer = None
        self.logger.debug("QtMarkView cleanup completed")
