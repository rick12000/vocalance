import math

from PySide6.QtCore import QMetaObject, QRectF, Qt, QTimer, Slot
from PySide6.QtGui import QBrush, QColor, QPainter
from PySide6.QtWidgets import QWidget

from vocalance.app.ui.qt_theme import theme


class SpinnerWidget(QWidget):
    """Modern spinner with animated bouncing dots.

    Features:
    - Playful bouncing dot animation
    - Theme-consistent gradient colors
    - Non-blocking - runs on Qt's timer framework
    - Thread-safe start/stop methods
    - Smooth wave animation across dots
    """

    def __init__(self, parent: QWidget = None, size: int = 24):
        """Initialize spinner widget.

        Args:
            parent: Parent widget
            size: Height of the spinner in pixels (width auto-calculated)
        """
        super().__init__(parent)

        # Configuration
        self._dot_count = 4
        self._dot_size = 6
        self._dot_spacing = 4
        self._bounce_height = 8

        # Calculate dimensions
        total_width = (self._dot_count * self._dot_size) + ((self._dot_count - 1) * self._dot_spacing)
        self.setFixedSize(total_width, size)

        # Animation state
        self._phase = 0.0
        self._is_animating = False

        # Animation timer for smooth 60 FPS
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        self._timer.setInterval(16)  # ~60 FPS

        # Don't start automatically
        self.setVisible(False)

    def _update_animation(self) -> None:
        """Update animation phase for smooth bouncing."""
        self._phase = (self._phase + 0.15) % (2 * math.pi)
        self.update()  # Trigger repaint

    def start(self) -> None:
        """Start the spinner animation - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_start", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_start(self) -> None:
        """Internal start - MUST run on main Qt thread."""
        if not self._is_animating:
            self._is_animating = True
            self._phase = 0.0
            self.setVisible(True)
            self._timer.start()

    def stop(self) -> None:
        """Stop the spinner animation - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_stop", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_stop(self) -> None:
        """Internal stop - MUST run on main Qt thread."""
        if self._is_animating:
            self._is_animating = False
            self._timer.stop()
            self.setVisible(False)

    def paintEvent(self, event) -> None:
        """Draw the bouncing dots with gradient colors."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get gradient colors from theme
        gradient_colors = theme.config.text.gradient_colors
        color_start = QColor(gradient_colors[0])
        color_end = QColor(gradient_colors[1])

        # Calculate total width of all dots
        total_width = (self._dot_count * self._dot_size) + ((self._dot_count - 1) * self._dot_spacing)
        start_x = (self.width() - total_width) / 2
        center_y = self.height() / 2

        for i in range(self._dot_count):
            # Calculate wave offset for each dot (creates cascading effect)
            dot_phase = self._phase - (i * 0.5)

            # Bounce using sine wave (0 to 1 range, then scaled)
            bounce = math.sin(dot_phase)
            # Only bounce upward (positive values), with smooth easing
            if bounce > 0:
                y_offset = -bounce * self._bounce_height
            else:
                y_offset = 0

            # Calculate dot position
            x = start_x + (i * (self._dot_size + self._dot_spacing))
            y = center_y + y_offset

            # Interpolate color based on dot index (gradient across dots)
            t = i / (self._dot_count - 1) if self._dot_count > 1 else 0
            dot_color = QColor(
                int(color_start.red() + t * (color_end.red() - color_start.red())),
                int(color_start.green() + t * (color_end.green() - color_start.green())),
                int(color_start.blue() + t * (color_end.blue() - color_start.blue())),
            )

            # Add subtle scale effect based on bounce
            scale = 1.0 + (max(0, -y_offset / self._bounce_height) * 0.2)
            scaled_size = self._dot_size * scale

            # Draw the dot
            painter.setBrush(QBrush(dot_color))
            painter.setPen(Qt.PenStyle.NoPen)

            rect = QRectF(x - (scaled_size - self._dot_size) / 2, y - scaled_size / 2, scaled_size, scaled_size)
            painter.drawEllipse(rect)

    def showEvent(self, event) -> None:
        """Ensure timer is running when widget becomes visible."""
        super().showEvent(event)
        # showEvent is always called on main Qt thread
        if self._is_animating and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:
        """Pause animation when widget becomes hidden."""
        super().hideEvent(event)
        # hideEvent is always called on main Qt thread
        if self._timer.isActive():
            self._timer.stop()
