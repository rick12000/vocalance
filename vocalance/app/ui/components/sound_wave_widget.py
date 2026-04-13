import math
from typing import Optional

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from vocalance.app.ui.qt_theme import theme


def _create_gradient(width: int, colors: list) -> QLinearGradient:
    """Create smooth linear gradient with blended transitions."""
    gradient = QLinearGradient(0, 0, width, 0)
    if not colors:
        return gradient

    num = len(colors)
    if num == 1:
        gradient.setColorAt(0, QColor(colors[0]))
        return gradient

    # For 2+ colors, add intermediate stops for smoother blending
    stops_per_color = 5

    for i, color in enumerate(colors):
        pos = i / (num - 1)
        gradient.setColorAt(pos, QColor(color))

        if i < num - 1:
            next_color = QColor(colors[i + 1])
            current = QColor(color)

            for step in range(1, stops_per_color):
                blend_pos = pos + (1 / (num - 1)) * (step / stops_per_color)
                blend_ratio = step / stops_per_color

                blended = QColor(
                    int(current.red() * (1 - blend_ratio) + next_color.red() * blend_ratio),
                    int(current.green() * (1 - blend_ratio) + next_color.green() * blend_ratio),
                    int(current.blue() * (1 - blend_ratio) + next_color.blue() * blend_ratio),
                )
                gradient.setColorAt(blend_pos, blended)

    return gradient


class SoundWaveWidget(QWidget):
    """Widget that displays animated sound waves based on audio levels.

    Features:
    - Smooth animation using interpolation
    - Responsive to audio amplitude
    - Modern aesthetic with rounded bars
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Visualizer config
        self.bar_count = 5
        self.bar_spacing = 4  # Reduced spacing for compact look
        self.bar_width = 4  # Smaller bars
        self.base_height = 1
        self.vertical_padding = 2  # Reduced padding

        # Calculate compact window dimensions
        total_bar_width = (self.bar_count * self.bar_width) + ((self.bar_count - 1) * self.bar_spacing)
        horizontal_padding = 4  # Minimal padding left/right
        window_width = total_bar_width + (horizontal_padding * 2)

        # Much smaller window: height ~33px (2/3 reduction from 100)
        max_window_height = 20
        self.max_height = max_window_height - (self.vertical_padding * 2)

        # Set fixed size - compact tight fit
        self.setFixedSize(window_width, max_window_height)

        # Animation state
        self.target_amplitude = 0.0
        self.current_amplitude = 0.0
        self.phase = 0.0

        # Animation timer (60 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(16)  # ~60 FPS

    def update_level(self, level: float) -> None:
        """Update target amplitude based on audio level (0.0 - 1.0)."""
        # More sensitive: use sqrt for faster response at lower volumes
        if level > 0.01:
            # Square root makes small values larger, so bars reach max with less audio
            self.target_amplitude = min(1.0, math.sqrt(level * 2.0))
        else:
            self.target_amplitude = 0.0

    def _animate(self) -> None:
        """Update animation state."""
        # Smoothly interpolate current amplitude towards target
        # Attack is fast, decay is slower
        if self.target_amplitude > self.current_amplitude:
            self.current_amplitude += (self.target_amplitude - self.current_amplitude) * 0.2
        else:
            self.current_amplitude += (self.target_amplitude - self.current_amplitude) * 0.05

        self.phase += 0.15
        if self.phase > math.pi * 2:
            self.phase -= math.pi * 2

        self.update()

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        """Draw the sound waves with gradient."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate horizontal centering
        total_width = (self.bar_count * self.bar_width) + ((self.bar_count - 1) * self.bar_spacing)
        start_x = (self.width() - total_width) / 2

        # Vertical center
        center_y = self.height() / 2

        # Create horizontal gradient from theme colors
        gradient = _create_gradient(int(total_width), theme.config.text.gradient_colors)
        # Adjust gradient to start position
        gradient.setStart(start_x, 0)
        gradient.setFinalStop(start_x + total_width, 0)

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)

        for i in range(self.bar_count):
            # Idle animation (breathing effect)
            idle_offset = math.sin(self.phase + (i * 0.5)) * 0.2 + 0.5  # 0.3 to 0.7 range

            # Active animation logic - center bars react more strongly
            dist_from_center = abs(i - (self.bar_count - 1) / 2)
            scale_factor = 1.0 - (dist_from_center * 0.15)

            # Combine idle and active with much stronger amplitude response
            active_height = self.current_amplitude * self.max_height * scale_factor

            # Minimum height + idle movement when quiet
            # When active, idle movement is suppressed
            idle_height = self.base_height + (2 * idle_offset * (1.0 - self.current_amplitude * 0.8))

            height = idle_height + active_height

            # Cap height to not exceed max_height
            height = min(height, self.max_height)

            # Draw rounded rect centered vertically
            x = start_x + (i * (self.bar_width + self.bar_spacing))
            y = center_y - (height / 2)

            painter.drawRoundedRect(QRectF(x, y, self.bar_width, height), self.bar_width / 2, self.bar_width / 2)
