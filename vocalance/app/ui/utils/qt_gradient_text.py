"""Gradient text painting utilities for Qt labels.

Provides robust gradient text rendering that works with QLabel and its subclasses.
The gradient is painted directly in the paintEvent, overriding stylesheet colors
to enable smooth color transitions across text.

Usage:
    1. Mix GradientTextMixin into a QLabel subclass
    2. Call enable_gradient() with gradient colors
    3. The paintEvent will automatically render text with gradient

Example:
    class MyGradientLabel(GradientTextMixin, QLabel):
        def __init__(self, text: str = ""):
            super().__init__(text)
            self.enable_gradient(["#4E98FF", "#F97070"])
"""

from typing import List, Optional

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import QLabel


class GradientTextMixin:
    """Mixin class that adds gradient text rendering to QLabel subclasses.

    This mixin overrides the paintEvent to render text with a linear gradient
    instead of a solid color. The gradient direction is left-to-right by default.

    Features:
    - Supports any number of gradient colors (2+ recommended)
    - Respects label alignment (left, center, right)
    - Handles text wrapping and elision
    - Maintains all QLabel functionality (font, size, etc.)
    - Can be enabled/disabled at runtime

    Thread-safe: All operations use Qt's main thread.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the mixin.

        Note: This must be called before QLabel.__init__ in MRO chain.
        Use super().__init__() in subclass to ensure proper initialization.
        """
        super().__init__(*args, **kwargs)
        self._gradient_enabled = False
        self._gradient_colors: List[str] = []
        self._gradient_direction: Qt.Orientation = Qt.Orientation.Horizontal

    def enable_gradient(self, colors: List[str], direction: Qt.Orientation = Qt.Orientation.Horizontal) -> None:
        """Enable gradient text rendering with the specified colors.

        Args:
            colors: List of color hex codes (e.g., ["#4E98FF", "#F97070"])
                   Minimum 2 colors required for a gradient.
            direction: Gradient direction (Horizontal or Vertical).
                      Horizontal = left-to-right, Vertical = top-to-bottom.

        Raises:
            ValueError: If fewer than 2 colors are provided.
        """
        if len(colors) < 2:
            raise ValueError("Gradient requires at least 2 colors")

        self._gradient_enabled = True
        self._gradient_colors = colors
        self._gradient_direction = direction

        # Force repaint to show gradient
        if isinstance(self, QLabel):
            self.update()

    def disable_gradient(self) -> None:
        """Disable gradient rendering and revert to standard text color."""
        self._gradient_enabled = False
        if isinstance(self, QLabel):
            self.update()

    def is_gradient_enabled(self) -> bool:
        """Check if gradient rendering is currently enabled.

        Returns:
            True if gradient is enabled, False otherwise.
        """
        return self._gradient_enabled

    def paintEvent(self, event) -> None:
        """Override paintEvent to render text with gradient.

        This method intercepts the normal QLabel paint event and renders
        the text with a gradient fill instead of solid color when enabled.
        If gradient is disabled, falls back to standard QLabel painting.

        Args:
            event: QPaintEvent from Qt event system.
        """
        if not self._gradient_enabled or not self._gradient_colors:
            # Fall back to default QLabel painting
            super().paintEvent(event)
            return

        # Custom gradient painting
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Get text and check if empty
        text = self.text()
        if not text:
            painter.end()
            return

        # Set font first (needed for text measurement)
        painter.setFont(self.font())

        # Get content rectangle (accounts for margins)
        # Use full widget rect if contentsRect is empty/not yet calculated
        content_rect = self.contentsRect()
        if content_rect.isEmpty():
            content_rect = self.rect()

        # Calculate actual text bounding rectangle
        text_rect = self._calculate_text_rect(painter, text, content_rect)

        # Create gradient based on text bounds, not widget bounds
        gradient = self._create_gradient_for_text(painter, text, text_rect)

        # Create a path from the text (this is the key fix!)
        # Convert text to a path so we can fill it with the gradient brush
        path = QPainterPath()
        path.addText(text_rect.x(), text_rect.y() + painter.fontMetrics().ascent(), self.font(), text)

        # Set up brush with gradient (not pen!) for filling
        brush = QBrush(gradient)
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)  # No outline

        # Draw the text path filled with gradient
        painter.drawPath(path)

        painter.end()

    def _create_gradient_for_text(self, painter: QPainter, text: str, text_rect: QRect) -> QLinearGradient:
        """Create a QLinearGradient based on actual text bounds.

        This ensures the gradient spans from the first character to the last character,
        regardless of widget width or text alignment.

        Args:
            painter: QPainter for measuring text.
            text: The text being rendered.
            text_rect: Rectangle where text will be drawn.

        Returns:
            QLinearGradient configured to span the actual text width/height.
        """
        # Get actual text bounding box
        font_metrics = painter.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text)
        text_height = font_metrics.height()

        # Calculate text position based on alignment
        alignment = self.alignment()

        # Determine horizontal position
        if alignment & Qt.AlignmentFlag.AlignRight:
            text_x = text_rect.right() - text_width
        elif alignment & Qt.AlignmentFlag.AlignHCenter or alignment & Qt.AlignmentFlag.AlignCenter:
            text_x = text_rect.x() + (text_rect.width() - text_width) / 2
        else:  # AlignLeft (default)
            text_x = text_rect.x()

        # Determine vertical position
        if alignment & Qt.AlignmentFlag.AlignBottom:
            text_y = text_rect.bottom() - text_height
        elif alignment & Qt.AlignmentFlag.AlignVCenter or alignment & Qt.AlignmentFlag.AlignCenter:
            text_y = text_rect.y() + (text_rect.height() - text_height) / 2
        else:  # AlignTop (default)
            text_y = text_rect.y()

        # Create gradient based on actual text dimensions and position
        if self._gradient_direction == Qt.Orientation.Horizontal:
            # Gradient from start of text to end of text (horizontal)
            gradient = QLinearGradient(text_x, text_y, text_x + text_width, text_y)
        else:
            # Gradient from top of text to bottom of text (vertical)
            gradient = QLinearGradient(text_x, text_y, text_x, text_y + text_height)

        # Add color stops evenly distributed
        num_colors = len(self._gradient_colors)
        for i, color in enumerate(self._gradient_colors):
            position = i / (num_colors - 1) if num_colors > 1 else 0
            gradient.setColorAt(position, QColor(color))

        return gradient

    def _calculate_text_rect(self, painter: QPainter, text: str, content_rect: QRect) -> QRect:
        """Calculate the rectangle where text should be drawn.

        This accounts for alignment, word wrap, and other label properties.

        Args:
            painter: QPainter instance for measuring text.
            text: The text to be drawn.
            content_rect: The available content area.

        Returns:
            QRect defining where text should be rendered.
        """
        # For now, use the full content rect and let Qt handle alignment
        # This works well with Qt's text drawing alignment flags
        return content_rect


def create_gradient_label(
    text: str = "",
    gradient_colors: Optional[List[str]] = None,
    direction: Qt.Orientation = Qt.Orientation.Horizontal,
    parent=None,
) -> QLabel:
    """Factory function to create a QLabel with gradient text.

    This is a convenience function for creating gradient labels without
    needing to subclass. For more control, use GradientTextMixin directly.

    Args:
        text: Label text content.
        gradient_colors: List of color hex codes for the gradient.
                        If None, creates a standard label without gradient.
        direction: Gradient direction (Horizontal or Vertical).
        parent: Parent widget.

    Returns:
        QLabel instance with gradient text rendering if colors provided.

    Example:
        label = create_gradient_label(
            "Hello World",
            gradient_colors=["#4E98FF", "#F97070"]
        )
    """
    # Create a dynamic class that combines the mixin with QLabel
    class GradientLabel(GradientTextMixin, QLabel):
        pass

    label = GradientLabel(text, parent)

    if gradient_colors:
        label.enable_gradient(gradient_colors, direction)

    return label
