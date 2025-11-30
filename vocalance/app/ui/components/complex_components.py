from typing import Optional, Tuple

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPixmap
from PySide6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BodyLabel, SmallLabel
from vocalance.app.ui.components.layouts import BaseContainer
from vocalance.app.ui.qt_theme import theme


class FormGroup(QWidget):
    """Container for a label and an input field.

    Note: Consider using FormField from layouts.py instead for new code.
    """

    def __init__(
        self,
        label: str,
        input_widget: QWidget,
        parent: Optional[QWidget] = None,
        description: Optional[str] = None,
    ):
        super().__init__(parent)

        # Transparent background
        self.setAutoFillBackground(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.config.spacing.tiny)

        # Label
        self.label = SmallLabel(label, color=theme.config.text.light)
        layout.addWidget(self.label)

        # Input
        layout.addWidget(input_widget)

        # Optional description
        if description:
            desc = SmallLabel(description, color=theme.config.text.medium)
            layout.addWidget(desc)

    @staticmethod
    def create_text(
        label: str,
        placeholder: str = "",
        default: str = "",
        parent: Optional[QWidget] = None,
    ) -> Tuple["FormGroup", TextInput]:
        """Factory to create a text input form group."""
        inp = TextInput(placeholder)
        if default:
            inp.setText(str(default))
        group = FormGroup(label, inp, parent)
        return group, inp


class Tile(BaseContainer):
    """Tile component for instructions or info cards with rounded corners and gradient title text."""

    def __init__(self, title: str, content: str, parent: Optional[QWidget] = None):
        super().__init__(
            parent=parent,
            layout="vertical",
            bg_color="transparent",
            border_color=theme.config.shapes.accent,
            border_radius=theme.config.radius.rounded,
        )

        padding = theme.config.spacing.medium
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(theme.config.spacing.small)

        # Title - medium size with Alata display font and gradient text
        from PySide6.QtGui import QBrush, QLinearGradient
        from PySide6.QtWidgets import QSizePolicy

        from vocalance.app.ui.utils.qt_gradient_text import GradientTextMixin

        class CenteredGradientLabel(GradientTextMixin, QLabel):
            """Gradient label with proper center alignment support."""

            def paintEvent(self, event):
                """Override paintEvent to render centered text with gradient."""
                if not self._gradient_enabled or not self._gradient_colors:
                    # Fall back to default QLabel painting
                    QLabel.paintEvent(self, event)
                    return

                # Custom gradient painting with center alignment
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

                text = self.text()
                if not text:
                    painter.end()
                    return

                painter.setFont(self.font())

                # Get content rectangle
                content_rect = self.contentsRect()
                if content_rect.isEmpty():
                    content_rect = self.rect()

                # Calculate text position based on alignment
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                text_height = font_metrics.height()
                alignment = self.alignment()

                # Determine horizontal position
                if alignment & Qt.AlignmentFlag.AlignRight:
                    text_x = content_rect.right() - text_width
                elif alignment & Qt.AlignmentFlag.AlignHCenter or alignment & Qt.AlignmentFlag.AlignCenter:
                    text_x = content_rect.x() + (content_rect.width() - text_width) / 2
                else:  # AlignLeft (default)
                    text_x = content_rect.x()

                # Determine vertical position
                if alignment & Qt.AlignmentFlag.AlignBottom:
                    text_y = content_rect.bottom() - text_height
                elif alignment & Qt.AlignmentFlag.AlignVCenter or alignment & Qt.AlignmentFlag.AlignCenter:
                    text_y = content_rect.y() + (content_rect.height() - text_height) / 2
                else:  # AlignTop (default)
                    text_y = content_rect.y()

                # Create gradient for text
                gradient = QLinearGradient(text_x, text_y, text_x + text_width, text_y)
                num_colors = len(self._gradient_colors)
                for i, color in enumerate(self._gradient_colors):
                    position = i / (num_colors - 1) if num_colors > 1 else 0
                    gradient.setColorAt(position, QColor(color))

                # Create text path at calculated position
                path = QPainterPath()
                path.addText(text_x, text_y + font_metrics.ascent(), self.font(), text)

                # Draw text with gradient
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

        # Content - small font, medium color (text that's less prominent)
        content_label = QLabel(content)
        content_label.setFont(theme.get_font(size="small", weight="regular"))
        content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        palette = content_label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.medium))
        content_label.setPalette(palette)
        content_label.setAutoFillBackground(False)
        content_label.setWordWrap(True)
        self.add(content_label, stretch=1)


class IconWidget(QWidget):
    """Widget that renders an icon pixmap with high-quality scaling.

    Used to replace QLabel for icons to ensure strict size constraints
    and high-quality rendering regardless of source pixmap DPI.
    """

    def __init__(self, pixmap: Optional[QPixmap], size: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._pixmap = pixmap
        # Enforce fixed size logic
        self.setFixedSize(size, size)
        # Transparent background
        self.setAutoFillBackground(False)
        # Let mouse events pass through to parent button
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_pixmap(self, pixmap: QPixmap):
        """Update the displayed pixmap."""
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        """Paint the pixmap scaled to the widget's fixed size."""
        if not self._pixmap or self._pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw pixmap into the full widget rect
        # The widget size is fixed to the target logical size.
        # QPainter handles the scaling from the source pixmap to this rect.
        # This guarantees the icon NEVER exceeds the specified size.
        painter.drawPixmap(self.rect(), self._pixmap)
        painter.end()


class SidebarButton(QWidget):
    """Sidebar navigation button with icon and text.

    Handles selection state and hover effects programmatically.
    Uses fixed icon positioning for smooth sidebar expansion animation.
    """

    clicked = Signal()

    def __init__(
        self,
        text: str,
        icon_pixmap: Optional[QPixmap] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._text_content = text
        self._selected = False
        self._expanded = False
        self._hovered = False
        self._default_icon = icon_pixmap

        # Colors
        self._bg_color_default = "transparent"
        self._bg_color_hover = theme.config.shapes.accent
        self._bg_color_selected = theme.config.shapes.accent
        self._text_color_default = theme.config.shapes.accent
        self._text_color_hover = theme.config.blue.blue_1
        self._text_color_selected = theme.config.blue.blue_1

        # Store border radius for custom painting
        self._border_radius = theme.config.radius.small

        # Calculate icon position (centered in collapsed width)
        self._icon_area_width = theme.config.sidebar.collapsed_width
        self._button_padding = 4

        # Ensure no background fill - we handle all painting in paintEvent
        self.setAutoFillBackground(False)

        # Setup UI
        self._setup_ui(icon_pixmap)

        # Enable mouse tracking for hover
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        # Set cursor
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Set minimum height
        self.setMinimumHeight(theme.config.sidebar.button_min_height)

        # Apply initial styling
        self._update_appearance()

    def _setup_ui(self, icon_pixmap: Optional[QPixmap]):
        """Setup button UI with fixed icon positioning.

        Design: Icon area has fixed width matching collapsed sidebar.
        Text appears next to icon area when expanded.
        """

        layout = QHBoxLayout(self)
        layout.setContentsMargins(self._button_padding, self._button_padding, self._button_padding, self._button_padding)
        layout.setSpacing(0)
        # CRITICAL: Anchor content to the left to prevent centering jolt when text is hidden
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Icon area: fixed width container that matches collapsed sidebar width
        self.icon_area = QWidget()
        self.icon_area.setFixedWidth(self._icon_area_width - (2 * self._button_padding))
        icon_area_layout = QHBoxLayout(self.icon_area)
        icon_area_layout.setContentsMargins(0, 0, 0, 0)
        icon_area_layout.setSpacing(0)

        if icon_pixmap:
            # Use IconWidget for robust scaling instead of QLabel
            self.icon_widget = IconWidget(icon_pixmap, theme.config.sidebar.button_icon_size)
            # Center icon within the fixed icon area
            icon_area_layout.addStretch()
            icon_area_layout.addWidget(self.icon_widget, alignment=Qt.AlignmentFlag.AlignCenter)
            icon_area_layout.addStretch()
        else:
            self.icon_widget = None

        layout.addWidget(self.icon_area)

        # Spacer widget (toggleable) - replaces addSpacing so we can hide it
        self.spacer = QWidget()
        self.spacer.setFixedWidth(theme.config.sidebar.button_icon_text_spacing)
        self.spacer.setVisible(False)  # Initially hidden
        layout.addWidget(self.spacer)

        # Text label (initially hidden, uses Alata display font)
        self.text_label = BodyLabel(self._text_content)
        font = theme.get_font(size="medium", weight="semibold", display=True)
        self.text_label.setFont(font)

        # Setup opacity effect for smooth fade-in
        self.opacity_effect = QGraphicsOpacityEffect(self.text_label)
        self.opacity_effect.setOpacity(1.0)
        self.text_label.setGraphicsEffect(self.opacity_effect)

        self.text_label.setVisible(False)
        # Use explicit alignment for text instead of stretch to avoid layout shifts
        layout.addWidget(self.text_label)

        # Add a final stretch to consume any remaining space on the right
        # This reinforces the left alignment of the icon
        layout.addStretch(1)

    def _update_appearance(self):
        """Update button appearance based on state."""
        # Determine text color: blue_2 on hover/select, shapes.accent by default
        if self._hovered or self._selected:
            text_color = theme.config.blue.blue_2
        else:
            text_color = theme.config.shapes.accent

        # Update text color
        if self.text_label:
            palette = self.text_label.palette()
            palette.setColor(QPalette.ColorRole.WindowText, QColor(text_color))
            self.text_label.setPalette(palette)

        # Update icon color based on state
        if self.icon_widget and self._default_icon:
            if self._hovered or self._selected:
                # Use blue_2 color on hover/select
                icon_color = theme.config.blue.blue_2
                colored_pixmap = self._color_pixmap(self._default_icon, icon_color)
                self.icon_widget.set_pixmap(colored_pixmap)
            else:
                # Use default icon (shapes.accent color)
                self.icon_widget.set_pixmap(self._default_icon)

        self.update()

    def _color_pixmap(self, pixmap: QPixmap, color: str) -> QPixmap:
        """Color a pixmap by replacing white pixels with the given color."""
        result = pixmap.copy()

        # Create a painter to recolor the pixmap
        painter = QPainter(result)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(result.rect(), QColor(color))
        painter.end()

        return result

    def set_selected(self, selected: bool):
        """Set selection state."""
        self._selected = selected
        self._update_appearance()

    def set_text_opacity(self, opacity: float):
        """Set opacity of the text label."""
        if hasattr(self, "opacity_effect"):
            self.opacity_effect.setOpacity(opacity)

    def set_expanded(self, expanded: bool):
        """Set expansion state."""
        self._expanded = expanded
        if self.text_label:
            self.text_label.setVisible(expanded)
        if hasattr(self, "spacer"):
            self.spacer.setVisible(expanded)

    def enterEvent(self, event):
        """Handle mouse enter."""
        self._hovered = True
        self._update_appearance()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave."""
        self._hovered = False
        self._update_appearance()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press - emit clicked signal."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        """No custom painting - rely on theme colors for text and icon highlighting."""
        # Remove the blue background highlight - just use default painting
        super().paintEvent(event)


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
        text_icon_spacing: int = None,
        icon_size: int = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._text_content = text
        self._hovered = False
        self._default_icon = icon_pixmap
        self._text_icon_spacing = text_icon_spacing if text_icon_spacing is not None else theme.config.spacing.medium

        # Colors - blue_1 for both text and icon, no background
        self._icon_color = theme.config.blue.blue_2
        self._text_color = theme.config.blue.blue_2

        # Icon size - can be customized, defaults to 40px
        self._icon_size = icon_size if icon_size is not None else 40
        self._button_padding = 8

        # No background
        self.setAutoFillBackground(False)

        # Setup UI
        self._setup_ui(icon_pixmap)

        # Enable mouse tracking for hover
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        # Set cursor
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Setup animation for text expansion
        self._setup_animation()

    def _setup_ui(self, icon_pixmap: Optional[QPixmap]):
        """Setup button UI with icon on right, text expands left on hover."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(self._button_padding, self._button_padding, self._button_padding, self._button_padding)
        layout.setSpacing(0)  # We'll handle spacing via the spacer widget
        layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Text label (initially hidden, will expand from right to left, uses Alata display font)
        self.text_label = BodyLabel(self._text_content)
        font = theme.get_font(size="medium", weight="regular", display=True)
        self.text_label.setFont(font)

        # Set text color to blue_1
        palette = self.text_label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(self._text_color))
        self.text_label.setPalette(palette)

        # Hide text initially by setting max width to 0
        self.text_label.setMaximumWidth(0)
        layout.addWidget(self.text_label)

        # Spacer between text and icon (initially 0, animates on hover)
        self.spacer = QWidget()
        self.spacer.setMinimumWidth(0)
        self.spacer.setMaximumWidth(0)  # Start collapsed
        self.spacer.setFixedHeight(1)  # Minimal height, won't affect layout
        layout.addWidget(self.spacer)

        # Icon area on the right
        if icon_pixmap:
            # Color the icon with blue_1
            colored_icon = self._color_pixmap(icon_pixmap, self._icon_color)

            # Use IconWidget for robust scaling
            self.icon_widget = IconWidget(colored_icon, self._icon_size)
            layout.addWidget(self.icon_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            self.icon_widget = None

    def _setup_animation(self):
        """Setup animation for text expansion."""
        # Animation for text label width
        self._text_anim = QPropertyAnimation(self.text_label, b"maximumWidth")
        self._text_anim.setDuration(200)
        self._text_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Animation for spacer width
        self._spacer_anim = QPropertyAnimation(self.spacer, b"maximumWidth")
        self._spacer_anim.setDuration(200)
        self._spacer_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _color_pixmap(self, pixmap: QPixmap, color: str) -> QPixmap:
        """Color a pixmap by replacing pixels with the given color."""
        result = pixmap.copy()
        painter = QPainter(result)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(result.rect(), QColor(color))
        painter.end()
        return result

    def _animate_expansion(self, expand: bool):
        """Animate text expansion/collapse."""
        if expand:
            # Calculate text width
            fm = self.text_label.fontMetrics()
            text_width = fm.horizontalAdvance(self._text_content) + 10  # Add padding

            # Expand text
            self._text_anim.setStartValue(self.text_label.maximumWidth())
            self._text_anim.setEndValue(text_width)
            self._text_anim.start()

            # Expand spacer
            self._spacer_anim.setStartValue(self.spacer.width())
            self._spacer_anim.setEndValue(self._text_icon_spacing)
            self._spacer_anim.start()
        else:
            # Collapse text
            self._text_anim.setStartValue(self.text_label.maximumWidth())
            self._text_anim.setEndValue(0)
            self._text_anim.start()

            # Collapse spacer
            self._spacer_anim.setStartValue(self.spacer.width())
            self._spacer_anim.setEndValue(0)
            self._spacer_anim.start()

    def enterEvent(self, event):
        """Handle mouse enter - show text."""
        self._hovered = True
        self._animate_expansion(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave - hide text."""
        self._hovered = False
        self._animate_expansion(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press - emit clicked signal."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
