"""Qt-based themed components for Vocalance UI.

Provides custom QWidget subclasses styled with the application theme.
These components replace CustomTkinter widgets with Qt equivalents.
"""

from typing import Optional

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vocalance.app.ui.qt_theme import theme_manager


class ThemedButton(QPushButton):
    """Base themed button with consistent styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        size: Optional[int] = None,
        compact: bool = False,
    ):
        """Initialize themed button.

        Args:
            parent: Parent widget.
            text: Button text.
            size: Font size (defaults to medium).
            compact: Whether to use compact sizing.
        """
        super().__init__(text, parent)

        if size is None:
            size = theme_manager.font_sizes.medium

        font = theme_manager.get_font(size=size, weight="semibold")
        self.setFont(font)

        # Set fixed height and minimum width to ensure pill shape
        button_height = theme_manager.dimensions.button_height
        self.setFixedHeight(button_height)
        # Minimum width is button height to ensure pill shape works properly
        self.setMinimumWidth(button_height)

        if compact:
            # Calculate width based on text
            fm = self.fontMetrics()
            text_width = fm.horizontalAdvance(text) if text else button_height
            self.setFixedWidth(text_width + theme_manager.dimensions.button_text_padding * 2)


class PrimaryButton(ThemedButton):
    """Primary action button with gradient background styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        size: Optional[int] = None,
    ):
        """Initialize primary button with gradient background.

        Args:
            parent: Parent widget.
            text: Button text.
            size: Font size.
        """
        super().__init__(parent, text, size)
        self.setProperty("buttonType", "primary")

        # Apply gradient background using inline stylesheet
        gradient_start = theme_manager.gradient_colors.blue_rose_start
        gradient_end = theme_manager.gradient_colors.blue_rose_end
        button_height = theme_manager.dimensions.button_height
        border_radius = button_height // 2

        gradient_style = f"""
            QPushButton[buttonType="primary"] {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {gradient_start}, stop:1 {gradient_end});
                color: {theme_manager.text_colors.light_blue_accent};
                border: none;
                border-radius: {border_radius}px;
                padding: 4px 16px;
                font-weight: bold;
            }}
            QPushButton[buttonType="primary"]:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {gradient_start}, stop:1 {gradient_end});
                opacity: 0.9;
            }}
            QPushButton[buttonType="primary"]:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {gradient_start}, stop:1 {gradient_end});
            }}
        """
        self.setStyleSheet(gradient_style)


class DangerButton(ThemedButton):
    """Danger/destructive action button with warning styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        size: Optional[int] = None,
    ):
        """Initialize danger button.

        Args:
            parent: Parent widget.
            text: Button text.
            size: Font size.
        """
        super().__init__(parent, text, size)
        self.setProperty("buttonType", "danger")


class SidebarButton(QPushButton):
    """Sidebar navigation button with icon and text."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        icon_pixmap: Optional[QPixmap] = None,
    ):
        """Initialize sidebar button.

        Args:
            parent: Parent widget.
            text: Button text.
            icon_pixmap: Optional icon pixmap.
        """
        super().__init__("", parent)  # Start with no text
        self.setProperty("buttonType", "sidebar")
        self._selected = False
        self._button_text = text
        self._expanded = False
        self._original_icon_pixmap = icon_pixmap
        self._hover_icon_pixmap = None
        self._default_icon_pixmap = None
        self._is_hovered = False

        if icon_pixmap:
            icon = QIcon(icon_pixmap)
            self.setIcon(icon)
            self._default_icon_pixmap = icon_pixmap
            # Pre-generate hover state icon
            self._generate_hover_icon()
            # Icon size will be set by expandable sidebar

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium, weight="semibold")
        self.setFont(font)

        self.setMinimumHeight(50)

    def _generate_hover_icon(self) -> None:
        """Generate the hover state icon with light_blue_accent color."""
        if not self._original_icon_pixmap:
            return

        try:
            from PIL import Image

            # Convert QPixmap to PIL Image for transformation
            image = self._original_icon_pixmap.toImage()
            width = image.width()
            height = image.height()

            # Convert to PIL Image
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            arr = ptr.asarray()
            pil_img = Image.frombytes("RGBA", (width, height), bytes(arr), "raw", "BGRA")

            # Create colored version
            colored = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            colored_pixels = colored.load()
            pixels = pil_img.load()

            light_blue = theme_manager.text_colors.light_blue_accent
            r = int(light_blue[1:3], 16)
            g = int(light_blue[3:5], 16)
            b = int(light_blue[5:7], 16)

            for y in range(pil_img.height):
                for x in range(pil_img.width):
                    pixel = pixels[x, y]
                    if pixel[3] > 0:  # Not transparent
                        luminance = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
                        opacity_factor = (255 - luminance) / 255.0
                        new_alpha = int(opacity_factor * pixel[3])
                        colored_pixels[x, y] = (r, g, b, new_alpha)

            # Convert back to QPixmap
            from vocalance.app.ui.utils.qt_icon_utils import pil_image_to_qpixmap

            self._hover_icon_pixmap = pil_image_to_qpixmap(colored)
        except Exception as e:
            # Fallback: if conversion fails, just use default
            import logging

            logging.error(f"Failed to generate hover icon: {e}")
            self._hover_icon_pixmap = self._original_icon_pixmap

    def set_selected(self, selected: bool) -> None:
        """Set button selected state.

        Args:
            selected: Whether button is selected.
        """
        self._selected = selected
        self.setProperty("selected", "true" if selected else "false")
        # Update icon color for selected state
        if selected and self._hover_icon_pixmap:
            self.setIcon(QIcon(self._hover_icon_pixmap))
        else:
            self.setIcon(QIcon(self._default_icon_pixmap))
        self.style().unpolish(self)
        self.style().polish(self)

    def set_expanded(self, expanded: bool) -> None:
        """Set button expanded state to show/hide text.

        Args:
            expanded: Whether to show text alongside icon.
        """
        self._expanded = expanded
        if expanded:
            self.setText(self._button_text)
        else:
            self.setText("")
        self.update()

    def enterEvent(self, event) -> None:
        """Handle mouse enter event."""
        self._is_hovered = True
        if self._hover_icon_pixmap:
            self.setIcon(QIcon(self._hover_icon_pixmap))
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        """Handle mouse leave event."""
        self._is_hovered = False
        if not self._selected:
            if self._default_icon_pixmap:
                self.setIcon(QIcon(self._default_icon_pixmap))
        else:
            # Keep hover icon if selected
            if self._hover_icon_pixmap:
                self.setIcon(QIcon(self._hover_icon_pixmap))
        super().leaveEvent(event)


class ThemedLabel(QLabel):
    """Themed label with pre-configured design attributes."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        size: Optional[int] = None,
        color: Optional[str] = None,
        bold: bool = False,
        word_wrap: bool = False,
    ):
        """Initialize themed label.

        Args:
            parent: Parent widget.
            text: Label text.
            size: Font size (defaults to medium).
            color: Text color (defaults to lightest).
            bold: Whether to use bold font.
            word_wrap: Whether to enable word wrapping.
        """
        super().__init__(text, parent)

        if size is None:
            size = theme_manager.font_sizes.medium

        font = theme_manager.get_font(size=size, bold=bold)
        self.setFont(font)

        if color:
            self.setStyleSheet(f"color: {color};")

        if word_wrap:
            self.setWordWrap(True)


class TitleLabel(ThemedLabel):
    """Large title label."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
    ):
        """Initialize title label.

        Args:
            parent: Parent widget.
            text: Title text.
        """
        super().__init__(
            parent,
            text,
            size=theme_manager.font_sizes.xxlarge,
            bold=False,
        )
        font = theme_manager.get_font(size=theme_manager.font_sizes.xxlarge, weight="semibold")
        self.setFont(font)
        self.setProperty("labelType", "title")


class SubtitleLabel(ThemedLabel):
    """Subtitle label with lighter color."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
    ):
        """Initialize subtitle label.

        Args:
            parent: Parent widget.
            text: Subtitle text.
        """
        super().__init__(
            parent,
            text,
            size=theme_manager.font_sizes.medium,
            color=theme_manager.text_colors.light,
        )
        self.setProperty("labelType", "subtitle")


class BoxTitle(QLabel):
    """Pre-configured label for box titles with gradient text (xxlarge, bold)."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
    ):
        """Initialize box title label with gradient.

        Args:
            parent: Parent widget.
            text: Title text.
        """
        super().__init__(text, parent)

        # Use xxlarge font size (same as header title)
        font = theme_manager.get_font(size=theme_manager.font_sizes.xxlarge, weight="semibold")
        self.setFont(font)

        # Store original text
        self._original_text = text

        # Apply gradient text effect using rich text HTML
        self._apply_gradient_effect()

    def _apply_gradient_effect(self) -> None:
        """Apply gradient text effect using character-by-character coloring."""
        from PySide6.QtGui import QColor

        text = self._original_text
        if not text:
            return

        # Get gradient colors
        start_color = QColor(theme_manager.gradient_colors.blue_rose_start)
        end_color = QColor(theme_manager.gradient_colors.blue_rose_end)

        # Get font size - CRITICAL: HTML ignores setFont(), so we must include it in styles
        font_size = theme_manager.font_sizes.xlarge

        # Build HTML with gradient
        html_parts = []
        text_length = len(text)

        for i, char in enumerate(text):
            # Calculate interpolation factor
            if text_length > 1:
                factor = i / (text_length - 1)
            else:
                factor = 0

            # Interpolate color
            r = int(start_color.red() + (end_color.red() - start_color.red()) * factor)
            g = int(start_color.green() + (end_color.green() - start_color.green()) * factor)
            b = int(start_color.blue() + (end_color.blue() - start_color.blue()) * factor)

            color_hex = f"#{r:02x}{g:02x}{b:02x}"

            # Add colored character with font size and weight in inline style
            if char == " ":
                html_parts.append("&nbsp;")
            else:
                html_parts.append(f'<span style="color: {color_hex}; font-size: {font_size}px; font-weight: bold;">{char}</span>')

        # Set the HTML text
        html_text = "".join(html_parts)
        super().setText(html_text)

    def setText(self, text: str) -> None:
        """Override setText to maintain gradient styling.

        Args:
            text: Text to set.
        """
        self._original_text = text
        self._apply_gradient_effect()


class TileTitle(ThemedLabel):
    """Pre-configured label for tile titles (large, bold)."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
    ):
        """Initialize tile title label.

        Args:
            parent: Parent widget.
            text: Title text.
        """
        super().__init__(
            parent,
            text,
            size=theme_manager.font_sizes.large,
            bold=False,
        )
        font = theme_manager.get_font(size=theme_manager.font_sizes.large, weight="semibold")
        self.setFont(font)


class TileContent(ThemedLabel):
    """Pre-configured label for tile content (small, center-aligned)."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
    ):
        """Initialize tile content label.

        Args:
            parent: Parent widget.
            text: Content text.
        """
        super().__init__(
            parent,
            text,
            size=theme_manager.font_sizes.small,
            color=theme_manager.text_colors.dark,
            word_wrap=True,
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class ThemedEntry(QLineEdit):
    """Themed text entry field with pre-configured styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        placeholder_text: str = "",
    ):
        """Initialize themed entry.

        Args:
            parent: Parent widget.
            placeholder_text: Placeholder text when empty.
        """
        super().__init__(parent)

        if placeholder_text:
            self.setPlaceholderText(placeholder_text)

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)

        self.setMinimumHeight(theme_manager.dimensions.entry_height)


class ThemedTextEdit(QTextEdit):
    """Themed multi-line text editor."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed text edit.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)


class ThemedPlainTextEdit(QPlainTextEdit):
    """Themed plain text editor for code/monospace content."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed plain text edit.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        font = theme_manager.get_monospace_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)


class ThemedComboBox(QComboBox):
    """Themed dropdown/combobox."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed combobox.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)

        self.setMinimumHeight(theme_manager.dimensions.entry_height)


class ThemedFrame(QFrame):
    """Base themed frame with surface styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        frame_type: str = "default",
    ):
        """Initialize themed frame.

        Args:
            parent: Parent widget.
            frame_type: Frame type for styling (default, box, tile, header, sidebar, transparent).
        """
        super().__init__(parent)

        if frame_type != "default":
            self.setProperty("frameType", frame_type)

        self.setFrameShape(QFrame.Shape.NoFrame)


class TransparentFrame(QFrame):
    """Transparent frame for grouping without visual boundaries."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize transparent frame.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setProperty("frameType", "transparent")
        self.setFrameShape(QFrame.Shape.NoFrame)


class BoxFrame(ThemedFrame):
    """Box frame with border and background."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize box frame.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent, frame_type="box")


class TileFrame(ThemedFrame):
    """Tile frame for instruction/info content."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize tile frame.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent, frame_type="tile")


class ThemedScrollArea(QScrollArea):
    """Scrollable area with themed scrollbars."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed scroll area.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)


class ScrollableFrame(QWidget):
    """Scrollable frame container (replacement for CTkScrollableFrame)."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize scrollable frame.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        # Create layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Create scroll area
        self._scroll_area = ThemedScrollArea(self)
        self._layout.addWidget(self._scroll_area)

        # Create content widget
        self._content_widget = TransparentFrame()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(5, 5, 5, 5)
        self._content_layout.setSpacing(5)
        self._content_layout.addStretch()

        self._scroll_area.setWidget(self._content_widget)

    def add_widget(self, widget: QWidget) -> None:
        """Add widget to scrollable content.

        Args:
            widget: Widget to add.
        """
        # Insert before the stretch
        count = self._content_layout.count()
        self._content_layout.insertWidget(count - 1, widget)

    def get_content_layout(self) -> QVBoxLayout:
        """Get the content layout for adding widgets.

        Returns:
            Content layout object.
        """
        return self._content_layout


class ThemedProgressBar(QProgressBar):
    """Themed progress bar."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed progress bar.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(0)
        self.setTextVisible(True)
        self.setMinimumHeight(theme_manager.dimensions.progress_bar_height)


class ThemedCheckBox(QCheckBox):
    """Themed checkbox/toggle."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
    ):
        """Initialize themed checkbox.

        Args:
            parent: Parent widget.
            text: Checkbox label text.
        """
        super().__init__(text, parent)

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)


class SidebarButtonManager:
    """Manages sidebar button selection state."""

    def __init__(self):
        """Initialize sidebar button manager."""
        self._buttons: list[SidebarButton] = []
        self._selected_button: Optional[SidebarButton] = None

    def add_button(self, button: SidebarButton) -> None:
        """Add button to manager.

        Args:
            button: Sidebar button to manage.
        """
        self._buttons.append(button)

        # Connect click to selection
        def on_click():
            self.select_button(button)

        button.clicked.connect(on_click)

    def select_button(self, button: SidebarButton) -> None:
        """Select a button (deselecting others).

        Args:
            button: Button to select.
        """
        # Deselect previous
        if self._selected_button:
            self._selected_button.set_selected(False)

        # Select new
        button.set_selected(True)
        self._selected_button = button

    def set_all_expanded(self, expanded: bool) -> None:
        """Set all buttons to expanded or collapsed state.

        Args:
            expanded: Whether buttons should show text.
        """
        for button in self._buttons:
            button.set_expanded(expanded)


class ExpandableSidebar(ThemedFrame):
    """Expandable sidebar that shows icons by default and expands on hover."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize expandable sidebar.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent, frame_type="sidebar")

        self._collapsed_width = theme_manager.sidebar_layout.collapsed_width
        self._expanded_width = theme_manager.sidebar_layout.expanded_width
        self._is_expanded = False

        # Set initial width
        self.setFixedWidth(self._collapsed_width)

        # Enable mouse tracking for hover detection
        self.setMouseTracking(True)

        # Create animation for width changes
        self._animation = QPropertyAnimation(self, b"minimumWidth")
        self._animation.setDuration(theme_manager.sidebar_layout.animation_duration)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Create second animation for maximum width
        self._max_animation = QPropertyAnimation(self, b"maximumWidth")
        self._max_animation.setDuration(theme_manager.sidebar_layout.animation_duration)
        self._max_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, theme_manager.sidebar_layout.top_spacing, 0, 0)
        self._main_layout.setSpacing(0)

        # Button manager
        self.button_manager = SidebarButtonManager()

    def add_button_widget(self, button_widget: QWidget) -> None:
        """Add button widget to sidebar.

        Args:
            button_widget: Widget containing buttons to add.
        """
        self._main_layout.addWidget(button_widget)

    def add_stretch(self) -> None:
        """Add stretch to push subsequent widgets to bottom."""
        self._main_layout.addStretch()

    def add_logo(self, logo_widget: QWidget) -> None:
        """Add logo widget to sidebar.

        Args:
            logo_widget: Logo widget to add.
        """
        self._main_layout.addWidget(logo_widget)

    def enterEvent(self, event) -> None:
        """Handle mouse entering sidebar to expand.

        Args:
            event: Enter event.
        """
        super().enterEvent(event)
        self._expand()

    def leaveEvent(self, event) -> None:
        """Handle mouse leaving sidebar to collapse.

        Args:
            event: Leave event.
        """
        super().leaveEvent(event)
        self._collapse()

    def _expand(self) -> None:
        """Expand sidebar to show text."""
        if self._is_expanded:
            return

        self._is_expanded = True

        # Animate width
        self._animation.setStartValue(self._collapsed_width)
        self._animation.setEndValue(self._expanded_width)
        self._animation.start()

        self._max_animation.setStartValue(self._collapsed_width)
        self._max_animation.setEndValue(self._expanded_width)
        self._max_animation.start()

        # Show text on all buttons
        self.button_manager.set_all_expanded(True)

    def _collapse(self) -> None:
        """Collapse sidebar to show icons only."""
        if not self._is_expanded:
            return

        self._is_expanded = False

        # Animate width
        self._animation.setStartValue(self._expanded_width)
        self._animation.setEndValue(self._collapsed_width)
        self._animation.start()

        self._max_animation.setStartValue(self._expanded_width)
        self._max_animation.setEndValue(self._collapsed_width)
        self._max_animation.start()

        # Hide text on all buttons
        self.button_manager.set_all_expanded(False)


class SpinnerLabel(QLabel):
    """Animated spinner label for loading indicators."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize spinner label.

        Args:
            parent: Parent widget.
        """
        super().__init__("", parent)

        font = theme_manager.get_monospace_font(size=theme_manager.font_sizes.small)
        self.setFont(font)

        self.setFixedWidth(15)
        self._frame = 0
        self._frames = ["|", "/", "-", "\\"]

    def update_frame(self) -> None:
        """Update spinner to next frame."""
        self.setText(self._frames[self._frame])
        self._frame = (self._frame + 1) % len(self._frames)


# Composite layout components


class InstructionTile(QWidget):
    """Pre-configured tile for instruction content matching legacy design."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "",
        content: str = "",
    ):
        """Initialize instruction tile.

        Args:
            parent: Parent widget.
            title: Tile title text.
            content: Tile content text.
        """
        super().__init__(parent)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme_manager.spacing.medium,
            theme_manager.spacing.medium,
            theme_manager.spacing.medium,
            theme_manager.spacing.medium,
        )
        layout.setSpacing(theme_manager.spacing.small)

        # Title - center aligned with transparent background
        title_label = TileTitle(self, text=title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(title_label)

        # Content - center aligned with transparent background
        content_label = TileContent(self, text=content)
        content_label.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(content_label)

        # Style the tile - transparent background and border
        self.setStyleSheet(
            f"""
            InstructionTile {{
                background-color: transparent;
                border: none;
                border-radius: {theme_manager.border_radius.rounded}px;
            }}
        """
        )


class TwoColumnTabLayout(TransparentFrame):
    """Pre-configured two-column layout for tabs with titles and boxes.

    Highly modular layout that provides left_content and right_content containers
    for views to populate. Matches legacy TwoColumnTabLayout pattern but using Qt layouts.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        left_title: str = "",
        right_title: str = "",
    ):
        """Initialize two-column layout.

        Args:
            parent: Parent widget.
            left_title: Title for left box.
            right_title: Title for right box.
        """
        super().__init__(parent)

        # Calculate half inner spacing
        half_inner_spacing = theme_manager.two_box_layout.base_spacing // 2

        # Main horizontal layout with outer padding
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)  # We'll handle spacing with padx on boxes

        # LEFT BOX
        self.left_box = ThemedFrame(frame_type="two_box")
        left_outer_layout = QVBoxLayout(self.left_box)
        left_outer_layout.setContentsMargins(0, 0, 0, 0)
        left_outer_layout.setSpacing(0)

        # Left title
        if left_title:
            left_title_label = BoxTitle(text=left_title)
            left_title_label.setStyleSheet("border: none; background: transparent;")
            left_title_container = TransparentFrame()
            left_title_container.setStyleSheet("border: none; background: transparent;")
            left_title_layout = QVBoxLayout(left_title_container)
            left_title_layout.setContentsMargins(
                theme_manager.two_box_layout.inner_content_padx,
                theme_manager.spacing.large,
                theme_manager.two_box_layout.inner_content_padx,
                theme_manager.spacing.small,
            )
            left_title_layout.addWidget(left_title_label)
            left_outer_layout.addWidget(left_title_container, stretch=0)

        # Left content container - this is where views add their content
        self.left_content = TransparentFrame()
        left_content_layout = QVBoxLayout(self.left_content)
        left_content_layout.setContentsMargins(0, 0, 0, theme_manager.spacing.large)
        left_content_layout.setSpacing(0)
        left_outer_layout.addWidget(self.left_content, stretch=1)

        # RIGHT BOX
        self.right_box = ThemedFrame(frame_type="two_box")
        right_outer_layout = QVBoxLayout(self.right_box)
        right_outer_layout.setContentsMargins(0, 0, 0, 0)
        right_outer_layout.setSpacing(0)

        # Right title
        if right_title:
            right_title_label = BoxTitle(text=right_title)
            right_title_label.setStyleSheet("border: none; background: transparent;")
            right_title_container = TransparentFrame()
            right_title_container.setStyleSheet("border: none; background: transparent;")
            right_title_layout = QVBoxLayout(right_title_container)
            right_title_layout.setContentsMargins(
                theme_manager.two_box_layout.inner_content_padx,
                theme_manager.spacing.large,
                theme_manager.two_box_layout.inner_content_padx,
                theme_manager.spacing.small,
            )
            right_title_layout.addWidget(right_title_label)
            right_outer_layout.addWidget(right_title_container, stretch=0)

        # Right content container - this is where views add their content
        self.right_content = TransparentFrame()
        right_content_layout = QVBoxLayout(self.right_content)
        right_content_layout.setContentsMargins(0, 0, 0, theme_manager.spacing.large)
        right_content_layout.setSpacing(0)
        right_outer_layout.addWidget(self.right_content, stretch=1)

        # Add boxes to main layout with proper spacing
        # Left box: inner_content_padx on left, half_inner_spacing on right, bottom padding
        left_container = QWidget()
        left_container_layout = QHBoxLayout(left_container)
        left_container_layout.setContentsMargins(
            half_inner_spacing,
            0,
            half_inner_spacing,
            half_inner_spacing,
        )
        left_container_layout.setSpacing(0)
        left_container_layout.addWidget(self.left_box)
        main_layout.addWidget(left_container, stretch=1)

        # Right box: half_inner_spacing on left, inner_content_padx on right, bottom padding
        right_container = QWidget()
        right_container_layout = QHBoxLayout(right_container)
        right_container_layout.setContentsMargins(
            half_inner_spacing,
            0,
            half_inner_spacing,
            half_inner_spacing,
        )
        right_container_layout.setSpacing(0)
        right_container_layout.addWidget(self.right_box)
        main_layout.addWidget(right_container, stretch=1)
