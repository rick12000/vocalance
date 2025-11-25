"""Qt theme system with QSS stylesheets and theme management.

Provides centralized theme configuration for PySide6 UI with:
- Design tokens (Colors, Fonts, Spacing)
- Container spacing system with clear hierarchy
- Component styling rules
- Runtime theme application

SPACING HIERARCHY:
1. Container Border (1px) - outermost boundary
2. Container Padding - space between border and content (box_padding)
3. Content Margin - space for content inside containers
4. Item Spacing - space between items in lists/grids
5. Element Padding - internal padding within elements
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtGui import QFont, QFontDatabase


@dataclass
class FontSizes:
    """Font size design tokens."""

    small: int = 11
    medium: int = 12
    large: int = 18
    xlarge: int = 24
    xxlarge: int = 30


@dataclass
class TextColors:
    """Text color design tokens."""

    color_accent: str = "#918f66"
    lightest: str = "#e8d6d6"
    light: str = "#c3afaf"
    medium: str = "#7a7a7a"
    dark: str = "#a79494"
    darkest: str = "#1f1f1f"
    success: str = "#28a745"
    error: str = "#dc3545"
    warning: str = "#ffc107"
    info: str = "#17a2b8"
    streaming_token: str = "#c79b9b"
    light_blue_accent: str = "#2b3054"

    # Gradient colors for buttons and UI elements
    gradient_colors: list = field(default_factory=lambda: ["#4E98FF", "#F97070"])
    gradient_colors_dark: list = field(default_factory=lambda: ["#5c1a17", "#0d3a6d"])


@dataclass
class ShapeColors:
    """Shape/background color design tokens."""

    accent: str = "#a3a3a3"
    accent_minus: str = "#7c7c7c"
    lightest: str = "#515151"
    light: str = "#404040"
    medium: str = "#393939"
    dark: str = "#2a2a2a"
    darkest: str = "#131313"
    background: str = "#131313"
    transparent: str = "transparent"


@dataclass
class BlueColors:
    """Blue color design tokens."""

    blue_1: str = "#1f3760"
    blue_2: str = "#a8c7fa"


@dataclass
class Spacing:
    """Core spacing scale for use between components and within layouts."""

    none: int = 0
    tiny: int = 4
    small: int = 8
    medium: int = 12
    large: int = 16
    xlarge: int = 24
    xxlarge: int = 32


@dataclass
class BorderRadius:
    """Border radius design tokens."""

    small: int = 8
    medium: int = 10
    large: int = 15
    rounded: int = 20
    xlarge: int = 30
    pill: int = 999  # Use for pill shapes


@dataclass
class ContainerLayout:
    """Container spacing system - defines relationship between containers and content.

    HIERARCHY:
    - border: 1px solid line (part of container style)
    - padding: space from border to content area
    - content_margin: additional margin for content widgets inside container
    """

    # Box containers (primary content boxes)
    box_padding: int = 20  # Padding from border to content
    box_spacing_between: int = 20  # Space between adjacent boxes

    # Content inside boxes
    content_horizontal_margin: int = 0  # Additional horizontal margin for content (titles align with border + padding)
    content_vertical_spacing: int = 8  # Space between stacked content items

    # List items
    list_item_padding_vertical: int = 4  # Vertical padding within each list item
    list_item_padding_horizontal: int = 0  # Horizontal padding within list item (uses content margin)
    list_item_spacing: int = 0  # Space between list items

    # Group headers in lists
    group_header_margin_top: int = 12  # Space above group header (first group has less)
    group_header_margin_bottom: int = 4  # Space below group header, before divider
    group_header_first_margin_top: int = 0  # No top margin for first group

    # Section dividers
    divider_margin_bottom: int = 4  # Space after divider before content


@dataclass
class ComponentSizes:
    """Component dimension tokens."""

    # Interactive elements
    button_height: int = 24
    button_padding_horizontal: int = 16
    button_padding_vertical: int = 2
    button_action_width: int = 80

    input_height: int = 35
    input_padding_horizontal: int = 10
    input_padding_vertical: int = 6

    # Windows
    main_window_width: int = 1000
    main_window_height: int = 600
    main_window_min_width: int = 1000
    main_window_min_height: int = 600

    # Dialogs
    dialog_width: int = 400
    dialog_min_height: int = 150
    dialog_message_max_width: int = 350
    sound_mapping_dialog_width: int = 500

    # Progress
    progress_bar_height: int = 5
    progress_bar_width: int = 300
    training_progress_width: int = 200
    training_progress_height: int = 20

    # Startup Window
    startup_width: int = 500
    startup_height: int = 250
    startup_logo_size: int = 110
    startup_spinner_width: int = 15

    # Dictation
    dictation_simple_width: int = 250
    dictation_simple_height: int = 100
    dictation_smart_width: int = 1000
    dictation_smart_height: int = 600


@dataclass
class SidebarLayout:
    """Sidebar layout configuration."""

    collapsed_width: int = 80
    expanded_width: int = 200
    animation_duration: int = 200
    padding_top: int = 20
    padding_horizontal: int = 10
    button_padding_left: int = 10
    button_padding_right: int = 10
    button_spacing_vertical: int = 6
    button_icon_size: int = 70
    button_min_height: int = 50
    logo_max_size: int = 38
    logo_padding_top: int = 0
    logo_padding_bottom: int = 30
    border_width: int = 1


@dataclass
class HeaderLayout:
    """Header layout configuration."""

    padding_horizontal: int = 30
    content_padding_left: int = 20  # Matches box_padding for vertical alignment with box titles
    content_padding_right: int = 20  # Matches box_padding for consistent spacing
    height: int = 100
    title_offset_y: int = 10
    title_y_offset: int = 0
    spacing: int = 2  # Reduced spacing between title and subtitle


@dataclass
class IconProperties:
    color: str = "#a3a3a3"
    full_logo_filename: str = "grey_icon_full_size.png"
    icon_logo_filename: str = "grey_icon_full_size.png"
    full_logo_apply_monochrome: bool = False
    icon_logo_apply_monochrome: bool = False


@dataclass
class ThemeConfig:
    """Aggregates all theme tokens."""

    fonts: FontSizes = field(default_factory=FontSizes)
    text: TextColors = field(default_factory=TextColors)
    shapes: ShapeColors = field(default_factory=ShapeColors)
    blue: BlueColors = field(default_factory=BlueColors)
    spacing: Spacing = field(default_factory=Spacing)
    radius: BorderRadius = field(default_factory=BorderRadius)
    container: ContainerLayout = field(default_factory=ContainerLayout)
    components: ComponentSizes = field(default_factory=ComponentSizes)
    sidebar: SidebarLayout = field(default_factory=SidebarLayout)
    header: HeaderLayout = field(default_factory=HeaderLayout)
    icon_properties: IconProperties = field(default_factory=IconProperties)

    font_family_primary: str = "DMSans"


class ThemeManager:
    """Manages theme configuration and resource loading."""

    def __init__(self):
        self.config = ThemeConfig()
        self._loaded_fonts = set()
        self._stylesheet: str = ""

    def load_stylesheet(self, qss_path: str = None) -> str:
        """Load the centralized QSS stylesheet.

        Args:
            qss_path: Optional path to QSS file. If None, uses default location.

        Returns:
            The loaded stylesheet content.
        """
        if qss_path is None:
            # Default location is alongside this module
            qss_path = Path(__file__).parent / "styles.qss"
        else:
            qss_path = Path(qss_path)

        if qss_path.exists():
            self._stylesheet = qss_path.read_text(encoding="utf-8")
        else:
            self._stylesheet = ""

        return self._stylesheet

    def apply_stylesheet(self, app) -> None:
        """Apply the loaded stylesheet to a QApplication.

        Args:
            app: QApplication instance to apply stylesheet to.
        """
        if not self._stylesheet:
            self.load_stylesheet()

        if self._stylesheet:
            app.setStyleSheet(self._stylesheet)

    def get_stylesheet(self) -> str:
        """Get the loaded stylesheet content.

        Returns:
            The stylesheet string, loading it if necessary.
        """
        if not self._stylesheet:
            self.load_stylesheet()
        return self._stylesheet

    def load_fonts(self, fonts_dir: str) -> None:
        """Load fonts from a directory."""
        path = Path(fonts_dir)
        if not path.exists():
            return

        for font_file in path.glob("**/*.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                self._loaded_fonts.update(families)

    def get_font_family(self, weight: str = "regular") -> str:
        """Get font family name."""
        if self.config.font_family_primary in self._loaded_fonts:
            return self.config.font_family_primary
        return self.config.font_family_primary

    def get_font(self, size: Any = "medium", weight: str = "regular", italic: bool = False, bold: bool = False) -> QFont:
        """Get a QFont object based on tokens."""
        # Handle size being int or string
        if isinstance(size, int):
            font_size = size
        else:
            font_size = getattr(self.config.fonts, size, self.config.fonts.medium)

        family = self.get_font_family(weight)

        font = QFont(family, font_size)

        if bold or weight == "bold":
            font.setWeight(QFont.Weight.Bold)
        elif weight == "semibold":
            font.setWeight(QFont.Weight.DemiBold)
        elif weight == "light":
            font.setWeight(QFont.Weight.Medium)

        if italic:
            font.setItalic(True)

        return font

    def get_monospace_font(self, size: Any = None) -> QFont:
        """Get monospace font."""
        if size is None:
            font_size = self.config.fonts.medium
        elif isinstance(size, int):
            font_size = size
        else:
            font_size = getattr(self.config.fonts, size, self.config.fonts.medium)

        return QFont(self.get_font_family(), font_size)

    def get_color(self, color_key: str) -> str:
        """Get color hex code by key (e.g. 'text.lightest')."""
        parts = color_key.split(".")
        if len(parts) == 2:
            category, name = parts
            if hasattr(self.config, category):
                cat_obj = getattr(self.config, category)
                return getattr(cat_obj, name, "#FF00FF")
        return "#FF00FF"

    def get_palette(self, bg_color: str, text_color: str):
        """Create a QPalette with specified colors.

        Args:
            bg_color: Background color hex
            text_color: Text/foreground color hex

        Returns:
            Configured QPalette
        """
        from PySide6.QtGui import QColor, QPalette

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(bg_color))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(text_color))
        palette.setColor(QPalette.ColorRole.Base, QColor(bg_color))
        palette.setColor(QPalette.ColorRole.Text, QColor(text_color))
        palette.setColor(QPalette.ColorRole.Button, QColor(bg_color))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(text_color))
        return palette

    def apply_colors_to_widget(self, widget, bg_color: str, text_color: str = None):
        """Apply colors to a widget programmatically.

        Args:
            widget: QWidget to style
            bg_color: Background color hex
            text_color: Optional text color hex
        """
        from PySide6.QtGui import QColor, QPalette

        palette = widget.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(bg_color))
        palette.setColor(QPalette.ColorRole.Base, QColor(bg_color))

        if text_color:
            palette.setColor(QPalette.ColorRole.WindowText, QColor(text_color))
            palette.setColor(QPalette.ColorRole.Text, QColor(text_color))

        widget.setPalette(palette)
        widget.setAutoFillBackground(True)

    def apply_frame_style(self, frame, border_color: str, bg_color: str, border_radius: int):
        """Apply frame styling programmatically.

        Args:
            frame: QFrame to style
            border_color: Border color hex
            bg_color: Background color hex
            border_radius: Border radius in pixels
        """
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QFrame

        # Set colors
        palette = frame.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(bg_color))
        frame.setPalette(palette)
        frame.setAutoFillBackground(True)

        # Store border info for custom painting if needed
        frame._theme_border_color = border_color
        frame._theme_border_radius = border_radius
        frame.setFrameShape(QFrame.Shape.NoFrame)


# Singleton instance
theme = ThemeManager()
