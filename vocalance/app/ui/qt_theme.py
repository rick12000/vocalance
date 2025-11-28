"""Qt theme system with QSS stylesheets and theme management.

Provides centralized theme configuration for PySide6 UI with:
- Design tokens: Text colors (lightest, light, medium), Shape colors (accent through darkest)
- Blue accent colors for interactive elements
- Spacing and border radius scales
- Container and layout spacing hierarchy
- Component dimensions and positioning
- Runtime theme application and stylesheet management
- Dual custom font system (Alata + DM Sans)

FONTS:
- Alata: Display font for titles, headers, and prominent text elements
- DM Sans: Primary font for body text and general UI elements
- Both fonts are loaded from custom directories, not system fonts
- Components can specify display=True to use Alata, otherwise default to DM Sans

SPACING HIERARCHY:
1. Container Border (1px) - outermost boundary
2. Container Padding - space between border and content (box_padding)
3. Content Margin - space for content inside containers
4. Item Spacing - space between items in lists/grids
5. Element Padding - internal padding within elements
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtGui import QFont, QFontDatabase

logger = logging.getLogger(__name__)


@dataclass
class FontSizes:
    """Font size design tokens."""

    small: int = 10
    medium: int = 12
    moderate: int = 16
    large: int = 20
    xlarge: int = 24
    xxlarge: int = 30


@dataclass
class TextColors:
    """Text color design tokens."""

    lightest: str = "#f4f4f5"
    light: str = "#a1a1aa"
    medium: str = "#52525b"
    gradient_colors: list = field(default_factory=lambda: ["#696AC2", "#CCD8E3"])
    title_gradient: list = field(default_factory=lambda: ["#696AC2", "#CCD8E3"])


@dataclass
class ShapeColors:
    """Shape/background color design tokens."""

    accent: str = "#71717a"
    lightest: str = "#52525b"
    light: str = "#3f3f46"
    medium: str = "#1d1d22"
    dark: str = "#141417"
    darkest: str = "#0f0f11"


@dataclass
class BlueColors:
    """Blue color design tokens."""

    blue_1: str = "#1d1d22"
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

    small: int = 4
    medium: int = 8
    rounded: int = 15


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
    box_title_spacing: int = 16  # Space after box title before content

    # Content inside boxes
    content_horizontal_margin: int = 0  # Additional horizontal margin for content (titles align with border + padding)
    content_vertical_spacing: int = 14  # Space between stacked content items

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

    # Startup Window
    startup_width: int = 500
    startup_height: int = 200
    startup_logo_size: int = 110
    startup_spinner_width: int = 15


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
    button_icon_text_spacing: int = 4


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
    padding_bottom: int = 20  # Space after header content (subtitle) - reduced from 20
    icon_size: int = 55  # Size of header icon button
    text_icon_spacing: int = 20  # Space between text and icon in header button


@dataclass
class IconProperties:
    color: str = "#71717a"
    full_logo_filename: str = "grey_icon_full_size.png"
    icon_logo_filename: str = "grey_icon_full_size.png"
    full_logo_apply_monochrome: bool = False
    icon_logo_apply_monochrome: bool = False
    documentation_icon_filename: str = "book_4_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"


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

    # Font family names (updated from TTF files after load_fonts() is called)
    font_family_primary: str = "DM Sans"  # Default font for most UI elements
    font_family_display: str = "Alata"  # Display font for titles and headers


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

    def load_fonts(self, fonts_dir: str = None) -> None:
        """Load custom fonts (Alata and DM Sans) from the fonts directory.

        Args:
            fonts_dir: Optional path to fonts directory. If None, uses default fonts location.
        """
        if fonts_dir is None:
            fonts_base = Path(__file__).parent.parent / "assets" / "fonts"
        else:
            fonts_base = Path(fonts_dir).parent

        # Load Alata (display font for titles)
        alata_dir = fonts_base / "custom_Alata"
        self._load_font_family(alata_dir, "Alata")

        # Load DM Sans (primary font for body text)
        dmsans_dir = fonts_base / "DM_Sans"
        self._load_font_family(dmsans_dir, "DM Sans")

        # Update config with loaded font families
        if "DM Sans" in self._loaded_fonts:
            self.config.font_family_primary = "DM Sans"
        if "Alata" in self._loaded_fonts:
            self.config.font_family_display = "Alata"

        logger.info(f"Fonts loaded - Primary: {self.config.font_family_primary}, Display: {self.config.font_family_display}")

    def _load_font_family(self, fonts_dir: Path, expected_family: str) -> bool:
        """Load all TTF files from a font directory.

        Args:
            fonts_dir: Path to font directory
            expected_family: Expected font family name for logging

        Returns:
            True if fonts were loaded successfully
        """
        if not fonts_dir.exists():
            logger.warning(f"Font directory not found: {fonts_dir}")
            return False

        loaded_count = 0
        for font_file in fonts_dir.glob("**/*.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                self._loaded_fonts.update(families)
                loaded_count += 1
                logger.debug(f"Loaded font: {font_file.name} -> families: {families}")
            else:
                logger.warning(f"Failed to load font: {font_file}")

        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} {expected_family} font files from {fonts_dir.name}")
            return True
        else:
            logger.error(f"No fonts loaded from {fonts_dir}")
            return False

    def get_font_family(self, weight: str = "regular", display: bool = False) -> str:
        """Get font family name from loaded fonts.

        Args:
            weight: Font weight (unused, kept for compatibility)
            display: If True, returns display font (Alata). If False, returns primary font (DM Sans).

        Returns:
            The configured font family name.
        """
        if display:
            return self.config.font_family_display
        return self.config.font_family_primary

    def get_font(
        self, size: Any = "medium", weight: str = "regular", italic: bool = False, bold: bool = False, display: bool = False
    ) -> QFont:
        """Get a QFont object based on tokens.

        Args:
            size: Font size (int or string like 'medium', 'large')
            weight: Font weight ('regular', 'semibold', 'bold', 'light')
            italic: Whether to italicize
            bold: Whether to bold
            display: If True, uses display font (Alata); if False, uses primary font (DM Sans)
        """
        # Handle size being int or string
        if isinstance(size, int):
            font_size = size
        else:
            font_size = getattr(self.config.fonts, size, self.config.fonts.medium)

        family = self.get_font_family(weight, display=display)

        font = QFont(family, font_size)

        if bold or weight == "bold":
            font.setWeight(QFont.Weight(550))  # Reduced from Bold (700) by ~14% (DemiBold equivalent)
        elif weight == "semibold":
            font.setWeight(QFont.Weight(530))  # Reduced from DemiBold (600) by ~17% (Medium equivalent)
        elif weight == "light":
            font.setWeight(QFont.Weight(200))

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
