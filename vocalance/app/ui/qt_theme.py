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
    moderate: int = 15
    large: int = 19
    xlarge: int = 22
    xxlarge: int = 28


@dataclass
class TextColors:
    """Text color design tokens."""

    lightest: str = "#e8e8ed"
    light: str = "#94a3b8"
    medium: str = "#64748b"
    gradient_colors: list = field(default_factory=lambda: ["#6f7699", "#8e96b8", "#a8b0cc"])
    title_gradient: list = field(default_factory=lambda: ["#7a80a3", "#959bb8", "#b0b6cc"])


@dataclass
class ShapeColors:
    """Shape/background color design tokens."""

    accent: str = "#788499"
    lightest: str = "#32323a"
    light: str = "#26262e"
    medium: str = "#1a1a22"
    dark: str = "#121218"
    darkest: str = "#0c0c10"
    orange: str = "#ff8c00"  # Orange color for dictation stop word indicator


@dataclass
class BlueColors:
    """Blue color design tokens."""

    blue_1: str = "#16161e"
    blue_2: str = "#98a3df"


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
    medium: int = 12
    rounded: int = 14


@dataclass
class ContainerLayout:
    """Container spacing system - defines relationship between containers and content.

    HIERARCHY:
    - border: 1px solid line (part of container style)
    - padding: space from border to content area
    - content_margin: additional margin for content widgets inside container
    """

    # Box containers (primary content boxes)
    box_padding: int = 16  # Padding from border to content
    box_spacing_between: int = 14  # Space between adjacent boxes
    box_title_spacing: int = 12  # Space after box title before content

    # Content inside boxes
    content_horizontal_margin: int = 0  # Additional horizontal margin for content (titles align with border + padding)
    content_vertical_spacing: int = 10  # Space between stacked content items

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
    button_height: int = 28
    button_padding_horizontal: int = 16
    button_padding_vertical: int = 2
    button_action_width: int = 80

    input_height: int = 36
    input_padding_horizontal: int = 12
    input_padding_vertical: int = 7

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
    animation_duration: int = 260
    padding_top: int = 18
    padding_horizontal: int = 0
    # Icons + logo are centered in collapsed_width; no extra L/R inset on the rail.
    button_padding_left: int = 0
    button_padding_right: int = 0
    button_spacing_vertical: int = 5
    button_icon_size: int = 36
    button_min_height: int = 44
    logo_max_size: int = 36
    logo_padding_top: int = 0
    logo_padding_bottom: int = 22
    border_width: int = 1
    button_icon_text_spacing: int = 4


@dataclass
class HeaderLayout:
    """Header layout configuration."""

    padding_horizontal: int = 28
    content_padding_left: int = 18  # Matches box_padding for vertical alignment with box titles
    content_padding_right: int = 18  # Matches box_padding for consistent spacing
    height: int = 88
    title_offset_y: int = 10
    title_y_offset: int = 0
    spacing: int = 4
    padding_bottom: int = 16
    icon_size: int = 36
    text_icon_spacing: int = 20  # Space between text and icon in header button


@dataclass
class IconProperties:
    color: str = "#788499"
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
        """Apply the loaded stylesheet and palette to a QApplication.

        Args:
            app: QApplication instance to apply stylesheet to.
        """
        if not self._stylesheet:
            self.load_stylesheet()

        if self._stylesheet:
            app.setStyleSheet(self._stylesheet)

        # Apply custom palette to override OS theme colors
        self._apply_app_palette(app)

    def _apply_app_palette(self, app) -> None:
        """Apply our custom palette to the QApplication to override OS theme.

        This ensures that selection colors, radio button indicators, and other
        OS-dependent colors use our theme colors instead of the system colors.

        Palette inheritance model:
        - QApplication palette is the base - inherited by all widgets
        - Widgets can override by calling setPalette() with their own palette
        - This creates a clean cascade where specific colors are overridden locally
        - Palette colors are INDEPENDENT of QSS stylesheets (separate systems)

        Args:
            app: QApplication instance to apply palette to.
        """
        from PySide6.QtGui import QColor, QPalette

        palette = app.palette()

        # Override OS theme selection colors with our theme colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(self.config.blue.blue_2))  # Selection background
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(self.config.text.lightest))  # Selection text

        app.setPalette(palette)

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
            fonts_base = Path(fonts_dir).resolve()

        # Load Alata (display font for titles)
        alata_dir = fonts_base / "Alata"
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
            True if at least one font was loaded successfully, False otherwise
        """
        fonts_dir = fonts_dir.resolve()

        if not fonts_dir.exists():
            logger.warning(f"Font directory not found: {fonts_dir}")
            return False

        loaded_count = 0
        for font_file in fonts_dir.glob("**/*.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file.absolute()))
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                self._loaded_fonts.update(families)
                loaded_count += 1

        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} {expected_family} font files")
            return True
        else:
            logger.warning(f"No fonts loaded from {expected_family} directory")
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

        Returns:
            QFont object using the custom loaded font family
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
        """Create a QPalette with specified colors and our theme's selection colors.

        This method ensures all palettes created through the theme include our custom
        selection colors, maintaining consistency across the application even if
        QApplication palette isn't applied yet.

        Args:
            bg_color: Background color hex
            text_color: Text/foreground color hex

        Returns:
            Configured QPalette with our theme selection colors included
        """
        from PySide6.QtGui import QColor, QPalette

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(bg_color))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(text_color))
        palette.setColor(QPalette.ColorRole.Base, QColor(bg_color))
        palette.setColor(QPalette.ColorRole.Text, QColor(text_color))
        palette.setColor(QPalette.ColorRole.Button, QColor(bg_color))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(text_color))

        # Always include our custom theme selection colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(self.config.blue.blue_2))  # Selection background
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(self.config.text.lightest))  # Selection text

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

        # Override OS theme selection colors with our theme colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(self.config.blue.blue_2))  # Selection background
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(self.config.text.lightest))  # Selection text

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
