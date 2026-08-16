import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

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
    orange: str = "#ff8c00"
    pause_yellow: str = "#f5c518"
    pause_yellow_dark: str = "#e6a800"


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
    """Spacing between container chrome, padding, and stacked content."""

    box_padding: int = 16
    box_spacing_between: int = 14
    box_title_spacing: int = 12

    content_horizontal_margin: int = 0
    content_vertical_spacing: int = 10

    list_item_padding_vertical: int = 4
    list_item_padding_horizontal: int = 0
    list_item_spacing: int = 0

    group_header_margin_top: int = 12
    group_header_margin_bottom: int = 4
    group_header_first_margin_top: int = 0

    divider_margin_bottom: int = 4


@dataclass
class ComponentSizes:
    """Component dimension tokens."""

    button_height: int = 28
    button_padding_horizontal: int = 16
    button_padding_vertical: int = 2
    button_action_width: int = 80

    input_height: int = 36
    input_padding_horizontal: int = 12
    input_padding_vertical: int = 7

    main_window_width: int = 1000
    main_window_height: int = 600
    main_window_min_width: int = 1000
    main_window_min_height: int = 600

    dialog_width: int = 400
    dialog_min_height: int = 150
    dialog_message_max_width: int = 350
    sound_mapping_dialog_width: int = 500

    # Progress
    progress_bar_height: int = 5
    progress_bar_width: int = 300

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
    content_padding_left: int = 18
    content_padding_right: int = 18
    height: int = 88
    title_offset_y: int = 10
    title_y_offset: int = 0
    spacing: int = 4
    padding_bottom: int = 16
    icon_size: int = 36
    text_icon_spacing: int = 20


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

    font_family_primary: str = "DM Sans"
    font_family_display: str = "Alata"


class ThemeManager:
    """Loads fonts, builds QSS, and exposes token-backed ``ThemeConfig``."""

    def __init__(self) -> None:
        self.config = ThemeConfig()
        self._registered_font_families: set[str] = set()
        self._cached_stylesheet_text: str = ""

    def load_stylesheet(self, qss_path: str | None = None) -> str:
        """Build the application stylesheet from packaged QSS partials and theme tokens.

        Optional ``qss_path`` appends an additional file after the built stylesheet.
        """
        from vocalance.app.ui.style.builder import build_app_stylesheet

        self._cached_stylesheet_text = build_app_stylesheet(self)
        if qss_path is not None:
            extra = Path(qss_path)
            if extra.is_file():
                self._cached_stylesheet_text = self._cached_stylesheet_text + "\n" + extra.read_text(encoding="utf-8")
        return self._cached_stylesheet_text

    def apply_stylesheet(self, app) -> None:
        """Apply the loaded stylesheet and palette to a QApplication.

        Args:
            app: QApplication instance to apply stylesheet to.
        """
        if not self._cached_stylesheet_text:
            self.load_stylesheet()

        if self._cached_stylesheet_text:
            app.setStyleSheet(self._cached_stylesheet_text)

        self._paint_application_palette(app)

    def _paint_application_palette(self, app) -> None:
        """Set highlight colors on ``app`` so controls match theme tokens."""
        from PySide6.QtGui import QColor, QPalette

        palette = app.palette()

        palette.setColor(QPalette.ColorRole.Highlight, QColor(self.config.blue.blue_2))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(self.config.text.lightest))

        app.setPalette(palette)

    def get_stylesheet(self) -> str:
        """Get the loaded stylesheet content.

        Returns:
            The stylesheet string, loading it if necessary.
        """
        if not self._cached_stylesheet_text:
            self.load_stylesheet()
        return self._cached_stylesheet_text

    def load_fonts(self, fonts_dir: str = None) -> None:
        """Load custom fonts (Alata and DM Sans) from the fonts directory.

        Args:
            fonts_dir: Optional path to fonts directory. If None, uses default fonts location.
        """
        if fonts_dir is None:
            fonts_base = Path(__file__).parent.parent / "assets" / "fonts"
        else:
            fonts_base = Path(fonts_dir).resolve()

        alata_dir = fonts_base / "Alata"
        self._register_fonts_in_directory(alata_dir, "Alata")

        dmsans_dir = fonts_base / "DM_Sans"
        self._register_fonts_in_directory(dmsans_dir, "DM Sans")

        if "DM Sans" in self._registered_font_families:
            self.config.font_family_primary = "DM Sans"
        if "Alata" in self._registered_font_families:
            self.config.font_family_display = "Alata"

        logger.debug(
            "Fonts loaded primary=%s display=%s",
            self.config.font_family_primary,
            self.config.font_family_display,
        )

    def _register_fonts_in_directory(self, fonts_dir: Path, expected_family: str) -> bool:
        """Register every ``.ttf`` under ``fonts_dir`` with Qt.

        Args:
            fonts_dir: Directory containing font files.
            expected_family: Human-readable family label for logs.

        Returns:
            True if at least one face was registered.
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
                self._registered_font_families.update(families)
                loaded_count += 1

        if loaded_count > 0:
            logger.debug("Registered %s font file(s) for %s", loaded_count, expected_family)
            return True
        logger.warning("No fonts loaded from %s directory", expected_family)
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
        self,
        size: Union[int, str] = "medium",
        weight: str = "regular",
        italic: bool = False,
        bold: bool = False,
        display: bool = False,
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
        if isinstance(size, int):
            font_size = size
        else:
            font_size = getattr(self.config.fonts, size, self.config.fonts.medium)

        family = self.get_font_family(weight, display=display)

        font = QFont(family, font_size)

        if bold or weight == "bold":
            font.setWeight(QFont.Weight(550))
        elif weight == "semibold":
            font.setWeight(QFont.Weight(530))
        elif weight == "light":
            font.setWeight(QFont.Weight(200))

        if italic:
            font.setItalic(True)

        return font

    def get_monospace_font(self, size: Optional[Union[int, str]] = None) -> QFont:
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
        palette.setColor(QPalette.ColorRole.Highlight, QColor(self.config.blue.blue_2))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(self.config.text.lightest))

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

        palette.setColor(QPalette.ColorRole.Highlight, QColor(self.config.blue.blue_2))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(self.config.text.lightest))

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

        palette = frame.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(bg_color))
        frame.setPalette(palette)
        frame.setAutoFillBackground(True)

        frame._theme_border_color = border_color
        frame._theme_border_radius = border_radius
        frame.setFrameShape(QFrame.Shape.NoFrame)


# Singleton instance
theme = ThemeManager()
