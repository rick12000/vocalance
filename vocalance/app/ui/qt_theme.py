"""Qt theme system with QSS stylesheets and theme management.

Provides centralized theme configuration for PySide6 UI with:
- QSS stylesheet generation
- Theme color constants
- Font configuration
- Component styling
- Runtime theme application
"""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


@dataclass
class FontSizes:
    """Font size design tokens for typography hierarchy."""

    small: int = 12
    medium: int = 15
    large: int = 17
    xlarge: int = 20
    xxlarge: int = 26


@dataclass
class TextColors:
    """Text color design tokens - 5 shades from light to dark."""

    color_accent: str = "#918f66"
    lightest: str = "#e8d6d6"
    light: str = "#c3afaf"
    medium: str = "#bdaaaa"
    dark: str = "#a79494"
    darkest: str = "#1f1f1f"
    success: str = "#28a745"
    streaming_token: str = "#c79b9b"


@dataclass
class ShapeColors:
    """Shape/background color design tokens - 5 shades from light to dark."""

    accent: str = "#b4c7c6"
    accent_minus: str = "#9dabaa"
    lightest: str = "#515151"
    light: str = "#404040"
    medium: str = "#393939"
    dark: str = "#2a2a2a"
    darkest: str = "#1c1c1c"


@dataclass
class AccentColors:
    """Accent color design tokens."""

    success: str = "#28a745"
    success_text: str = "#ffffff"


@dataclass
class GradientColors:
    """Gradient color design tokens for special text effects."""

    blue_rose_start: str = "#4a90e2"  # Blue
    blue_rose_end: str = "#a0657f"  # Rose


@dataclass
class Spacing:
    """Spacing design tokens."""

    none: int = 0
    tiny: int = 5
    small: int = 10
    medium: int = 15
    large: int = 20
    xlarge: int = 30


@dataclass
class BorderRadius:
    """Border radius design tokens."""

    small: int = 8
    medium: int = 10
    rounded: int = 20
    xlarge: int = 30


class IconPropertiesInstance:
    """Icon properties instance for runtime access."""

    def __init__(self):
        self.width_percentage = 0.42
        self.icon_text_spacing = 5

    @property
    def color(self) -> str:
        return ShapeColors().light


@dataclass
class Dimensions:
    """Layout dimension design tokens."""

    # Window dimensions
    main_window_width: int = 1000
    main_window_height: int = 600
    main_window_min_width: int = 1000
    main_window_min_height: int = 600
    header_height: int = 80

    # Component dimensions
    button_height: int = 30
    button_text_padding: int = 1
    entry_height: int = 35
    entry_height_standard: int = 35
    logo_size: int = 13
    sidebar_logo_size: int = 13

    # Entry field dimensions
    entry_width_small: int = 150
    entry_width_large: int = 300

    # Textbox dimensions
    textbox_height_small: int = 150

    # Progress bar dimensions
    training_progress_width: int = 200
    training_progress_height: int = 20

    # Dictation popup dimensions
    dictation_simple_width: int = 250
    dictation_simple_height: int = 100
    dictation_smart_width: int = 1000
    dictation_smart_height: int = 600

    # Startup window dimensions
    startup_width: int = 500
    startup_height: int = 250
    startup_logo_size: int = 110
    progress_bar_width: int = 300
    progress_bar_height: int = 5

    # Dialog dimensions
    dialog_width: int = 400
    dialog_min_height: int = 200
    dialog_content_width: int = 350
    sound_mapping_dialog_width: int = 400
    sound_mapping_dialog_min_height: int = 250
    dictation_view_dialog_width: int = 600
    dictation_view_dialog_min_height: int = 350
    command_dialog_width: int = 500
    command_dialog_min_height: int = 300


@dataclass
class SidebarLayout:
    """Sidebar layout configuration."""

    # Width for collapsed (icon-only) and expanded (icon + text) states
    collapsed_width: int = 80
    expanded_width: int = 200
    width: int = 80  # Default width (collapsed)
    border_width: int = 1

    # Animation
    animation_duration: int = 200  # milliseconds

    # Container padding
    container_padding_left: int = 0
    container_padding_right: int = 0
    container_padding_top: int = 0
    container_padding_bottom: int = 0

    # Logo configuration
    logo_max_size: int = 50
    logo_padding_left: int = 0
    logo_padding_right: int = 0
    logo_padding_top: int = 0
    logo_padding_bottom: int = 30

    # Button configuration
    button_padding_left: int = 10
    button_padding_right: int = 10
    button_spacing_vertical: int = 2
    button_hover_border_width: int = 1
    button_icon_size: int = 48  # Larger icons for collapsed state

    # Top spacing
    top_spacing: int = 20


@dataclass
class HeaderLayout:
    """Header layout configuration."""

    frame_padding_top: int = 20
    frame_padding_bottom: int = 0
    frame_padding_left: int = 25  # Must match TwoBoxLayout.outer_padding_left
    frame_padding_right: int = 25  # Must match TwoBoxLayout.outer_padding_right
    border_width: int = 1
    title_y_offset: int = 10
    subtitle_y_offset: int = 11
    content_padding_left: int = 30  # Aligns with box inner content
    content_padding_right: int = 30  # Aligns with box inner content


@dataclass
class TwoBoxLayout:
    """Configuration for two-box layout used across tabs."""

    base_spacing: int = 25
    box_border_width: int = 1
    inner_content_padx: int = 30

    # Outer padding - matches legacy values
    outer_padding_left: int = 25
    outer_padding_right: int = 25
    outer_padding_top: int = 25  # Top padding for outer border frame
    outer_padding_bottom: int = 25

    # Outer content border - wraps around entire view content
    outer_border_width: int = 3
    outer_border_padding: int = 15  # Padding between border and content


class ThemeManager:
    """Manages Qt theme application and font loading."""

    def __init__(self, asset_paths_config=None):
        """Initialize theme manager.

        Args:
            asset_paths_config: Optional asset paths configuration for font loading.
        """
        self.asset_paths_config = asset_paths_config
        self.font_sizes = FontSizes()
        self.text_colors = TextColors()
        self.shape_colors = ShapeColors()
        self.accent_colors = AccentColors()
        self.gradient_colors = GradientColors()
        self.spacing = Spacing()
        self.border_radius = BorderRadius()
        self.dimensions = Dimensions()
        self.sidebar_layout = SidebarLayout()
        self.header_layout = HeaderLayout()
        self.two_box_layout = TwoBoxLayout()
        self.icon_properties = IconPropertiesInstance()

        self._font_family = "Manrope"
        self._font_family_secondary = "Segoe UI"
        self._font_family_fallback = "Arial"
        self._font_family_monospace = "Courier New"
        self._loaded_fonts = set()

    def load_fonts(self) -> None:
        """Load custom fonts using Qt font database."""
        if not self.asset_paths_config:
            return

        try:
            font_dir = Path(self.asset_paths_config.fonts_dir)
            if not font_dir.exists():
                return

            # Load all font files
            for font_file in font_dir.glob("*.ttf"):
                font_id = QFontDatabase.addApplicationFont(str(font_file))
                if font_id != -1:
                    families = QFontDatabase.applicationFontFamilies(font_id)
                    self._loaded_fonts.update(families)
        except Exception as e:
            print(f"Error loading fonts: {e}")

    def get_font_family(self, weight: str = "regular") -> str:
        """Get font family name for specified weight.

        Args:
            weight: Font weight (regular, semibold, bold, etc.)

        Returns:
            Font family name string.
        """
        if self._font_family in self._loaded_fonts:
            return self._font_family
        return self._font_family_secondary

    def get_font(self, size: int = None, weight: str = "regular", bold: bool = False) -> QFont:
        """Create QFont object with specified parameters.

        Args:
            size: Font size in points (defaults to medium).
            weight: Font weight name.
            bold: Whether to make font bold.

        Returns:
            Configured QFont object.
        """
        if size is None:
            size = self.font_sizes.medium

        family = self.get_font_family(weight)
        font = QFont(family, size)

        if bold or weight == "bold":
            font.setWeight(QFont.Weight.Bold)
        elif weight == "semibold":
            font.setWeight(QFont.Weight.DemiBold)

        return font

    def get_monospace_font(self, size: int = None) -> QFont:
        """Get monospace font for code/fixed-width text.

        Args:
            size: Font size in points.

        Returns:
            Monospace QFont object.
        """
        if size is None:
            size = self.font_sizes.medium
        return QFont(self._font_family_monospace, size)

    def generate_stylesheet(self) -> str:
        """Generate complete QSS stylesheet for the application.

        Returns:
            QSS stylesheet string.
        """
        return f"""
/* =============================================================================
   VOCALANCE QT THEME - DARK MODE
   ============================================================================= */

/* Main Window */
QMainWindow {{
    background-color: {self.shape_colors.darkest};
    color: {self.text_colors.lightest};
}}

QWidget {{
    background-color: transparent;
    color: {self.text_colors.lightest};
    font-family: {self.get_font_family()};
    font-size: {self.font_sizes.medium}px;
}}

/* =============================================================================
   BUTTONS
   ============================================================================= */

QPushButton {{
    background-color: {self.shape_colors.accent};
    color: {self.shape_colors.dark};
    border: none;
    border-radius: {self.dimensions.button_height // 2}px;  /* Pill-shaped: radius = height/2 */
    padding: 4px 16px;
    font-weight: bold;
    font-size: {self.font_sizes.medium}px;
}}

QPushButton:hover {{
    background-color: {self.shape_colors.lightest};
}}

QPushButton:pressed {{
    background-color: {self.shape_colors.accent_minus};
}}

QPushButton:disabled {{
    background-color: {self.shape_colors.medium};
    color: {self.shape_colors.light};
}}

/* Primary Button */
QPushButton[buttonType="primary"] {{
    background-color: {self.shape_colors.accent};
    color: {self.shape_colors.dark};
    padding: 6px 12px;
}}

QPushButton[buttonType="primary"]:hover {{
    background-color: {self.shape_colors.accent_minus};
}}

/* Danger Button - matches legacy styling */
QPushButton[buttonType="danger"] {{
    background-color: {self.shape_colors.darkest};
    color: {self.text_colors.light};
    border: 1px solid {self.shape_colors.lightest};
    border-radius: {self.dimensions.button_height // 2}px;  /* Pill-shaped: radius = height/2 */
    padding: 4px 16px;
}}

QPushButton[buttonType="danger"]:hover {{
    background-color: {self.shape_colors.medium};
}}

QPushButton[buttonType="danger"]:pressed {{
    background-color: {self.shape_colors.dark};
}}

/* Sidebar Button */
QPushButton[buttonType="sidebar"] {{
    background-color: transparent;
    color: {self.text_colors.light};
    border: 2px solid transparent;
    border-radius: {self.border_radius.small}px;
    text-align: left;
    padding: 8px;
    font-weight: normal;
    qproperty-iconSize: {self.sidebar_layout.button_icon_size}px {self.sidebar_layout.button_icon_size}px;
}}

QPushButton[buttonType="sidebar"]:hover {{
    border: 2px solid {self.shape_colors.lightest};
    background-color: transparent;
}}

QPushButton[buttonType="sidebar"][selected="true"] {{
    border: 2px solid {self.shape_colors.accent};
    background-color: transparent;
}}

/* =============================================================================
   INPUT FIELDS
   ============================================================================= */

QLineEdit {{
    background-color: {self.shape_colors.darkest};
    color: {self.text_colors.light};
    border: 1px solid {self.shape_colors.medium};
    border-radius: {self.border_radius.small}px;
    padding: 6px 10px;
    min-height: {self.dimensions.entry_height}px;
    font-size: {self.font_sizes.medium}px;
}}

QLineEdit:focus {{
    border: 1px solid {self.shape_colors.accent};
}}

QLineEdit:disabled {{
    background-color: {self.shape_colors.dark};
    color: {self.shape_colors.light};
}}

QTextEdit, QPlainTextEdit {{
    background-color: {self.shape_colors.darkest};
    color: {self.text_colors.light};
    border: 1px solid {self.shape_colors.medium};
    border-radius: {self.border_radius.small}px;
    padding: 8px;
    font-size: {self.font_sizes.medium}px;
}}

QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {self.shape_colors.accent};
}}

/* =============================================================================
   COMBOBOX / DROPDOWN
   ============================================================================= */

QComboBox {{
    background-color: {self.shape_colors.darkest};
    color: {self.text_colors.light};
    border: 1px solid {self.shape_colors.medium};
    border-radius: {self.border_radius.small}px;
    padding: 6px 10px;
    min-height: {self.dimensions.entry_height}px;
}}

QComboBox:hover {{
    border: 1px solid {self.shape_colors.lightest};
}}

QComboBox:focus {{
    border: 1px solid {self.shape_colors.accent};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {self.text_colors.light};
    width: 0;
    height: 0;
}}

QComboBox QAbstractItemView {{
    background-color: {self.shape_colors.dark};
    color: {self.text_colors.light};
    border: 1px solid {self.shape_colors.medium};
    selection-background-color: {self.shape_colors.medium};
    selection-color: {self.text_colors.lightest};
}}

/* =============================================================================
   LABELS
   ============================================================================= */

QLabel {{
    background-color: transparent;
    background: transparent;
    color: {self.text_colors.lightest};
    border: none;
    border-width: 0px;
    border-color: transparent;
}}

QLabel[labelType="subtitle"] {{
    color: {self.text_colors.light};
}}

QLabel[labelType="title"] {{
    font-size: {self.font_sizes.xxlarge}px;
    font-weight: bold;
}}

/* =============================================================================
   FRAMES
   ============================================================================= */

QFrame {{
    background-color: {self.shape_colors.medium};
    border: 1px solid {self.shape_colors.medium};
    border-radius: {self.border_radius.medium}px;
}}

QFrame[frameType="box"] {{
    background-color: {self.shape_colors.darkest};
    border: 1px solid {self.shape_colors.medium};
    border-radius: {self.border_radius.rounded}px;
}}

QFrame[frameType="two_box"] {{
    background-color: {self.shape_colors.darkest};
    border: 1px solid {self.shape_colors.medium};
    border-radius: {self.border_radius.rounded}px;
}}

QFrame[frameType="tile"] {{
    background-color: {self.shape_colors.darkest};
    border: 1px solid {self.shape_colors.medium};
    border-radius: {self.border_radius.rounded}px;
}}

QFrame[frameType="header"] {{
    background-color: {self.shape_colors.darkest};
    border: 1px solid {self.shape_colors.medium};
    border-radius: {self.border_radius.xlarge}px;
}}

QFrame[frameType="sidebar"] {{
    background-color: {self.shape_colors.darkest};
    border: none;
}}

QFrame[frameType="transparent"] {{
    background-color: transparent;
    background: transparent;
    border: none;
    border-width: 0px;
}}

QFrame[frameType="content_border"] {{
    background-color: {self.shape_colors.darkest};
    border: {self.two_box_layout.outer_border_width}px solid {self.shape_colors.light};
    border-radius: {self.border_radius.small}px;
}}

/* List item frames should be borderless */
QWidget[itemType="list_item"] {{
    background-color: transparent;
    border: none;
    border-width: 0px;
}}

QFrame[itemType="list_item"] {{
    background-color: transparent;
    border: none;
    border-width: 0px;
}}

/* =============================================================================
   SCROLLBARS
   ============================================================================= */

QScrollBar:vertical {{
    background-color: {self.shape_colors.dark};
    width: 12px;
    border: none;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {self.shape_colors.light};
    min-height: 30px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {self.shape_colors.lightest};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {self.shape_colors.dark};
    height: 12px;
    border: none;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal {{
    background-color: {self.shape_colors.light};
    min-width: 30px;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {self.shape_colors.lightest};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* =============================================================================
   SCROLL AREA
   ============================================================================= */

QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}

/* =============================================================================
   PROGRESS BAR
   ============================================================================= */

QProgressBar {{
    background-color: {self.shape_colors.medium};
    color: {self.text_colors.lightest};
    border: none;
    border-radius: {self.dimensions.progress_bar_height // 2}px;
    text-align: center;
    height: {self.dimensions.progress_bar_height}px;
}}

QProgressBar::chunk {{
    background-color: {self.shape_colors.lightest};
    border-radius: {self.dimensions.progress_bar_height // 2}px;
}}

/* =============================================================================
   CHECKBOX / TOGGLE
   ============================================================================= */

QCheckBox {{
    color: {self.text_colors.light};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {self.shape_colors.light};
    border-radius: 3px;
    background-color: {self.shape_colors.dark};
}}

QCheckBox::indicator:checked {{
    background-color: {self.shape_colors.accent};
    border-color: {self.shape_colors.accent};
}}

QCheckBox::indicator:hover {{
    border-color: {self.shape_colors.lightest};
}}

/* =============================================================================
   DIALOGS
   ============================================================================= */

QDialog {{
    background-color: {self.shape_colors.darkest};
    color: {self.text_colors.lightest};
}}

QMessageBox {{
    background-color: {self.shape_colors.darkest};
}}

QMessageBox QLabel {{
    color: {self.text_colors.lightest};
}}

/* =============================================================================
   LIST WIDGETS
   ============================================================================= */

QListWidget {{
    background-color: transparent;
    border: none;
    outline: none;
}}

QListWidget::item {{
    background-color: transparent;
    border: none;
    padding: {self.spacing.tiny}px;
}}

QListWidget::item:selected {{
    background-color: transparent;
    border: none;
}}

QListWidget::item:hover {{
    background-color: transparent;
}}

/* =============================================================================
   TOOLTIPS
   ============================================================================= */

QToolTip {{
    background-color: {self.shape_colors.light};
    color: {self.text_colors.lightest};
    border: 1px solid {self.shape_colors.medium};
    border-radius: {self.border_radius.small}px;
    padding: 4px 8px;
}}

/* =============================================================================
   MENU
   ============================================================================= */

QMenu {{
    background-color: {self.shape_colors.dark};
    color: {self.text_colors.lightest};
    border: 1px solid {self.shape_colors.medium};
}}

QMenu::item {{
    padding: 8px 24px;
}}

QMenu::item:selected {{
    background-color: {self.shape_colors.medium};
}}

/* =============================================================================
   TAB WIDGET
   ============================================================================= */

QTabWidget::pane {{
    border: none;
    background-color: transparent;
}}

QTabBar::tab {{
    background-color: {self.shape_colors.dark};
    color: {self.text_colors.light};
    padding: 8px 16px;
    border: 1px solid {self.shape_colors.medium};
    border-bottom: none;
    border-top-left-radius: {self.border_radius.small}px;
    border-top-right-radius: {self.border_radius.small}px;
}}

QTabBar::tab:selected {{
    background-color: {self.shape_colors.medium};
    color: {self.text_colors.lightest};
}}

QTabBar::tab:hover {{
    background-color: {self.shape_colors.light};
}}

/* =============================================================================
   STACKED WIDGET
   ============================================================================= */

QStackedWidget {{
    background-color: transparent;
    border: none;
}}
"""

    def apply_theme(self, app: QApplication) -> None:
        """Apply theme stylesheet to Qt application.

        Args:
            app: QApplication instance to apply theme to.
        """
        stylesheet = self.generate_stylesheet()
        app.setStyleSheet(stylesheet)


# Global theme instance
theme_manager = ThemeManager()
