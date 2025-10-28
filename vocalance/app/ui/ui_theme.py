"""UI theme design tokens for consistent styling across the application.

Defines color palettes, font configurations, dimensions, spacing, and component-specific
styles using Pydantic models. Provides centralized theme management for all UI components.
"""
from pydantic import BaseModel


class FontSizes(BaseModel):
    """Font size design tokens for typography hierarchy."""

    small: int = 12
    medium: int = 15
    large: int = 17
    xlarge: int = 20
    xxlarge: int = 26


class FontFamily(BaseModel):
    """Font family design tokens"""

    primary: str = "Manrope"
    secondary: str = "Segoe UI"
    fallback: str = "Arial"
    monospace: str = "Courier"
    _font_service = None

    def set_font_service(self, font_service):
        """Set the font service for this font family"""
        self._font_service = font_service

    def get_primary_font(self, weight: str = "regular") -> str:
        """Get the primary font family with specified weight"""
        if self._font_service:
            try:
                return self._font_service.get_font_family(weight)
            except Exception:
                pass
        # Fallback to default fonts
        return self.secondary

    def get_monospace_font(self) -> str:
        """Get the monospace font family (e.g. for spinner animation)"""
        return self.monospace

    def get_button_font(self, size: int = None) -> tuple:
        """Get standardized button font configuration"""
        if size is None:
            size = FontSizes().medium
        font_family = self.get_primary_font("semibold")
        return (font_family, size, "normal")


class TextColors(BaseModel):
    """Text color design tokens - 5 shades from light to dark"""

    color_accent: str = "#918f66"
    lightest: str = "#e8d6d6"
    light: str = "#c3afaf"
    medium: str = "#bdaaaa"
    dark: str = "#a79494"
    darkest: str = "#1f1f1f"
    success: str = "#28a745"  # Green color for success states
    streaming_token: str = "#c79b9b"  # Highlight color for currently streaming token


class ShapeColors(BaseModel):
    """Shape/background color design tokens - 5 shades from light to dark"""

    accent: str = "#b4c7c6"
    accent_minus: str = "#9dabaa"
    lightest: str = "#494e4e"
    light: str = "#2a2c2c"
    medium: str = "#202020"
    dark: str = "#0f0f0f"
    darkest: str = "#0d0d0d"


class AccentColors(BaseModel):
    """Accent color design tokens"""

    success: str = "#28a745"
    success_text: str = "#ffffff"


class Spacing(BaseModel):
    """Spacing design tokens"""

    none: int = 0
    tiny: int = 5
    small: int = 10
    medium: int = 15
    large: int = 20
    xlarge: int = 30


class ButtonText(BaseModel):
    """Centralized button text values"""

    # Dialog buttons
    ok: str = "OK"
    cancel: str = "Cancel"
    yes: str = "Yes"
    no: str = "No"

    # Action buttons
    save: str = "Save"
    save_changes: str = "Save Changes"
    add: str = "Add"
    edit: str = "Edit"
    delete: str = "Delete"
    change: str = "Info"
    confirm: str = "Confirm"

    # Specific actions
    map: str = "Map"
    record: str = "Record"
    show_marks: str = "Show Marks"
    refresh: str = "Refresh"
    reset: str = "Reset"

    # Compound actions
    add_command: str = "Add"
    add_prompt: str = "Add"
    delete_all_sounds: str = "Delete All Sounds"
    delete_all_marks: str = "Delete All Marks"
    save_llm_settings: str = "Save LLM Settings"
    delete_command: str = "Delete Command"


class SidebarIcons(BaseModel):
    """Sidebar icon configuration"""

    marks: str = "circle-center-icon.png"
    sounds: str = "microphone-icon.png"
    commands: str = "speaking-bubbles-black-icon.png"
    dictation: str = "neural-network-black-icon.png"
    settings: str = "settings-icon.png"


class HeaderLayout(BaseModel):
    """
    Centralized header layout configuration

    This class provides centralized control over header frame positioning and spacing.
    The header frame is the dark rounded rectangle at the top of each tab that contains
    the tab title and subtitle.

    IMPORTANT: Header padding MUST align with TwoBoxLayout for visual consistency.
    Header frame_padding_left/right should match TwoBoxLayout.outer_padding_left/right
    """

    # Header frame padding (space around the header frame itself) - references Spacing
    @property
    def frame_padding_top(self) -> int:
        return Spacing().large

    @property
    def frame_padding_bottom(self) -> int:
        return Spacing().none

    # Header content padding - references TwoBoxLayout for consistency
    @property
    def content_padding_left(self) -> int:
        return TwoBoxLayout().inner_content_padx

    @property
    def content_padding_right(self) -> int:
        return TwoBoxLayout().inner_content_padx

    @property
    def content_padding_top(self) -> int:
        return Spacing().small

    @property
    def content_padding_bottom(self) -> int:
        return Spacing().small

    # Header border configuration
    border_width: int = 1

    @property
    def border_color(self) -> str:
        """Border color - shape_colors.light"""
        return ShapeColors().lightest

    @property
    def frame_padding_left(self) -> int:
        """Frame left padding - references TwoBoxLayout for alignment"""
        return TwoBoxLayout().outer_padding_left

    @property
    def frame_padding_right(self) -> int:
        """Frame right padding - references TwoBoxLayout for alignment"""
        return TwoBoxLayout().outer_padding_right

    # Title and subtitle positioning - references Spacing
    @property
    def title_y_offset(self) -> int:
        return Spacing().small

    @property
    def subtitle_y_offset(self) -> int:
        return Spacing().small + 1

    @property
    def frame_padx(self) -> tuple:
        """Get the horizontal padding tuple for header frame grid placement"""
        return (self.frame_padding_left, self.frame_padding_right)

    @property
    def frame_pady(self) -> tuple:
        """Get the vertical padding tuple for header frame grid placement"""
        return (self.frame_padding_top, self.frame_padding_bottom)

    @property
    def title_padx(self) -> int:
        """Get the horizontal padding for title content inside header"""
        return self.content_padding_left

    @property
    def subtitle_padx(self) -> int:
        """Get the horizontal padding for subtitle content inside header"""
        return self.content_padding_left


class SidebarLayout(BaseModel):
    """
    Centralized sidebar layout configuration

    This class provides a single source of truth for all sidebar dimensions and spacing.
    All sidebar-related components (buttons, logo, container) derive their sizes from
    these properties, ensuring consistency and easy customization.

    Key benefits:
    - Single place to control all sidebar dimensions
    - Automatic calculation of component sizes based on available space
    - No hardcoded values scattered across multiple files
    - Easy to modify padding without touching component code
    """

    # Core sidebar dimensions
    width: int = 120

    # Sidebar border configuration - custom right-only border
    border_width: int = 1
    use_custom_border: bool = True  # Use custom border implementation instead of CTkFrame border

    @property
    def border_color(self) -> str:
        """Border color - shape_colors.lightest"""
        return ShapeColors().lightest

    border_side: str = "right"  # Which side to show border on

    # Sidebar container padding (space around the entire sidebar)
    container_padding_left: int = 0
    container_padding_right: int = 0
    container_padding_top: int = 0
    container_padding_bottom: int = 0

    # Logo configuration
    logo_max_size: int = 50  # Further reduced for even smaller sidebar logo
    logo_padding_left: int = 0
    logo_padding_right: int = 0
    logo_padding_top: int = 0
    logo_padding_bottom: int = 30  # Increased to 20px for more space below the logo

    # Button configuration - increased padding for better spacing
    button_padding_left: int = 15  # Increased from 5
    button_padding_right: int = 15  # Increased from 5
    button_spacing_vertical: int = 2  # Space between buttons

    # Button hover styling - border instead of background
    button_hover_border_width: int = 1

    @property
    def button_hover_border_color(self) -> str:
        """Button hover border color - shape_colors.lightest"""
        return ShapeColors().lightest

    @property
    def effective_content_width(self) -> int:
        """Calculate the available width for sidebar content"""
        return self.width - self.container_padding_left - self.container_padding_right

    @property
    def logo_size(self) -> int:
        """Calculate the actual logo size that fits within constraints"""
        available_width = self.effective_content_width - self.logo_padding_left - self.logo_padding_right
        return min(self.logo_max_size, available_width)

    @property
    def button_width(self) -> int:
        """Calculate the actual button width that fits within constraints"""
        return self.effective_content_width - self.button_padding_left - self.button_padding_right

    @property
    def grid_column_minsize(self) -> int:
        """Get the minimum size for the grid column containing the sidebar"""
        return self.width


class TwoBoxLayout(BaseModel):
    """Configuration for two-box layout used across tabs"""

    @property
    def box_corner_radius(self) -> int:
        return BorderRadius().rounded

    @property
    def box_background_color(self) -> str:
        """Box background color - shape_colors.dark"""
        return ShapeColors().dark

    # Box border configuration
    box_border_width: int = 1

    @property
    def box_border_color(self) -> str:
        """Box border color - shape_colors.medium"""
        return ShapeColors().medium

    # Box spacing and padding
    # SINGLE SOURCE OF TRUTH - all spacing derives from base_spacing
    base_spacing: int = 25

    @property
    def outer_padding_left(self) -> int:
        return self.base_spacing

    @property
    def outer_padding_right(self) -> int:
        return self.base_spacing

    @property
    def outer_padding_top(self) -> int:
        return self.base_spacing

    @property
    def outer_padding_bottom(self) -> int:
        return self.base_spacing

    @property
    def inner_spacing(self) -> int:
        return self.base_spacing

    # Box content padding (inside the box frame) - references Spacing
    @property
    def box_content_padding(self) -> int:
        return Spacing().large

    # Inner content padding - used for form fields, list items, titles, and all nested elements
    inner_content_padx: int = 30  # Horizontal padding for all content inside boxes

    # Inner content vertical padding - references Spacing
    @property
    def inner_content_pady_small(self) -> int:
        return Spacing().tiny

    @property
    def inner_content_pady_medium(self) -> int:
        return Spacing().small

    # Title padding - horizontally aligned with inner content
    @property
    def title_padx_left(self) -> int:
        return self.inner_content_padx

    @property
    def title_padx_right(self) -> int:
        return self.inner_content_padx

    @property
    def title_padding_top(self) -> int:
        return Spacing().large

    @property
    def title_padding_bottom(self) -> int:
        return self.inner_content_pady_medium

    # Bottom padding for last element to show rounded corners
    @property
    def last_element_bottom_padding(self) -> int:
        return Spacing().large


class TileLayout(BaseModel):
    """Configuration for instruction tiles"""

    # Tile corner radius - references TwoBoxLayout for consistency
    @property
    def corner_radius(self) -> int:
        return TwoBoxLayout().box_corner_radius

    # Tile spacing - references Spacing
    @property
    def padding_between_tiles(self) -> int:
        return Spacing().small

    # Tile content styling
    title_font_size: str = "large"  # Increased from medium to large
    content_font_size: str = "small"  # Smaller content font as requested
    content_text_alignment: str = "center"  # Center align text as requested

    # Tile background and border
    border_width: int = 1

    @property
    def background_color(self) -> str:
        """Background color - shape_colors.medium"""
        return ShapeColors().darkest

    @property
    def border_color(self) -> str:
        """Border color - shape_colors.lightest"""
        return ShapeColors().lightest


class ListLayout(BaseModel):
    """Configuration for scrollable list layouts - applied automatically"""

    # Vertical spacing between list items - references Spacing
    @property
    def item_vertical_spacing(self) -> int:
        return Spacing().tiny

    # Horizontal padding for list items - references Spacing
    @property
    def item_padx(self) -> int:
        return Spacing().tiny

    @property
    def item_padx_right(self) -> int:
        return Spacing().tiny

    # Scrollable frame grid padding - references Spacing
    @property
    def scrollbar_right_padding(self) -> int:
        return Spacing().large

    # List container padding - minimal
    container_padding_x: int = 1
    container_padding_y: int = 1


class LogoProperties(BaseModel):
    """Logo styling properties"""

    full_logo_filename: str = "grey_red_icon_full_size.png"
    icon_logo_filename: str = "grey_red_icon_full_size.png"

    # Monochrome conversion toggles
    full_logo_apply_monochrome: bool = False
    icon_logo_apply_monochrome: bool = False

    @property
    def color(self) -> str:
        """Logo color - shape_colors.light"""
        return ShapeColors().medium


class EntryFieldStyling(BaseModel):
    """Entry field styling configuration"""

    border_width: int = 1
    corner_radius: int = 8  # border_radius.small

    @property
    def border_color(self) -> str:
        """Border color - shape_colors.lightest"""
        return ShapeColors().medium

    @property
    def background_color(self) -> str:
        """Background color - shape_colors.dark"""
        return ShapeColors().darkest


class IconProperties(BaseModel):
    """Icon styling properties"""

    width_percentage: float = 0.35
    color: str = ShapeColors().medium
    icon_text_spacing: int = 5


class BorderRadius(BaseModel):
    """Border radius design tokens"""

    small: int = 8
    medium: int = 10
    rounded: int = 20
    xlarge: int = 30


class Dimensions(BaseModel):
    """Layout dimension design tokens"""

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


class LayoutProperties(BaseModel):
    """Layout positioning and spacing properties"""

    # Content area positioning
    # NO padding - TwoBoxLayout.outer_padding controls ALL spacing
    content_area_padding_x: int = 0
    content_area_padding_y: int = 0  # Let TwoBoxLayout.outer_padding control vertical spacing


class Theme(BaseModel):
    """Main theme containing all design tokens"""

    font_sizes: FontSizes = FontSizes()
    font_family: FontFamily = FontFamily()
    text_colors: TextColors = TextColors()
    shape_colors: ShapeColors = ShapeColors()
    accent_colors: AccentColors = AccentColors()
    button_text: ButtonText = ButtonText()
    sidebar_icons: SidebarIcons = SidebarIcons()
    header_layout: HeaderLayout = HeaderLayout()
    sidebar_layout: SidebarLayout = SidebarLayout()
    two_box_layout: TwoBoxLayout = TwoBoxLayout()  # New configuration
    tile_layout: TileLayout = TileLayout()  # New configuration
    list_layout: ListLayout = ListLayout()  # New configuration
    logo_properties: LogoProperties = LogoProperties()
    entry_field_styling: EntryFieldStyling = EntryFieldStyling()
    icon_properties: IconProperties = IconProperties()
    spacing: Spacing = Spacing()
    border_radius: BorderRadius = BorderRadius()
    dimensions: Dimensions = Dimensions()
    layout: LayoutProperties = LayoutProperties()


# Global theme instance
theme = Theme()
