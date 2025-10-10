"""
UI Theme system for Iris Control Room
Core design tokens using Pydantic models
"""

from pydantic import BaseModel
from typing import Tuple


class FontSizes(BaseModel):
    """Font size design tokens"""
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
    
    def get_primary_font(self, weight: str = "regular") -> str:
        """Get the primary font family with specified weight"""
        try:
            from iris.app.ui.utils.font_service import get_font_service
            font_service = get_font_service()
            return font_service.get_font_family(weight)
        except ImportError:
            # Fallback if font service is not available
            return self.secondary
    
    def get_button_font(self, size: int = None) -> tuple:
        """Get standardized button font configuration"""
        if size is None:
            from iris.app.ui.ui_theme import theme
            size = theme.font_sizes.medium
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


class ShapeColors(BaseModel):
    """Shape/background color design tokens - 5 shades from light to dark"""
    accent: str = "#b4c7c6"
    lightest: str = "#494e4e"
    light: str = "#2a2c2c"
    medium: str = "#222424"
    dark: str = "#131515"
    darkest: str = "#111111"


class AccentColors(BaseModel):
    """Accent color design tokens"""
    primary: str = "#c79b9b"
    primary_hover: str = "#bd9393"
    primary_text: str = "#625353"
    
    danger: str = "#ffffff"
    danger_hover: str = "#ffffff"
    danger_text: str = "#2c2626"
    
    success: str = "#28a745"
    success_hover: str = "#218838"
    success_text: str = "#ffffff"
    
    warning: str = "#ffc107"
    warning_hover: str = "#e0a800"
    warning_text: str = "#212529"

class Spacing(BaseModel):
    """Spacing design tokens"""
    none: int = 0
    tiny: int = 5
    small: int = 10
    medium: int = 15
    large: int = 20
    xlarge: int = 30
    frame_padding: int = 20

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
    change: str = "Change"
    confirm: str = "Confirm"
    
    # Specific actions
    map: str = "Map"
    record: str = "Record"
    show_marks: str = "Show Marks"
    refresh: str = "Refresh"
    reset: str = "Reset to Defaults"
    
    # Compound actions
    add_command: str = "Add Hotkey Command"
    add_prompt: str = "Add Custom Prompt"
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
    """

    # Header frame padding (space around the header frame itself)
    frame_padding_left: int = 50
    frame_padding_right: int = 50
    frame_padding_top: int = 20
    frame_padding_bottom: int = 10

    # Header content padding (space inside the header frame)
    content_padding_left: int = 30
    content_padding_right: int = 30
    content_padding_top: int = 10
    content_padding_bottom: int = 10

    # Header border configuration
    border_width: int = 1

    @property
    def border_color(self) -> str:
        """Border color - shape_colors.medium"""
        return ShapeColors().medium
    
    # Title and subtitle positioning
    title_y_offset: int = 10
    subtitle_y_offset: int = 11
    
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
    logo_max_size: int = 70  # Reduced from 100 to 50 for smaller sidebar logo
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
    
    # Box corner radius - more rounded for modern look
    box_corner_radius: int = 20  # Increased from default medium (10)
    
    @property
    def box_background_color(self) -> str:
        """Box background color - shape_colors.dark"""
        return ShapeColors().dark
    
    # Box spacing and padding
    outer_padding_left: int = 50
    outer_padding_right: int = 50
    outer_padding_top: int = 25
    outer_padding_bottom: int = 25
    inner_spacing: int = 25  # Space between left and right boxes
    
    # Box content padding (inside the box frame)
    box_content_padding: int = 20  # Padding inside boxes
    
    # Title positioning inside box
    title_padding_top: int = 20
    title_padding_bottom: int = 10
    
    # Bottom padding for last element to show rounded corners
    last_element_bottom_padding: int = 20


class TileLayout(BaseModel):
    """Configuration for instruction tiles"""
    
    # Tile corner radius - matches box corner radius for consistency
    corner_radius: int = 20  # Increased from default medium (10)
    
    # Tile spacing - increased for better visual separation
    padding_between_tiles: int = 10  # Increased from 5
    
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
        """Border color - shape_colors.medium"""
        return ShapeColors().medium


class ListLayout(BaseModel):
    """Configuration for scrollable list layouts"""
    
    # Vertical spacing between list items - greatly reduced as requested
    item_vertical_spacing: int = Spacing().tiny  # Reduced from 3 to 1 for tighter spacing
    
    # List container padding
    container_padding_x: int = 1
    container_padding_y: int = 1


class LogoProperties(BaseModel):
    """Logo styling properties"""
    filename: str = "iris_logo_full_size.png"
    
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
    # Base icon size in pixels
    base_size: int = 30
    
    # Icon size multiplier (1.0 = base size, 0.5 = half size, 2.0 = double size)
    size_multiplier: float = 1.2
    
    # Icon size as percentage of available button width for responsive sizing
    # Set to None to use base_size * size_multiplier instead
    width_percentage: float = 0.35
    
    # Icon color (hex color for monochrome icon recoloring)
    color: str = ShapeColors().medium
    
    # Spacing between icon and text (vertical padding)
    icon_text_spacing: int = 5


class BorderRadius(BaseModel):
    """Border radius design tokens"""
    none: int = 0
    small: int = 8
    medium: int = 10
    large: int = 15
    xlarge: int = 30
    # New rounded option for modern look
    rounded: int = 20  # More rounded than large, less than xlarge


class Dimensions(BaseModel):
    """Layout dimension design tokens"""
    # Window dimensions
    main_window_width: int = 1000
    main_window_height: int = 600
    main_window_min_width: int = 1000
    main_window_min_height: int = 600
    header_height: int = 80
    
    # Component dimensions
    button_height: int = 35
    entry_height: int = 35
    entry_height_standard: int = 35
    logo_size: int = 15
    sidebar_logo_size: int = 15
    
    # Button dimensions
    button_width_small: int = 60
    button_width_medium: int = 80
    button_width_large: int = 100
    button_width_xlarge: int = 120
    
    # Entry field dimensions
    entry_width_small: int = 150
    entry_width_medium: int = 200
    entry_width_large: int = 300
    entry_width_xlarge: int = 400
    
    # Textbox dimensions
    textbox_height_small: int = 150
    textbox_height_medium: int = 200
    textbox_height_large: int = 300
    
    # Progress bar dimensions
    training_progress_width: int = 200
    training_progress_height: int = 20
    
    # Text wrapping
    text_wrap_small: int = 250
    text_wrap_medium: int = 350
    text_wrap_large: int = 450
    
    # Dictation popup dimensions
    dictation_simple_width: int = 250
    dictation_simple_height: int = 100
    dictation_smart_width: int = 1000
    dictation_smart_height: int = 600
    
    # Startup window dimensions
    startup_width: int = 500
    startup_height: int = 300
    startup_logo_size: int = 150
    progress_bar_width: int = 300
    progress_bar_height: int = 20
    
    # Dialog dimensions
    dialog_width: int = 400
    dialog_height: int = 200
    dialog_content_width: int = 350
    sound_mapping_dialog_width: int = 400
    sound_mapping_dialog_height: int = 300
    dictation_view_dialog_width: int = 600
    dictation_view_dialog_height: int = 350
    command_dialog_width: int = 500
    command_dialog_height: int = 500
    
    # Layout spacing
    tab_box_padding: int = 20
    content_padding: int = 10
    title_shift_right: int = 15
    
    # Legacy dimensions for compatibility
    two_box_min_width: int = 300
    two_box_min_height: int = 500
    
    @property
    def sidebar_width(self) -> int:
        """Backward compatibility property - use sidebar_layout.width instead"""
        return 80  # Default fallback, should use theme.sidebar_layout.width


class LayoutProperties(BaseModel):
    """Layout positioning and spacing properties"""
    # Tab layout
    tab_title_row_weight: int = 0
    tab_content_row_weight: int = 1
    left_panel_weight: int = 1
    right_panel_weight: int = 1
    
    # Box positioning
    tab_box_outer_padding_left: int = 50
    tab_box_outer_padding_right: int = 50
    tab_box_outer_padding_top: int = 25
    tab_box_outer_padding_bottom: int = 25
    tab_box_inner_spacing: int = 25
    
    # Title positioning
    tab_title_padding_x: int = 10
    tab_title_padding_y: int = 10
    tab_title_shift_right: int = 15
    
    # Content area positioning
    content_area_padding_x: int = 10
    content_area_padding_y: int = 5


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