"""
View configuration models using Pydantic that integrate with ui_theme and settings_service.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field

from iris.app.ui import ui_theme


class ViewTimings(BaseModel):
    """Timing configurations for view operations"""

    reset_interface_delay_ms: int = Field(default=3000, description="Delay before resetting interface after operations")
    refresh_list_delay_ms: int = Field(default=500, description="Delay for refreshing lists to prevent rapid updates")
    error_message_display_ms: int = Field(default=3000, description="Duration to display temporary error messages")
    status_message_display_ms: int = Field(default=2000, description="Duration to display status messages")


class GridViewConfig(BaseModel):
    """Configuration for grid view that reads from settings"""

    rect_fill_color: str = Field(default="", description="Fill color for grid rectangles - empty for transparent cells")
    rect_outline_color: str = Field(default="gray30", description="Outline color for grid rectangles")
    text_color: str = Field(default="gray80", description="Text color for grid labels")
    text_font_size_divisor: int = Field(default=25, description="Divisor for calculating font size from cell height")
    text_font_max_size: int = Field(default=48, description="Maximum font size for grid cell text")
    text_font_min_size: int = Field(default=8, description="Minimum font size for grid cell text")
    window_alpha: float = Field(default=0.65, description="Window transparency level")

    # Transparency controls for grid elements
    rect_fill_alpha: float = Field(default=0.1, description="Alpha transparency for rectangle fill - 0.0 for transparent cells")
    rect_outline_alpha: float = Field(default=0.3, description="Alpha transparency for rectangle outline (0.0-1.0)")
    text_alpha: float = Field(default=0.8, description="Alpha transparency for text labels (0.0-1.0)")

    @property
    def text_font_family(self) -> str:
        """Get font family from ui_theme"""
        return ui_theme.theme.font_family.get_primary_font("regular")

    def get_font_tuple(self, cell_height: float) -> tuple:
        """Calculate font tuple based on cell height with min/max constraints"""
        calculated_size = int(cell_height / self.text_font_size_divisor)
        # Apply min constraint (no maximum constraint as per requirements)
        font_size = max(self.text_font_min_size, calculated_size)
        # Apply max constraint to prevent overly large fonts
        font_size = min(self.text_font_max_size, font_size)
        return (self.text_font_family, font_size, "bold")

    def get_fill_color_with_alpha(self) -> str:
        """Get fill color with applied alpha transparency"""
        if self.rect_fill_alpha == 0.0 or not self.rect_fill_color:
            return ""  # No fill - transparent
        return self._apply_alpha_to_color(self.rect_fill_color, self.rect_fill_alpha)

    def get_outline_color_with_alpha(self) -> str:
        """Get outline color with applied alpha transparency"""
        return self._apply_alpha_to_color(self.rect_outline_color, self.rect_outline_alpha)

    def get_text_color_with_alpha(self) -> str:
        """Get text color with applied alpha transparency"""
        return self._apply_alpha_to_color(self.text_color, self.text_alpha)

    def _apply_alpha_to_color(self, color_name: str, alpha: float) -> str:
        """Convert a color name to hex with alpha applied"""
        # Color name to RGB mapping for common Tkinter colors
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "gray": (128, 128, 128),
            "gray10": (26, 26, 26),
            "gray20": (51, 51, 51),
            "gray30": (77, 77, 77),
            "gray40": (102, 102, 102),
            "gray50": (128, 128, 128),
            "gray60": (153, 153, 153),
            "gray70": (179, 179, 179),
            "gray80": (204, 204, 204),
            "gray90": (230, 230, 230),
        }

        # Handle hex colors
        if color_name.startswith("#"):
            try:
                # Remove # and convert to RGB
                hex_color = color_name[1:]
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    rgb = (r, g, b)
                else:
                    rgb = color_map.get("white", (255, 255, 255))
            except ValueError:
                rgb = color_map.get("white", (255, 255, 255))
        else:
            # Use color map
            rgb = color_map.get(color_name.lower(), color_map.get("white", (255, 255, 255)))

        # Apply alpha by blending with black background
        # This simulates transparency over a dark background
        r, g, b = rgb
        final_r = int(r * alpha)
        final_g = int(g * alpha)
        final_b = int(b * alpha)

        return f"#{final_r:02x}{final_g:02x}{final_b:02x}"


class MarkViewConfig(BaseModel):
    """Configuration for mark visualization"""

    mark_radius: int = Field(default=5, description="Radius of mark indicators")
    mark_fill_color: str = Field(default="red", description="Fill color for mark indicators")
    mark_outline_color: str = Field(default="white", description="Outline color for mark indicators")
    label_offset_x: int = Field(default=10, description="X offset for mark labels")
    label_offset_y: int = Field(default=-10, description="Y offset for mark labels")

    @property
    def mark_font(self) -> tuple:
        """Get mark font from ui_theme"""
        return ui_theme.theme.font_family.get_button_font(10)

    @property
    def themed_mark_fill_color(self) -> str:
        """Get mark fill color from ui_theme shape_colors.dark"""
        return ui_theme.theme.shape_colors.dark

    @property
    def themed_mark_outline_color(self) -> str:
        """Get mark outline color from ui_theme shape_colors.medium"""
        return ui_theme.theme.shape_colors.medium


class DictationPopupConfig(BaseModel):
    """Configuration for dictation popup dimensions"""

    window_margin_x: int = Field(default=20, description="Horizontal margin for popup positioning")
    window_margin_y_bottom: int = Field(default=80, description="Bottom margin for popup positioning")

    @property
    def simple_window_size(self) -> tuple:
        """Get simple window dimensions from ui_theme"""
        return (ui_theme.theme.dimensions.dictation_simple_width, ui_theme.theme.dimensions.dictation_simple_height)

    @property
    def smart_window_size(self) -> tuple:
        """Get smart window dimensions from ui_theme"""
        return (ui_theme.theme.dimensions.dictation_smart_width, ui_theme.theme.dimensions.dictation_smart_height)

    @property
    def font_family(self) -> str:
        """Get font family from ui_theme"""
        return ui_theme.theme.font_family.get_primary_font("regular")


class FormDefaults(BaseModel):
    """Default values for form fields that may be overridden by settings"""

    placeholder_samples: str = Field(default="5", description="Default placeholder for sample count")
    example_prefix: str = Field(default="e.g. ", description="Prefix for example text")

    # These will be read from settings_service at runtime
    @staticmethod
    async def get_dynamic_defaults(settings_service) -> Dict[str, Any]:
        """Get dynamic default values from settings service"""
        try:
            return {
                "grid_rects": await settings_service.get_setting("grid.default_rect_count", "500"),
                "context_length": await settings_service.get_setting("llm.context_length", "4096"),
                "max_tokens": await settings_service.get_setting("llm.max_tokens", "1024"),
            }
        except Exception:
            # Fallback to static defaults if settings service unavailable
            return {
                "grid_rects": "500",
                "context_length": "4096",
                "max_tokens": "1024",
            }


class ViewMessages(BaseModel):
    """Standard messages used across views"""

    validation_error_title: str = Field(default="Validation Error", description="Title for validation error dialogs")
    save_success_title: str = Field(default="Success", description="Title for save success dialogs")
    save_error_title: str = Field(default="Save Error", description="Title for save error dialogs")
    confirm_delete_title: str = Field(default="Confirm Delete", description="Title for delete confirmation dialogs")

    # Common message templates
    delete_confirmation_template: str = Field(
        default="Are you sure you want to delete {item}? This cannot be undone.",
        description="Template for delete confirmation messages",
    )
    delete_all_confirmation_template: str = Field(
        default="Are you sure you want to delete all {items}? This cannot be undone.",
        description="Template for delete all confirmation messages",
    )


class CommandsViewConfig(BaseModel):
    """Configuration for the commands view"""

    # Removed column width configurations since we simplified the layout


class ViewConfiguration(BaseModel):
    """Main view configuration that combines all view settings"""

    timings: ViewTimings = Field(default_factory=ViewTimings)
    grid: GridViewConfig = Field(default_factory=GridViewConfig)
    marks: MarkViewConfig = Field(default_factory=MarkViewConfig)
    dictation_popup: DictationPopupConfig = Field(default_factory=DictationPopupConfig)
    form_defaults: FormDefaults = Field(default_factory=FormDefaults)
    messages: ViewMessages = Field(default_factory=ViewMessages)
    commands: CommandsViewConfig = Field(default_factory=CommandsViewConfig)

    @property
    def theme(self):
        """Access to ui_theme for convenience"""
        return ui_theme.theme


# Singleton instance
view_config = ViewConfiguration()
