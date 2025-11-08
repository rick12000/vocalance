import logging
import tkinter.font as tkFont
from pathlib import Path
from typing import Callable, Optional

import customtkinter as ctk

from vocalance.app.config.app_config import AssetPathsConfig
from vocalance.app.ui.ui_theme import theme
from vocalance.app.ui.utils.icon_transform_utils import transform_monochrome_icon


class ThemedButton(ctk.CTkButton):
    """Base themed button with fixed height and optional compact width mode"""

    def __init__(
        self,
        parent,
        text: str = "",
        command: Optional[Callable] = None,
        size: int = None,
        compact: bool = False,
        **kwargs,
    ):
        if size is None:
            size = theme.font_sizes.medium

        font_tuple = theme.font_family.get_button_font(size)
        font_tuple = (font_tuple[0], font_tuple[1], "bold")

        fg_color = theme.shape_colors.accent
        text_color = theme.shape_colors.dark
        hover_color = theme.shape_colors.lightest

        button_height = theme.dimensions.button_height
        corner_radius = int(button_height / 2)

        default_kwargs = {
            "font": font_tuple,
            "fg_color": fg_color,
            "text_color": text_color,
            "hover_color": hover_color,
            "corner_radius": corner_radius,
            "border_width": 0,
            "height": button_height,
        }

        if compact and "width" not in kwargs:
            temp_font = tkFont.Font(family=font_tuple[0], size=font_tuple[1], weight=font_tuple[2])
            text_width = temp_font.measure(text) if text else button_height
            default_kwargs["width"] = text_width + (theme.dimensions.button_text_padding * 2)

        default_kwargs.update(kwargs)

        super().__init__(parent, text=text, command=command, **default_kwargs)


class PrimaryButton(ThemedButton):
    """Primary themed button"""

    def __init__(self, parent, text: str = "", command: Optional[Callable] = None, **kwargs):
        # Set bold font like danger buttons
        if "font" not in kwargs:
            size = kwargs.get("size", theme.font_sizes.medium)
            font_family = theme.font_family.get_primary_font("bold")
            kwargs["font"] = (font_family, size, "bold")

        kwargs.setdefault("fg_color", theme.shape_colors.accent)
        kwargs.setdefault("hover_color", theme.shape_colors.accent_minus)

        super().__init__(parent, text=text, command=command, **kwargs)


class DangerButton(ThemedButton):
    """Danger themed button"""

    def __init__(self, parent, text: str = "", command: Optional[Callable] = None, **kwargs):
        # Set danger colors - transparent with border
        kwargs.setdefault("fg_color", theme.shape_colors.darkest)
        kwargs.setdefault("text_color", theme.text_colors.light)
        kwargs.setdefault("hover_color", theme.shape_colors.medium)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", theme.shape_colors.lightest)

        if "font" not in kwargs:
            size = kwargs.get("size", theme.font_sizes.medium)
            font_family = theme.font_family.get_primary_font("bold")
            kwargs["font"] = (font_family, size, "bold")

        super().__init__(parent, text=text, command=command, **kwargs)


class ThemedLabel(ctk.CTkLabel):
    """Themed label with pre-configured design attributes"""

    def __init__(
        self,
        parent,
        text: str = "",
        size: int = None,  # Direct font size value from theme
        color: str = None,  # Direct color value from theme
        bold: bool = False,
        **kwargs,
    ):
        # Use theme defaults if not provided
        if size is None:
            size = theme.font_sizes.medium
        if color is None:
            color = theme.text_colors.lightest

        # Get font configuration using font service
        weight = "bold" if bold else "regular"
        font_family = theme.font_family.get_primary_font(weight)
        font_tuple = (font_family, size, "normal")  # Use "normal" since we select the right font variant

        # Set default attributes
        default_kwargs = {
            "font": font_tuple,
            "text_color": color,
        }

        # Override with any custom kwargs
        default_kwargs.update(kwargs)

        super().__init__(parent, text=text, **default_kwargs)


class ThemedEntry(ctk.CTkEntry):
    """Themed entry with pre-configured design attributes and borders"""

    def __init__(self, parent, placeholder_text: str = "", **kwargs):
        # Set default attributes with new border styling
        font_family = theme.font_family.get_primary_font("regular")
        default_kwargs = {
            "font": (font_family, theme.font_sizes.medium),
            "fg_color": theme.entry_field_styling.background_color,  # Use new entry background
            "text_color": theme.text_colors.light,
            "placeholder_text_color": theme.shape_colors.light,
            "border_color": theme.entry_field_styling.border_color,  # Use new border color
            "corner_radius": theme.entry_field_styling.corner_radius,  # Use new corner radius
            "height": theme.dimensions.entry_height,
            "border_width": theme.entry_field_styling.border_width,  # Use new border width
        }

        # Override with any custom kwargs
        default_kwargs.update(kwargs)

        super().__init__(parent, placeholder_text=placeholder_text, **default_kwargs)


class ThemedFrame(ctk.CTkFrame):
    """Base themed frame with surface styling"""

    def __init__(self, parent, **kwargs):
        # Set default attributes
        default_kwargs = {
            "fg_color": theme.shape_colors.medium,
            "corner_radius": theme.border_radius.medium,
            "border_width": 1,
            "border_color": theme.shape_colors.medium,
        }

        # Override with any custom kwargs
        default_kwargs.update(kwargs)

        super().__init__(parent, **default_kwargs)


class TransparentFrame(ctk.CTkFrame):
    """Transparent frame for grouping without visual boundaries"""

    def __init__(self, parent, **kwargs):
        default_kwargs = {
            "fg_color": "transparent",
            "corner_radius": 0,
            "border_width": 0,
        }
        default_kwargs.update(kwargs)
        super().__init__(parent, **default_kwargs)


BorderlessFrame = TransparentFrame


class TileFrame(ctk.CTkFrame):
    """Base themed tile frame for instruction content"""

    def __init__(self, parent, **kwargs):
        # Set default attributes using theme configuration
        default_kwargs = {
            "fg_color": theme.tile_layout.background_color,  # Use theme background color
            "corner_radius": theme.tile_layout.corner_radius,
            "border_width": theme.tile_layout.border_width,
            "border_color": theme.tile_layout.border_color,
        }

        # Override with any custom kwargs
        default_kwargs.update(kwargs)

        super().__init__(parent, **default_kwargs)


class ThemedScrollableFrame(ctk.CTkScrollableFrame):
    """Themed scrollable frame with pre-configured design attributes - borderless for clean lists"""

    def __init__(self, parent, **kwargs):
        # Set default attributes - no borders for clean list appearance, using shape colors directly
        default_kwargs = {
            "fg_color": theme.shape_colors.dark,  # Use shape_colors.dark for content boxes
            "corner_radius": theme.two_box_layout.box_corner_radius,
            "border_width": 0,  # No border for clean appearance
            "scrollbar_button_color": theme.shape_colors.light,
            "scrollbar_button_hover_color": theme.shape_colors.darkest,
        }

        # Override with any custom kwargs
        default_kwargs.update(kwargs)

        super().__init__(parent, **default_kwargs)


class ThemedTextbox(ctk.CTkTextbox):
    """Themed textbox with pre-configured design attributes"""

    def __init__(self, parent, **kwargs):
        # Set default attributes
        font_family = theme.font_family.get_primary_font("regular")
        default_kwargs = {
            "font": (font_family, theme.font_sizes.medium),
            "fg_color": theme.shape_colors.medium,
            "text_color": theme.shape_colors.light,
            "border_color": theme.shape_colors.darkest,
            "corner_radius": theme.border_radius.medium,
            "border_width": 1,
        }

        # Override with any custom kwargs
        default_kwargs.update(kwargs)

        super().__init__(parent, **default_kwargs)


class BoxTitle(ThemedLabel):
    """Pre-configured label for tab titles"""

    def __init__(self, parent, text: str = "", **kwargs):
        super().__init__(parent, text=text, size=theme.font_sizes.xlarge, bold=True, color=theme.text_colors.lightest, **kwargs)


class TileTitle(ThemedLabel):
    """Pre-configured label for tile titles"""

    def __init__(self, parent, text: str = "", **kwargs):
        # Get the actual font size value from the theme's tile layout configuration
        title_size = getattr(theme.font_sizes, theme.tile_layout.title_font_size)
        super().__init__(parent, text=text, size=title_size, bold=True, **kwargs)


class TileContent(ThemedLabel):
    """Pre-configured label for tile content with center alignment and smaller font"""

    def __init__(self, parent, text: str = "", **kwargs):
        # Set default justify to center for text alignment
        kwargs.setdefault("justify", theme.tile_layout.content_text_alignment)
        # Enable text wrapping with center anchor for vertical and horizontal centering
        kwargs.setdefault("anchor", "center")
        # Get the actual font size value from the theme's tile layout configuration
        content_size = getattr(theme.font_sizes, theme.tile_layout.content_font_size)
        super().__init__(parent, text=text, size=content_size, color=theme.text_colors.dark, **kwargs)


# Composite layout components


class TwoColumnTabLayout(TransparentFrame):
    """Pre-configured two-column layout for tabs with titles and boxes - ensures consistent dimensions"""

    def __init__(self, parent, left_title: str, right_title: str):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=300)
        self.grid_columnconfigure(1, weight=1, minsize=300)

        half_inner_spacing = theme.two_box_layout.inner_spacing // 2

        self.left_box = ctk.CTkFrame(
            self,
            fg_color=theme.two_box_layout.box_background_color,
            corner_radius=theme.two_box_layout.box_corner_radius,
            border_width=theme.two_box_layout.box_border_width,
            border_color=theme.two_box_layout.box_border_color,
        )
        self.left_box.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(theme.two_box_layout.outer_padding_left, half_inner_spacing),
            pady=(theme.two_box_layout.outer_padding_top, theme.two_box_layout.outer_padding_bottom),
        )

        self.right_box = ctk.CTkFrame(
            self,
            fg_color=theme.two_box_layout.box_background_color,
            corner_radius=theme.two_box_layout.box_corner_radius,
            border_width=theme.two_box_layout.box_border_width,
            border_color=theme.two_box_layout.box_border_color,
        )
        self.right_box.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(half_inner_spacing, theme.two_box_layout.outer_padding_right),
            pady=(theme.two_box_layout.outer_padding_top, theme.two_box_layout.outer_padding_bottom),
        )

        self.left_box.grid_rowconfigure(0, weight=0)
        self.left_box.grid_rowconfigure(1, weight=1)
        self.left_box.grid_columnconfigure(0, weight=1)

        self.right_box.grid_rowconfigure(0, weight=0)
        self.right_box.grid_rowconfigure(1, weight=1)
        self.right_box.grid_columnconfigure(0, weight=1)

        left_title_label = BoxTitle(self.left_box, text=left_title)
        left_title_label.grid(
            row=0,
            column=0,
            sticky="w",
            padx=(theme.two_box_layout.title_padx_left, theme.two_box_layout.title_padx_right),
            pady=(theme.two_box_layout.title_padding_top, theme.two_box_layout.title_padding_bottom),
        )

        right_title_label = BoxTitle(self.right_box, text=right_title)
        right_title_label.grid(
            row=0,
            column=0,
            sticky="w",
            padx=(theme.two_box_layout.title_padx_left, theme.two_box_layout.title_padx_right),
            pady=(theme.two_box_layout.title_padding_top, theme.two_box_layout.title_padding_bottom),
        )

        self.left_content = TransparentFrame(self.left_box)
        self.left_content.grid(row=1, column=0, sticky="nsew", padx=1, pady=(0, theme.two_box_layout.last_element_bottom_padding))
        self.left_content.grid_rowconfigure(0, weight=1)
        self.left_content.grid_columnconfigure(0, weight=1)

        self.right_content = TransparentFrame(self.right_box)
        self.right_content.grid(row=1, column=0, sticky="nsew", padx=1, pady=(0, theme.two_box_layout.last_element_bottom_padding))
        self.right_content.grid_rowconfigure(0, weight=1)
        self.right_content.grid_columnconfigure(0, weight=1)

        self.left_box.grid_propagate(False)
        self.right_box.grid_propagate(False)


class InstructionTile(TileFrame):
    """Pre-configured tile for instruction content with vertical centering"""

    def __init__(self, parent, title: str, content: str, **kwargs):
        super().__init__(parent, **kwargs)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Add title - centered horizontally and vertically, with padding matching box content
        title_label = TileTitle(self, text=title)
        title_label.grid(row=0, column=0, sticky="ew", padx=theme.spacing.small, pady=(theme.spacing.small, theme.spacing.none))

        # Add content - centered both horizontally and vertically, reduced spacing from title
        content_label = TileContent(self, text=content)
        content_label.grid(
            row=1, column=0, sticky="nsew", padx=theme.spacing.small, pady=(theme.spacing.none, theme.spacing.small)
        )


class BorderlessListItemFrame(BorderlessFrame):
    """Pre-configured borderless frame for list items with text and action button"""

    def __init__(
        self, parent, item_text: str, button_text: str, button_command: Callable, button_variant: str = "danger", **kwargs
    ):
        super().__init__(parent, **kwargs)

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)  # Text column
        self.grid_columnconfigure(1, weight=0)  # Button column

        # Add item text - use standardized vertical spacing
        text_label = ThemedLabel(self, text=item_text, color=theme.text_colors.light)
        text_label.grid(row=0, column=0, sticky="w", padx=(theme.spacing.medium, theme.spacing.small), pady=0)

        # Add action button - use standardized vertical spacing with compact mode
        if button_variant == "primary":
            action_button = PrimaryButton(self, text=button_text, command=button_command, compact=True)
        else:  # danger or default
            action_button = DangerButton(self, text=button_text, command=button_command, compact=True)
        action_button.grid(row=0, column=1, sticky="e", padx=(theme.spacing.small, theme.spacing.medium), pady=0)


class CustomSidebarFrame(ctk.CTkFrame):
    """Custom sidebar frame with right-only border using Tkinter Frame for reliable rendering"""

    def __init__(self, parent, **kwargs):
        pass

        border_kwargs = {}
        if "border_width" in kwargs:
            border_kwargs["border_width"] = kwargs.pop("border_width")
        if "border_color" in kwargs:
            border_kwargs["border_color"] = kwargs.pop("border_color")

        kwargs["border_width"] = 0
        if "fg_color" in kwargs:
            kwargs["border_color"] = kwargs["fg_color"]

        super().__init__(parent, **kwargs)

        self._border_width = border_kwargs.get("border_width", 0)
        self._border_color = border_kwargs.get("border_color", theme.sidebar_layout.border_color)
        self.border_line = None

        if self._border_width > 0:
            self._create_border_line()

    def _create_border_line(self):
        """Create and configure the border line using a standard Tkinter Frame"""
        import tkinter as tk

        self.border_line = tk.Frame(
            self,
            width=self._border_width,
            bg=self._border_color,
            highlightthickness=0,
            bd=0,
        )
        self.border_line.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

        self.bind("<Map>", lambda e: self._ensure_border_on_top())

        self.after(100, self._ensure_border_on_top)

    def _ensure_border_on_top(self):
        """Ensure the border line stays on top of all widgets"""
        if self.border_line and self.border_line.winfo_exists():
            self.border_line.tkraise()

    def raise_border(self):
        """Public method to raise the border line - call after adding children"""
        self._ensure_border_on_top()


class SidebarIconButton(ctk.CTkFrame):
    """Sidebar button with icon above text, selection state, and border hover effect"""

    _icon_cache = {}

    def __init__(
        self,
        parent,
        asset_paths_config: AssetPathsConfig,
        text: str = "",
        icon_filename: str = "",
        command: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(parent, fg_color="transparent", corner_radius=0, **kwargs)

        self.is_selected = False
        self.command = command
        self.icon_filename = icon_filename
        self._asset_paths_config = asset_paths_config

        available_width = theme.sidebar_layout.button_width
        self.icon_size = int(available_width * theme.icon_properties.width_percentage)

        self.grid_columnconfigure(0, weight=1)

        text_height = theme.font_sizes.small + 2
        button_height = (
            theme.spacing.small + self.icon_size + theme.icon_properties.icon_text_spacing + text_height + theme.spacing.small
        )

        self.button_frame = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=theme.border_radius.rounded,
            width=theme.sidebar_layout.button_width,
            height=button_height,
            border_width=0,  # Start with no border
        )
        self.button_frame.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(theme.sidebar_layout.button_padding_left, theme.sidebar_layout.button_padding_right),
            pady=theme.sidebar_layout.button_spacing_vertical,
        )
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_propagate(False)

        # Load and transform icon
        self.icon_image = None
        self._load_icon(self.icon_size)

        # Create icon label
        if self.icon_image:
            self.icon_label = ctk.CTkLabel(self.button_frame, text="", image=self.icon_image, fg_color="transparent")
            self.icon_label.grid(row=0, column=0, pady=(theme.spacing.small, theme.icon_properties.icon_text_spacing))

        # Create text label
        button_font = theme.font_family.get_button_font(theme.font_sizes.small)
        self.text_label = ctk.CTkLabel(
            self.button_frame, text=text, font=button_font, text_color=theme.icon_properties.color, fg_color="transparent"
        )
        text_row = 1 if self.icon_image else 0
        self.text_label.grid(row=text_row, column=0, pady=(0, theme.spacing.small))

        # Configure interaction
        self._setup_interaction()

        # Store colors for state management
        self.normal_color = theme.icon_properties.color
        self.hover_color = theme.text_colors.lightest
        self.selected_color = theme.text_colors.lightest

    def _load_icon(self, icon_size: int):
        """Load and transform the icon with caching"""
        if not self.icon_filename:
            return

        try:
            # Create cache key
            cache_key = f"{self.icon_filename}_{icon_size}_{theme.icon_properties.color}"

            # Check cache first
            if cache_key in self._icon_cache:
                self.icon_image = self._icon_cache[cache_key]
                return

            icon_path = Path(self._asset_paths_config.icons_dir) / self.icon_filename

            if icon_path.exists():
                # Transform the icon with theme color
                # force_all_pixels=True prevents luminance-based alpha reduction for light-colored source icons
                pil_image = transform_monochrome_icon(
                    str(icon_path), theme.icon_properties.color, (icon_size, icon_size), force_all_pixels=True
                )

                if pil_image:
                    # Use CTkImage for proper HiDPI scaling
                    icon_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(icon_size, icon_size))
                    # Cache the icon
                    self._icon_cache[cache_key] = icon_image
                    self.icon_image = icon_image
            else:
                logging.warning(f"Sidebar icon not found: {icon_path}")
        except Exception as e:
            logging.error(f"Failed to load sidebar icon {self.icon_filename}: {e}")

    def _setup_interaction(self):
        """Set up hover and click interactions"""
        widgets_to_bind = [self, self.button_frame, self.text_label]
        if hasattr(self, "icon_label"):
            widgets_to_bind.append(self.icon_label)

        for widget in widgets_to_bind:
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

    def _on_click(self, event):
        """Handle click event"""
        if self.command:
            self.command()

    def _on_enter(self, event):
        """Handle mouse enter event - no border, just color change"""
        if not self.is_selected:
            # No border - just color change for hover effect
            self.button_frame.configure(border_width=0)
            self.text_label.configure(text_color=self.hover_color)
            if hasattr(self, "icon_label"):
                self._update_icon_color(self.hover_color)

    def _on_leave(self, event):
        """Handle mouse leave event"""
        if not self.is_selected:
            self._reset_to_normal_state()

    def _reset_to_normal_state(self):
        """Reset button to normal (unselected) state"""
        self.button_frame.configure(border_width=0)  # Remove border
        self.text_label.configure(text_color=self.normal_color)
        if hasattr(self, "icon_label"):
            self._update_icon_color(self.normal_color)

    def set_selected(self, selected: bool):
        """Set the selection state of the button"""
        self.is_selected = selected
        if selected:
            # No border for selection - just color change
            self.button_frame.configure(border_width=0)
            self.text_label.configure(text_color=self.selected_color)
            if hasattr(self, "icon_label"):
                self._update_icon_color(self.selected_color)
        else:
            self._reset_to_normal_state()

    def _update_icon_color(self, color: str):
        """Update the icon color with caching"""
        if not self.icon_filename:
            return

        try:
            cache_key = f"{self.icon_filename}_{self.icon_size}_{color}"

            # Check cache first
            if cache_key in self._icon_cache:
                cached_image = self._icon_cache[cache_key]
                self.icon_label.configure(image=cached_image)
                self.icon_image = cached_image
                return

            icon_path = Path(self._asset_paths_config.icons_dir) / self.icon_filename

            if icon_path.exists():
                # force_all_pixels=True prevents luminance-based alpha reduction for light-colored source icons
                pil_image = transform_monochrome_icon(
                    str(icon_path), color, (self.icon_size, self.icon_size), force_all_pixels=True
                )

                if pil_image:
                    # Use CTkImage for proper HiDPI scaling
                    new_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(self.icon_size, self.icon_size))
                    # Cache the colored icon
                    self._icon_cache[cache_key] = new_image
                    self.icon_label.configure(image=new_image)
                    self.icon_image = new_image  # Keep reference
        except Exception as e:
            logging.error(f"Failed to update icon color: {e}")


class SidebarButtonManager:
    """Manages selection state across multiple sidebar buttons"""

    def __init__(self):
        self.buttons = []
        self.selected_button = None

    def add_button(self, button: SidebarIconButton):
        """Add a button to the manager"""
        self.buttons.append(button)

        # Wrap the original command to handle selection
        original_command = button.command

        def wrapped_command():
            self.select_button(button)
            if original_command:
                original_command()

        button.command = wrapped_command

    def select_button(self, button: SidebarIconButton):
        """Select a button and deselect others"""
        # Deselect current button
        if self.selected_button:
            self.selected_button.set_selected(False)

        # Select new button
        button.set_selected(True)
        self.selected_button = button

    def get_selected_button(self) -> Optional[SidebarIconButton]:
        """Get the currently selected button"""
        return self.selected_button
