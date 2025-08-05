"""
Themed UI Components for Iris Control Room
Pre-configured CustomTkinter widgets with design attributes
"""

import customtkinter as ctk
from typing import Optional, Callable
from iris.ui.ui_theme import theme


class ThemedButton(ctk.CTkButton):
    """Base themed button with pre-configured design attributes and compact logic"""
    
    def __init__(
        self,
        parent,
        text: str = "",
        command: Optional[Callable] = None,
        size: int = None,  # Direct font size value from theme
        compact: bool = False,  # If True, button wraps around text with padding
        padding_x: int = 0,  # Horizontal padding when compact=True
        padding_y: int = 0,   # Vertical padding when compact=True
        **kwargs
    ):
        # Use theme default if not provided
        if size is None:
            size = theme.font_sizes.medium
        
        # Get standardized button font configuration
        font_tuple = theme.font_family.get_button_font(size)
        
        # Get default colors using shape colors directly
        fg_color = theme.shape_colors.medium  # Use shape_colors.medium for normal buttons
        text_color = theme.text_colors.light
        hover_color = theme.shape_colors.lightest
        
        # Set default attributes
        default_kwargs = {
            "font": font_tuple,
            "fg_color": fg_color,
            "text_color": text_color,
            "hover_color": hover_color,
            "corner_radius": theme.border_radius.small,
            "border_width": 0,
        }
        
        # Handle compact sizing
        if compact:
            # Calculate text dimensions for compact sizing
            import tkinter.font as tkFont
            from math import floor
            temp_font = tkFont.Font(family=font_tuple[0], size=font_tuple[1], weight=font_tuple[2])
            text_width = temp_font.measure(text)
            text_height = temp_font.metrics("linespace")
            
            default_kwargs["width"] = floor(text_width/2) + padding_x
            default_kwargs["height"] = floor(text_height/2) + padding_y
        else:
            default_kwargs["height"] = theme.dimensions.button_height
        
        # Override with any custom kwargs
        default_kwargs.update(kwargs)
        
        super().__init__(parent, text=text, command=command, **default_kwargs)
        
        # Initialize _font attribute to prevent destruction errors
        self._font = default_kwargs["font"]


class PrimaryButton(ThemedButton):
    """Primary themed button"""
    
    def __init__(self, parent, text: str = "", command: Optional[Callable] = None, **kwargs):
        # Set primary colors before calling parent
        # kwargs.setdefault("fg_color", theme.accent_colors.primary)
        # kwargs.setdefault("text_color", theme.text_colors.lightest)
        # kwargs.setdefault("hover_color", theme.accent_colors.primary_hover)
        super().__init__(parent, text=text, command=command, **kwargs)


class DangerButton(ThemedButton):
    """Danger themed button"""
    
    def __init__(self, parent, text: str = "", command: Optional[Callable] = None, **kwargs):
        # Set danger colors using shape colors directly
        kwargs.setdefault("fg_color", theme.shape_colors.light)  # Use shape_colors.light for danger buttons
        kwargs.setdefault("text_color", theme.text_colors.dark)
        kwargs.setdefault("hover_color", theme.shape_colors.darkest)
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
        **kwargs
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
    
    def __init__(
        self,
        parent,
        placeholder_text: str = "",
        **kwargs
    ):
        # Set default attributes with new border styling
        font_family = theme.font_family.get_primary_font("regular")
        default_kwargs = {
            "font": (font_family, theme.font_sizes.medium),
            "fg_color": theme.entry_field_styling.background_color,  # Use new entry background
            "text_color": theme.text_colors.light,
            "placeholder_text_color": theme.text_colors.light,
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
            "border_width": 0.5,
            "border_color": theme.shape_colors.darkest,
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


class BorderlessFrame(ctk.CTkFrame):
    """Borderless transparent frame for seamless content"""
    
    def __init__(self, parent, **kwargs):
        default_kwargs = {
            "fg_color": "transparent",
            "corner_radius": theme.border_radius.medium,
            "border_width": 0,
        }
        default_kwargs.update(kwargs)
        super().__init__(parent, **default_kwargs)


class TileFrame(ctk.CTkFrame):
    """Base themed tile frame for instruction content"""
    
    def __init__(self, parent, **kwargs):
        # Set default attributes using theme configuration
        default_kwargs = {
            "fg_color": theme.tile_layout.background_color,  # Use theme background color
            "corner_radius": theme.tile_layout.corner_radius,
            "border_width": 0,
        }
        
        # Override with any custom kwargs
        default_kwargs.update(kwargs)
        
        super().__init__(parent, **default_kwargs)


class ThemedScrollableFrame(ctk.CTkScrollableFrame):
    """Themed scrollable frame with pre-configured design attributes - borderless for clean lists"""
    
    def __init__(
        self,
        parent,
        **kwargs
    ):
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
    
    def __init__(
        self,
        parent,
        **kwargs
    ):
        # Set default attributes
        font_family = theme.font_family.get_primary_font("regular")
        default_kwargs = {
            "font": (font_family, theme.font_sizes.medium),
            "fg_color": theme.shape_colors.medium,
            "text_color": theme.text_colors.medium,
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
        super().__init__(parent, text=text, size=theme.font_sizes.xlarge, bold=True, color=theme.text_colors.light, **kwargs)


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
        # Get the actual font size value from the theme's tile layout configuration
        content_size = getattr(theme.font_sizes, theme.tile_layout.content_font_size)
        super().__init__(parent, text=text, size=content_size, color=theme.text_colors.dark, **kwargs)


# Composite layout components

class TwoColumnTabLayout(TransparentFrame):
    """Pre-configured two-column layout for tabs with titles and boxes - ensures consistent dimensions"""
    
    def __init__(self, parent, left_title: str, right_title: str):
        super().__init__(parent)
        
        # Configure main grid for full expansion with consistent weights and minimum sizes
        self.grid_rowconfigure(0, weight=0)  # Title row - fixed height
        self.grid_rowconfigure(1, weight=1)  # Content row - expandable, fills remaining space
        self.grid_columnconfigure(0, weight=1, minsize=300)  # Left column - exactly 50% width, minimum 300px
        self.grid_columnconfigure(1, weight=1, minsize=300)  # Right column - exactly 50% width, minimum 300px
        
        # Create title labels - align with their respective box content
        left_title_label = BoxTitle(self, text=left_title)
        left_title_label.grid(
            row=0, column=0, sticky="w",
            padx=(theme.two_box_layout.outer_padding_left, 0),  # Align with left box content
            pady=(theme.two_box_layout.title_padding_y, theme.spacing.tiny)
        )
        
        right_title_label = BoxTitle(self, text=right_title)
        right_title_label.grid(
            row=0, column=1, sticky="w",
            padx=(theme.two_box_layout.inner_spacing // 2, 0),  # Align with right box content
            pady=(theme.two_box_layout.title_padding_y, theme.spacing.tiny)
        )
        
        # Create content boxes using shape colors directly - using BorderlessFrame for clean appearance
        self.left_box = BorderlessFrame(
            self,
            fg_color=theme.shape_colors.dark,  # Use shape_colors.dark for content boxes
            corner_radius=theme.two_box_layout.box_corner_radius
        )
        self.left_box.grid(
            row=1, column=0, sticky="nsew",
            padx=(theme.two_box_layout.outer_padding_left, theme.two_box_layout.inner_spacing // 2),
            pady=(0, theme.two_box_layout.outer_padding_bottom)
        )
        
        self.right_box = BorderlessFrame(
            self,
            fg_color=theme.shape_colors.dark,  # Use shape_colors.dark for content boxes
            corner_radius=theme.two_box_layout.box_corner_radius
        )
        self.right_box.grid(
            row=1, column=1, sticky="nsew",
            padx=(theme.two_box_layout.inner_spacing // 2, theme.two_box_layout.outer_padding_right),
            pady=(0, theme.two_box_layout.outer_padding_bottom)
        )
        
        # Force consistent internal grid configuration for all boxes
        # This ensures all content expands properly and consistently
        self.left_box.grid_rowconfigure(0, weight=1)    # Content expands vertically
        self.left_box.grid_columnconfigure(0, weight=1) # Content expands horizontally
        self.right_box.grid_rowconfigure(0, weight=1)   # Content expands vertically
        self.right_box.grid_columnconfigure(0, weight=1)# Content expands horizontally
        
        # Ensure boxes expand to fill available space consistently
        # This prevents content from dictating the box size
        self.left_box.grid_propagate(False)  # Allow expansion
        self.right_box.grid_propagate(False) # Allow expansion
        
        # Override any child grid configuration that might break consistency
        # This ensures no child content can override the consistent column sizing
        self.configure_consistent_child_grids()

    def configure_consistent_child_grids(self):
        """Ensure all child containers maintain consistent grid configuration"""
        # Monitor and enforce grid consistency on scrollable frames
        # This prevents complex table layouts from affecting box sizing
        
        def after_widget_added():
            """Called after any widget is added to enforce consistency"""
            # Recursively ensure all scrollable frames use single-column layout
            for box in [self.left_box, self.right_box]:
                for child in box.winfo_children():
                    if hasattr(child, 'grid_columnconfigure'):
                        # For scrollable frames, enforce single-column layout
                        if isinstance(child, ctk.CTkScrollableFrame):
                            # Override any multi-column configuration
                            child.grid_columnconfigure(0, weight=1)
                            # Clear any other column configurations that might interfere
                            try:
                                child.grid_columnconfigure(1, weight=0)
                                child.grid_columnconfigure(2, weight=0)
                                child.grid_columnconfigure(3, weight=0)
                                child.grid_columnconfigure(4, weight=0)
                            except:
                                pass  # Columns may not exist
        
        # Schedule the check to run after initial setup
        self.after(100, after_widget_added)


class InstructionTile(TileFrame):
    """Pre-configured tile for instruction content with vertical centering"""
    
    def __init__(self, parent, title: str, content: str, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Configure grid for vertical centering
        self.grid_rowconfigure(0, weight=1)  # Top spacer
        self.grid_rowconfigure(1, weight=0)  # Title
        self.grid_rowconfigure(2, weight=0)  # Content
        self.grid_rowconfigure(3, weight=1)  # Bottom spacer
        self.grid_columnconfigure(0, weight=1)
        
        # Add title - centered vertically
        title_label = TileTitle(self, text=title)
        title_label.grid(row=1, column=0, sticky="ew", padx=theme.spacing.tiny, pady=(0, theme.spacing.tiny))
        
        # Add content - centered vertically with center text alignment
        content_label = TileContent(self, text=content)
        content_label.grid(row=2, column=0, sticky="ew", padx=theme.spacing.tiny, pady=(0, theme.spacing.tiny))


class BorderlessListItemFrame(BorderlessFrame):
    """Pre-configured borderless frame for list items with text and action button"""
    
    def __init__(self, parent, item_text: str, button_text: str, button_command: Callable, 
                 button_variant: str = "danger", **kwargs):
        super().__init__(parent, **kwargs)
        
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)  # Text column
        self.grid_columnconfigure(1, weight=0)  # Button column
        
        # Add item text - use standardized vertical spacing
        text_label = ThemedLabel(self, text=item_text, color=theme.text_colors.light)
        text_label.grid(row=0, column=0, sticky="w", padx=(theme.spacing.medium, theme.spacing.small), pady=0)
        
        # Add action button - use standardized vertical spacing
        if button_variant == "primary":
            action_button = PrimaryButton(self, text=button_text, command=button_command, compact=True)
        else:  # danger or default
            action_button = DangerButton(self, text=button_text, command=button_command, compact=True)
        action_button.grid(row=0, column=1, sticky="e", padx=(theme.spacing.small, theme.spacing.medium), pady=0)


class CustomSidebarFrame(ctk.CTkFrame):
    """Custom sidebar frame with right-only border"""
    
    def __init__(self, parent, **kwargs):
        # Remove border from kwargs if present since we'll handle it custom
        border_kwargs = {}
        if 'border_width' in kwargs:
            border_kwargs['border_width'] = kwargs.pop('border_width')
        if 'border_color' in kwargs:
            border_kwargs['border_color'] = kwargs.pop('border_color')
        
        # Initialize frame without border
        super().__init__(parent, border_width=0, **kwargs)
        
        # Create the border frame - a thin frame positioned on the right edge
        if border_kwargs.get('border_width', 0) > 0:
            self.border_frame = ctk.CTkFrame(
                self,
                width=border_kwargs.get('border_width', 1),  # Updated default from 3 to 1
                fg_color=border_kwargs.get('border_color', theme.sidebar_layout.border_color),
                corner_radius=0
            )
            # Position the border frame on the right edge
            self.border_frame.place(relx=1.0, rely=0, relheight=1.0, anchor='ne')
            # Ensure it stays on top by lifting it after any child widgets are added
            self.after_idle(lambda: self.border_frame.lift())


class SidebarIconButton(ctk.CTkFrame):
    """Sidebar button with icon above text, selection state, and border hover effect"""
    
    def __init__(
        self,
        parent,
        text: str = "",
        icon_filename: str = "",
        command: Optional[Callable] = None,
        **kwargs
    ):
        from iris.ui.utils.ui_icon_utils import transform_monochrome_icon
        from iris.ui.ui_theme import theme
        from pathlib import Path
        import os
        
        # Initialize frame with transparent background
        super().__init__(
            parent,
            fg_color="transparent",
            corner_radius=0,
            **kwargs
        )
        
        # State management
        self.is_selected = False
        self.command = command
        self.icon_filename = icon_filename
        
        # Calculate icon size
        icon_size = int(theme.icon_properties.base_size * theme.icon_properties.size_multiplier)
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Create button frame with minimum size and spacing
        # Calculate height: icon (28px) + top padding (5px) + icon_text_spacing + text height (~15px) + bottom padding (5px)
        icon_height = int(theme.icon_properties.base_size * theme.icon_properties.size_multiplier)
        text_height = theme.font_sizes.small + 2  # Add small buffer for text
        button_height = icon_height + theme.spacing.small + theme.icon_properties.icon_text_spacing + text_height + theme.spacing.small
        
        self.button_frame = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=theme.border_radius.small,
            width=theme.sidebar_layout.button_width,
            height=button_height,
            border_width=0  # Start with no border
        )
        self.button_frame.grid(row=0, column=0, sticky="ew", 
                              padx=(theme.sidebar_layout.button_padding_left, theme.sidebar_layout.button_padding_right),
                              pady=theme.sidebar_layout.button_spacing_vertical)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_propagate(False)
        
        # Load and transform icon
        self.icon_image = None
        self._load_icon(icon_size)
        
        # Create icon label
        if self.icon_image:
            self.icon_label = ctk.CTkLabel(
                self.button_frame,
                text="",
                image=self.icon_image,
                fg_color="transparent"
            )
            self.icon_label.grid(row=0, column=0, 
                               pady=(theme.spacing.small, theme.icon_properties.icon_text_spacing))
        
        # Create text label
        button_font = theme.font_family.get_button_font(theme.font_sizes.small)
        self.text_label = ctk.CTkLabel(
            self.button_frame,
            text=text,
            font=button_font,
            text_color=theme.icon_properties.color,
            fg_color="transparent"
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
        """Load and transform the icon"""
        if not self.icon_filename:
            return
            
        try:
            from iris.ui.utils.ui_icon_utils import transform_monochrome_icon
            from iris.ui.ui_theme import theme
            from pathlib import Path
            import os
            
            # Get icon path
            project_root = Path(os.getcwd())
            icon_path = project_root / "assets" / "icons" / self.icon_filename
            
            if icon_path.exists():
                # Transform the icon with theme color
                pil_image = transform_monochrome_icon(
                    str(icon_path),
                    theme.icon_properties.color,
                    (icon_size, icon_size)
                )
                
                if pil_image:
                    from PIL import ImageTk
                    self.icon_image = ImageTk.PhotoImage(pil_image)
        except Exception as e:
            import logging
            logging.error(f"Failed to load sidebar icon {self.icon_filename}: {e}")
    
    def _setup_interaction(self):
        """Set up hover and click interactions"""
        widgets_to_bind = [self, self.button_frame, self.text_label]
        if hasattr(self, 'icon_label'):
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
        """Handle mouse enter event - show border instead of background"""
        if not self.is_selected:
            # Use border instead of background color
            self.button_frame.configure(
                border_width=theme.sidebar_layout.button_hover_border_width,  # Now 1px
                border_color=theme.sidebar_layout.button_hover_border_color
            )
            self.text_label.configure(text_color=self.hover_color)
            if hasattr(self, 'icon_label'):
                self._update_icon_color(self.hover_color)
    
    def _on_leave(self, event):
        """Handle mouse leave event"""
        if not self.is_selected:
            self._reset_to_normal_state()
    
    def _reset_to_normal_state(self):
        """Reset button to normal (unselected) state"""
        self.button_frame.configure(border_width=0)  # Remove border
        self.text_label.configure(text_color=self.normal_color)
        if hasattr(self, 'icon_label'):
            self._update_icon_color(self.normal_color)
    
    def set_selected(self, selected: bool):
        """Set the selection state of the button"""
        self.is_selected = selected
        if selected:
            # Use border for selection as well
            self.button_frame.configure(
                border_width=theme.sidebar_layout.button_hover_border_width,  # Now 1px
                border_color=theme.sidebar_layout.button_hover_border_color
            )
            self.text_label.configure(text_color=self.selected_color)
            if hasattr(self, 'icon_label'):
                self._update_icon_color(self.selected_color)
        else:
            self._reset_to_normal_state()
    
    def _update_icon_color(self, color: str):
        """Update the icon color"""
        if not self.icon_filename:
            return
            
        try:
            from iris.ui.utils.ui_icon_utils import transform_monochrome_icon
            from iris.ui.ui_theme import theme
            from pathlib import Path
            import os
            
            # Get icon path
            project_root = Path(os.getcwd())
            icon_path = project_root / "assets" / "icons" / self.icon_filename
            
            if icon_path.exists():
                icon_size = int(theme.icon_properties.base_size * theme.icon_properties.size_multiplier)
                pil_image = transform_monochrome_icon(str(icon_path), color, (icon_size, icon_size))
                
                if pil_image:
                    from PIL import ImageTk
                    new_image = ImageTk.PhotoImage(pil_image)
                    self.icon_label.configure(image=new_image)
                    self.icon_image = new_image  # Keep reference
        except Exception as e:
            import logging
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