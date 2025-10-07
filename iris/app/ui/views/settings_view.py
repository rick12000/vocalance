import customtkinter as ctk
import tkinter as tk
from iris.app.ui.views.components import themed_dialogs as messagebox
from typing import Optional, Dict, Any
import logging

from iris.app.ui import ui_theme
from iris.app.ui.controls.settings_control import SettingsController
from iris.app.ui.views.components.themed_components import (
    ThemedButton, ThemedLabel, ThemedEntry, ThemedFrame, ThemedScrollableFrame,
    TileFrame, TransparentFrame, BorderlessFrame, BoxTitle, TileTitle, PrimaryButton, DangerButton
)


class SettingsView(ctk.CTkFrame):
    """UI view for settings tab - handles LLM settings configuration"""
    
    def __init__(self, parent_frame, controller: SettingsController, root_window):
        super().__init__(parent_frame, fg_color=ui_theme.theme.shape_colors.darkest)
        
        self.controller = controller
        self.parent_frame = parent_frame
        self.root_window = root_window
        self._is_alive = True
        
        self.llm_model_size_var = ctk.StringVar()
        self.llm_context_length_var = ctk.StringVar()
        self.llm_max_tokens_var = ctk.StringVar()
        self.llm_threads_var = ctk.StringVar()
        self.grid_default_cells_var = ctk.StringVar()

        self._build_tab_ui()
        self._load_current_settings()
        
        self.controller.set_view_callback(self)

    def _build_tab_ui(self):
        """Build the settings tab UI with LLM settings only"""
        # Configure main frame grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create scrollable frame for content
        scrollable_frame = ThemedScrollableFrame(self)
        scrollable_frame.grid(row=0, column=0, sticky="nsew", 
                             padx=(ui_theme.theme.header_layout.frame_padding_left, ui_theme.theme.header_layout.frame_padding_right), 
                             pady=ui_theme.theme.spacing.small)
        
        # Configure scrollable frame grid
        scrollable_frame.grid_rowconfigure(0, weight=0)  # LLM frame
        scrollable_frame.grid_rowconfigure(1, weight=0)  # Grid frame
        scrollable_frame.grid_columnconfigure(0, weight=1)
        
        # LLM Settings Section
        llm_frame = BorderlessFrame(
            scrollable_frame,
            fg_color=ui_theme.theme.shape_colors.dark,
            corner_radius=ui_theme.theme.two_box_layout.box_corner_radius
        )
        llm_frame.grid(row=0, column=0, sticky="ew", 
                      padx=(ui_theme.theme.two_box_layout.outer_padding_left, ui_theme.theme.two_box_layout.outer_padding_right), 
                      pady=(ui_theme.theme.spacing.medium, ui_theme.theme.spacing.small))
        
        # LLM Settings Header
        llm_header = BoxTitle(llm_frame, text="LLM Model Settings")
        llm_header.grid(row=0, column=0, columnspan=3, 
                       padx=ui_theme.theme.spacing.medium, 
                       pady=(ui_theme.theme.spacing.medium, ui_theme.theme.spacing.small), 
                       sticky="w")
        
        # Model Size Selection
        ThemedLabel(llm_frame, text="Model Size:", bold=True).grid(
            row=1, column=0, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        model_sizes = ["XS", "S"]
        model_descriptions = {
            "XS": "Qwen2.5-1.5B Q5 (Fastest, Basic)",
            "S": "Qwen2.5-1.5B Q8 (Better Quality, Slower)"
        }
        
        font_family = ui_theme.theme.font_family.get_primary_font("regular")
        self.llm_model_dropdown = ctk.CTkOptionMenu(
            llm_frame, 
            values=model_sizes,
            variable=self.llm_model_size_var,
            font=(font_family, ui_theme.theme.font_sizes.medium),
            fg_color=ui_theme.theme.shape_colors.darkest,
            button_color=ui_theme.theme.shape_colors.darkest,
            button_hover_color=ui_theme.theme.shape_colors.dark
        )
        self.llm_model_dropdown.grid(row=1, column=1, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="ew")
        
        self.model_desc_label = ThemedLabel(llm_frame, text="", color=ui_theme.theme.text_colors.medium)
        self.model_desc_label.grid(row=1, column=2, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        def update_model_description(*args):
            if not self._is_alive:
                return
            try:
                size = self.llm_model_size_var.get()
                desc = model_descriptions.get(size, "")
                self.model_desc_label.configure(text=desc)
            except tk.TclError:
                pass
        
        self.llm_model_size_var.trace_add("write", update_model_description)
        
        # Context Length
        ThemedLabel(llm_frame, text="Context Length:", bold=True).grid(
            row=2, column=0, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        self.llm_context_entry = ThemedEntry(
            llm_frame, 
            textvariable=self.llm_context_length_var,
            width=ui_theme.theme.dimensions.entry_width_small
        )
        self.llm_context_entry.grid(row=2, column=1, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        # Context length description
        context_desc = ThemedLabel(llm_frame, text="(128-32768, higher = more context)", color=ui_theme.theme.text_colors.medium)
        context_desc.grid(row=2, column=2, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        # Max Tokens
        ThemedLabel(llm_frame, text="Max Tokens:", bold=True).grid(
            row=3, column=0, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        self.llm_max_tokens_entry = ThemedEntry(
            llm_frame, 
            textvariable=self.llm_max_tokens_var,
            width=ui_theme.theme.dimensions.entry_width_small
        )
        self.llm_max_tokens_entry.grid(row=3, column=1, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        # Max tokens description
        tokens_desc = ThemedLabel(llm_frame, text="(1-1024, higher = longer responses)", color=ui_theme.theme.text_colors.medium)
        tokens_desc.grid(row=3, column=2, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        # Threads
        ThemedLabel(llm_frame, text="Processing Threads:", bold=True).grid(
            row=4, column=0, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        self.llm_threads_entry = ThemedEntry(
            llm_frame, 
            textvariable=self.llm_threads_var,
            width=ui_theme.theme.dimensions.entry_width_small
        )
        self.llm_threads_entry.grid(row=4, column=1, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        # Threads description
        threads_desc = ThemedLabel(llm_frame, text="(1-32, match your CPU cores)", color=ui_theme.theme.text_colors.medium)
        threads_desc.grid(row=4, column=2, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        # Configure column weights for LLM frame
        llm_frame.grid_columnconfigure(1, weight=1)
        llm_frame.grid_columnconfigure(2, weight=2)

        # Buttons frame for LLM settings
        buttons_frame = TransparentFrame(llm_frame)
        buttons_frame.grid(row=5, column=0, columnspan=3, pady=20, padx=20, sticky="ew")
        
        # Configure buttons_frame grid
        buttons_frame.grid_columnconfigure(0, weight=0)
        buttons_frame.grid_columnconfigure(1, weight=0)
        
        # Save button
        save_button = PrimaryButton(
            buttons_frame,
            text="Save LLM Settings",
            command=self._save_llm_settings
        )
        save_button.grid(row=0, column=0, padx=(0, ui_theme.theme.spacing.small), sticky="w")
        
        # Reset to defaults button
        reset_button = DangerButton(
            buttons_frame,
            text="Reset to Defaults",
            command=self._reset_llm_to_defaults
        )
        reset_button.grid(row=0, column=1, sticky="w")
        
        # Grid Settings Section
        grid_frame = BorderlessFrame(
            scrollable_frame,
            fg_color=ui_theme.theme.shape_colors.dark,
            corner_radius=ui_theme.theme.two_box_layout.box_corner_radius
        )
        grid_frame.grid(row=1, column=0, sticky="ew", 
                       padx=(ui_theme.theme.two_box_layout.outer_padding_left, ui_theme.theme.two_box_layout.outer_padding_right), 
                       pady=(ui_theme.theme.spacing.small, ui_theme.theme.spacing.medium))
        
        # Grid Settings Header
        grid_header = BoxTitle(grid_frame, text="Grid Settings")
        grid_header.grid(row=0, column=0, columnspan=3, 
                        padx=ui_theme.theme.spacing.medium, 
                        pady=(ui_theme.theme.spacing.medium, ui_theme.theme.spacing.small), 
                        sticky="w")
        
        # Default Cell Count
        ThemedLabel(grid_frame, text="Default Cell Count:", bold=True).grid(
            row=1, column=0, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        self.grid_default_cells_entry = ThemedEntry(
            grid_frame, 
            textvariable=self.grid_default_cells_var,
            width=ui_theme.theme.dimensions.entry_width_small
        )
        self.grid_default_cells_entry.grid(row=1, column=1, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        # Default cell count description
        cells_desc = ThemedLabel(grid_frame, text="(100-10000, cells shown when saying 'golf')", color=ui_theme.theme.text_colors.medium)
        cells_desc.grid(row=1, column=2, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
        
        # Configure column weights for Grid frame
        grid_frame.grid_columnconfigure(1, weight=1)
        grid_frame.grid_columnconfigure(2, weight=2)

        # Buttons frame for Grid settings
        grid_buttons_frame = TransparentFrame(grid_frame)
        grid_buttons_frame.grid(row=2, column=0, columnspan=3, pady=20, padx=20, sticky="ew")
        
        # Configure grid_buttons_frame grid
        grid_buttons_frame.grid_columnconfigure(0, weight=0)
        grid_buttons_frame.grid_columnconfigure(1, weight=0)
        
        # Save button for grid settings
        grid_save_button = PrimaryButton(
            grid_buttons_frame,
            text="Save Grid Settings",
            command=self._save_grid_settings
        )
        grid_save_button.grid(row=0, column=0, padx=(0, ui_theme.theme.spacing.small), sticky="w")
        
        # Reset to defaults button for grid settings
        grid_reset_button = DangerButton(
            grid_buttons_frame,
            text="Reset to Defaults",
            command=self._reset_grid_to_defaults
        )
        grid_reset_button.grid(row=0, column=1, sticky="w")

    def on_settings_updated(self):
        if not self._is_alive:
            return
        self._load_current_settings()

    def on_validation_error(self, title: str, message: str):
        """Handle validation errors from controller"""
        messagebox.showerror(title, message, parent=self.root_window)

    def on_save_success(self, message: str):
        """Handle successful save from controller"""
        messagebox.showinfo("Success", message, parent=self.root_window)

    def on_save_error(self, message: str):
        """Handle save errors from controller"""
        messagebox.showerror("Error", message, parent=self.root_window)

    def on_reset_complete(self):
        """Handle reset completion from controller"""
        # Reload settings from controller to get updated values
        self._load_current_settings()
        
        messagebox.showinfo("Reset Complete", "Settings have been reset to defaults", parent=self.root_window)

    def _load_current_settings(self):
        """Load current settings from controller"""
        try:
            settings = self.controller.load_current_settings()
            
            if settings:
                # LLM settings
                llm_settings = settings.get('llm', {})
                self.llm_model_size_var.set(llm_settings.get('model_size', 'S'))
                self.llm_context_length_var.set(str(llm_settings.get('context_length', 2048)))
                self.llm_max_tokens_var.set(str(llm_settings.get('max_tokens', 512)))
                self.llm_threads_var.set(str(llm_settings.get('n_threads', 4)))
                
                # Grid settings
                grid_settings = settings.get('grid', {})
                self.grid_default_cells_var.set(str(grid_settings.get('default_rect_count', 500)))
            else:
                # Set error values
                for var in [self.llm_model_size_var, self.llm_context_length_var, self.llm_max_tokens_var, self.llm_threads_var, self.grid_default_cells_var]:
                    if isinstance(var, ctk.StringVar):
                        var.set("Error")
                        
        except Exception as e:
            self.controller.logger.error(f"Error loading settings into UI: {e}")

    def _save_llm_settings(self):
        """Save LLM settings through controller"""
        self.controller.save_llm_settings(
            self.llm_model_size_var.get(),
            self.llm_context_length_var.get(),
            self.llm_max_tokens_var.get(),
            self.llm_threads_var.get()
        )
    
    def _reset_llm_to_defaults(self):
        """Reset LLM settings to defaults through controller"""
        result = messagebox.askyesno(
            "Reset LLM Settings", 
            "Are you sure you want to reset LLM settings to defaults?", 
            parent=self.root_window
        )
        
        if result:
            self.controller.reset_llm_to_defaults()

    def _save_grid_settings(self):
        """Save Grid settings through controller"""
        self.controller.save_grid_settings(
            self.grid_default_cells_var.get()
        )
    
    def _reset_grid_to_defaults(self):
        """Reset Grid settings to defaults through controller"""
        result = messagebox.askyesno(
            "Reset Grid Settings", 
            "Are you sure you want to reset Grid settings to defaults?", 
            parent=self.root_window
        )
        
        if result:
            self.controller.reset_grid_to_defaults()

    def refresh_settings(self):
        if not self._is_alive:
            return
        self._load_current_settings()
    
    def destroy(self):
        self._is_alive = False
        self.controller.set_view_callback(None)
        super().destroy() 