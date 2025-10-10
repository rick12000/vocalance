"""
Form builder utility to simplify and standardize widget creation across views.
"""

import customtkinter as ctk
import tkinter as tk
from typing import Optional, Callable, Dict, Any, List, Tuple
from iris.app.ui import ui_theme
from iris.app.ui.views.components.themed_components import (
    ThemedLabel, ThemedEntry, ThemedTextbox, PrimaryButton, DangerButton
)
from iris.app.ui.views.components.view_config import view_config

class FormBuilder:
    """Utility class for building common form patterns with minimal code"""
    
    @staticmethod
    def create_labeled_entry(
        parent: ctk.CTkFrame,
        label_text: str,
        placeholder: str = "",
        default_value: str = "",
        row: int = 0,
        width: Optional[int] = None
    ) -> Tuple[ThemedLabel, ThemedEntry]:
        """Create a label and entry pair with consistent styling"""
        label = ThemedLabel(parent, text=label_text, bold=True)
        label.grid(row=row, column=0, sticky="w", 
                  pady=(view_config.theme.spacing.medium, view_config.theme.spacing.tiny), 
                  padx=view_config.theme.spacing.medium)
        
        entry_kwargs = {"placeholder_text": placeholder}
        if width:
            entry_kwargs["width"] = width
            
        entry = ThemedEntry(parent, **entry_kwargs)
        if default_value:
            entry.insert(0, default_value)
        entry.grid(row=row + 1, column=0, sticky="ew", 
                  pady=(0, view_config.theme.spacing.small), 
                  padx=view_config.theme.spacing.medium)
        
        return label, entry
    
    @staticmethod
    def create_labeled_textbox(
        parent: ctk.CTkFrame,
        label_text: str,
        placeholder: str = "",
        height: Optional[int] = None,
        row: int = 0
    ) -> Tuple[ThemedLabel, ThemedTextbox]:
        """Create a label and textbox pair with consistent styling"""
        label = ThemedLabel(parent, text=label_text, bold=True)
        label.grid(row=row, column=0, sticky="w", 
                  pady=(view_config.theme.spacing.tiny, 0), 
                  padx=view_config.theme.spacing.medium)
        
        textbox_height = height or view_config.theme.dimensions.textbox_height_small
        textbox = ThemedTextbox(
            parent,
            height=textbox_height,
            fg_color=view_config.theme.shape_colors.darkest,
            border_width=view_config.theme.entry_field_styling.border_width,
            border_color=view_config.theme.entry_field_styling.border_color
        )
        textbox.grid(row=row + 1, column=0, sticky="ew", 
                    pady=(view_config.theme.spacing.tiny, view_config.theme.spacing.small), 
                    padx=view_config.theme.spacing.medium)
        
        if placeholder:
            textbox.insert("1.0", placeholder)
        
        return label, textbox
    
    @staticmethod
    def create_button_row(
        parent: ctk.CTkFrame,
        buttons: List[Dict[str, Any]],
        row: int = 0,
        extra_pady: Optional[Tuple[int, int]] = None,
        extra_padx: Optional[int] = None
    ) -> List[ctk.CTkButton]:
        """Create a row of buttons with consistent spacing"""
        pady = extra_pady if extra_pady is not None else view_config.theme.spacing.small
        padx = extra_padx if extra_padx is not None else view_config.theme.spacing.medium
        
        button_frame = ctk.CTkFrame(parent, fg_color="transparent")
        button_frame.grid(row=row, column=0, sticky="ew", 
                         pady=pady, 
                         padx=padx)
        
        created_buttons = []
        for i, btn_config in enumerate(buttons):
            btn_type = btn_config.get("type", "primary")
            ButtonClass = PrimaryButton if btn_type == "primary" else DangerButton
            
            button = ButtonClass(
                button_frame,
                text=btn_config["text"],
                command=btn_config["command"]
            )
            button.grid(row=0, column=i, sticky="ew", 
                       padx=(0 if i == 0 else view_config.theme.spacing.tiny, 
                            view_config.theme.spacing.tiny if i < len(buttons) - 1 else 0))
            
            button_frame.grid_columnconfigure(i, weight=1)
            created_buttons.append(button)
        
        return created_buttons
    
    @staticmethod
    def setup_form_grid(parent: ctk.CTkFrame, rows: int) -> None:
        """Configure grid weights for a standard form layout"""
        parent.grid_columnconfigure(0, weight=1)
        for i in range(rows):
            parent.grid_rowconfigure(i, weight=0)
    
    @staticmethod
    def show_temporary_message(
        parent: ctk.CTkFrame,
        message: str,
        row: int,
        duration_ms: Optional[int] = None,
        is_error: bool = True
    ) -> None:
        """Show a temporary message that auto-dismisses"""
        if duration_ms is None:
            duration_ms = view_config.timings.error_message_display_ms
            
        color = (view_config.theme.text_colors.darkest if is_error 
                else view_config.theme.accent_colors.success_text)
        prefix = "⚠ " if is_error else "✓ "
        
        msg_label = ThemedLabel(parent, text=f"{prefix}{message}", color=color)
        msg_label.grid(row=row, column=0, sticky="ew", 
                      padx=view_config.theme.spacing.medium, 
                      pady=view_config.theme.spacing.tiny)
        
        def remove_message():
            try:
                if msg_label.winfo_exists():
                    msg_label.destroy()
            except:
                pass
        
        parent.after(duration_ms, remove_message)
    
    @staticmethod
    def create_example_textbox_with_placeholder(
        parent: ctk.CTkFrame,
        example_text: str,
        row: int = 0
    ) -> ThemedTextbox:
        """Create a textbox with example text that disappears on click"""
        textbox = ThemedTextbox(
            parent,
            height=view_config.theme.dimensions.textbox_height_small,
            fg_color=view_config.theme.shape_colors.dark,
            border_width=view_config.theme.entry_field_styling.border_width,
            border_color=view_config.theme.entry_field_styling.border_color
        )
        textbox.grid(row=row, column=0, sticky="nsew", 
                    pady=(view_config.theme.spacing.tiny, view_config.theme.spacing.small), 
                    padx=view_config.theme.spacing.medium)
        
        # Insert example text with prefix
        full_example = f"{view_config.form_defaults.example_prefix}{example_text}"
        textbox.insert("1.0", full_example)
        
        # Make placeholder disappear on click
        def on_textbox_click(event):
            current_text = textbox.get("1.0", "end-1c")
            if current_text.startswith(view_config.form_defaults.example_prefix):
                textbox.delete("1.0", "end")
        
        textbox.bind("<Button-1>", on_textbox_click)
        return textbox 