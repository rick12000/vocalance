import customtkinter as ctk
import tkinter as tk
from typing import List, Dict, Any
import logging

from iris.app.ui.controls.dictation_control import DictationController
from iris.app.ui.views.components.base_view import BaseView
from iris.app.ui.views.components.form_builder import FormBuilder
from iris.app.ui.views.components.view_config import view_config
from iris.app.ui.views.components.themed_components import (
    TwoColumnTabLayout, PrimaryButton, ThemedLabel, ThemedScrollableFrame,
    BorderlessFrame, DangerButton, ThemedFrame
)
from iris.app.ui.utils.ui_icon_utils import set_window_icon_robust

class DictationView(BaseView):
    """Simplified dictation view using base components and form builder"""
    
    def __init__(self, parent, controller: DictationController):
        super().__init__(parent, controller)
        self.selected_prompt_var = tk.StringVar()
        self._setup_ui()
        self.controller.refresh_prompts()

    def _setup_ui(self) -> None:
        """Setup the main UI layout"""
        self.setup_main_layout()
        
        self.layout = TwoColumnTabLayout(self, "Add Custom Prompt", "Manage Prompts")
        self.layout.grid(row=0, column=0, sticky="nsew")
        
        self._setup_add_prompt_form()
        self._setup_manage_prompts_panel()

    def _setup_add_prompt_form(self) -> None:
        """Setup the add prompt form using form builder"""
        container = self.layout.left_content
        FormBuilder.setup_form_grid(container, 4)
        
        # Create form fields using form builder
        self.title_label, self.title_entry = FormBuilder.create_labeled_entry(
            container, "Prompt Title:", "e.g. 'Professional Email Cleanup'", row=0
        )
        
        self.prompt_label, self.prompt_textbox = FormBuilder.create_labeled_textbox(
            container, "Prompt Instructions:", "e.g. Convert informal speech to professional writing while preserving all key information and maintaining clarity.", row=2, height=100,
        )
        
        # Make placeholder disappear on click
        def on_textbox_click(event):
            current_text = self.prompt_textbox.get("1.0", "end-1c")
            if current_text.startswith("e.g. "):
                self.prompt_textbox.delete("1.0", "end")
        
        self.prompt_textbox.bind("<Button-1>", on_textbox_click)
        
        # Add button
        FormBuilder.create_button_row(
            container,
            [{"text": view_config.theme.button_text.add_prompt, "command": self._add_prompt, "type": "primary"}],
            row=4
        )

    def _setup_manage_prompts_panel(self) -> None:
        """Setup the manage prompts panel"""
        container = self.layout.right_content
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.prompts_container = ThemedScrollableFrame(container)
        self.prompts_container.grid(row=0, column=0, sticky="nsew", 
                                   padx=view_config.theme.two_box_layout.box_content_padding, 
                                   pady=(0, view_config.theme.two_box_layout.last_element_bottom_padding))

    def _add_prompt(self) -> None:
        """Add a new prompt with validation"""
        title = self.title_entry.get().strip()
        prompt_text = self.prompt_textbox.get("1.0", "end-1c").strip()
        
        # Remove example prefix if present
        if prompt_text.startswith("e.g. "):
            prompt_text = ""
        
        # Validate inputs
        if not title:
            FormBuilder.show_temporary_message(
                self.layout.left_content, "Please enter a title for the prompt.", 4
            )
            return
        
        if not prompt_text:
            FormBuilder.show_temporary_message(
                self.layout.left_content, "Please enter prompt instructions.", 4
            )
            return
        
        # Add prompt and clear form on success
        if self.controller.add_prompt(title, prompt_text):
            self.clear_form_fields(self.title_entry, self.prompt_textbox)
            # Restore example text
            example_text = "e.g. Convert informal speech to professional writing while preserving all key information and maintaining clarity."
            self.prompt_textbox.insert("1.0", example_text)

    def display_prompts(self, prompts: List[Dict[str, Any]]) -> None:
        """Display prompts using simplified tile creation"""
        # Clear existing prompts
        for widget in self.prompts_container.winfo_children():
            widget.destroy()
        
        self.prompts_container.grid_columnconfigure(0, weight=1)
        
        for row_idx, prompt_data in enumerate(prompts):
            self._create_prompt_tile(prompt_data, row_idx)

    def _create_prompt_tile(self, prompt_data: Dict[str, Any], row_idx: int) -> None:
        """Create a simplified prompt tile with 4-column layout"""
        # Create a borderless frame for the prompt item
        prompt_frame = BorderlessFrame(self.prompts_container)
        prompt_frame.grid(row=row_idx, column=0, sticky="ew", 
                         padx=view_config.theme.spacing.tiny, 
                         pady=view_config.theme.list_layout.item_vertical_spacing)
        
        # Configure 4-column grid
        prompt_frame.grid_columnconfigure(0, weight=0)  # Radio button
        prompt_frame.grid_columnconfigure(1, weight=1)  # Prompt title (expandable)
        prompt_frame.grid_columnconfigure(2, weight=0)  # Edit button
        prompt_frame.grid_columnconfigure(3, weight=0)  # Delete button
        
        def on_radio_select():
            self.selected_prompt_var.set(prompt_data['id'])
            self.controller.select_prompt(prompt_data['id'])
        
        # Column 1: Radio button
        radio = ctk.CTkRadioButton(
            prompt_frame,
            text="",
            width=24,  # NOTE: Hardcoded to width of radio button in Tkinter
            variable=self.selected_prompt_var,
            value=prompt_data['id'],
            command=on_radio_select,
            fg_color=view_config.theme.shape_colors.light,
            hover_color=view_config.theme.shape_colors.lightest,
            border_width_checked=4,
            border_width_unchecked=2,
        )
        radio.grid(row=0, column=0, padx=view_config.theme.spacing.tiny, pady=(view_config.theme.list_layout.item_vertical_spacing,0))
        
        # Column 2: Prompt title
        title_label = ThemedLabel(prompt_frame, text=prompt_data.get('name', 'Unnamed'), anchor="w", color=view_config.theme.text_colors.light)
        title_label.grid(row=0, column=1, sticky="ew", padx=view_config.theme.spacing.tiny, pady=(view_config.theme.list_layout.item_vertical_spacing,0))
        
        # Column 3: Edit button
        edit_button = PrimaryButton(prompt_frame, text=view_config.theme.button_text.edit, compact=True,
                                   command=lambda: self._edit_prompt(prompt_data))
        edit_button.grid(row=0, column=2, padx=view_config.theme.spacing.tiny, pady=(view_config.theme.list_layout.item_vertical_spacing,0))
        
        # Column 4: Delete button
        delete_button = DangerButton(prompt_frame, text=view_config.theme.button_text.delete, compact=True,
                                    command=lambda: self._delete_prompt(prompt_data['id']))
        delete_button.grid(row=0, column=3, padx=view_config.theme.spacing.tiny, pady=(view_config.theme.list_layout.item_vertical_spacing,0))
        
        # Set selection state
        if prompt_data.get('is_current', False):
            self.selected_prompt_var.set(prompt_data['id'])

    def _edit_prompt(self, prompt_data: Dict[str, Any]) -> None:
        """Edit prompt with simplified dialog"""
        dialog = ctk.CTkToplevel(self.root_window)
        dialog.title(f"Edit Prompt: {prompt_data.get('name', 'Unnamed')}")
        # Increase height to fit all elements
        dialog.geometry(f"{view_config.theme.dimensions.dictation_view_dialog_width}x{view_config.theme.dimensions.dictation_view_dialog_height + 100}")
        dialog.transient(self.root_window)
        dialog.grab_set()
        
        # Set dialog background color
        dialog.configure(fg_color=view_config.theme.shape_colors.darkest)
        
        # Set icon
        try:
            set_window_icon_robust(dialog)
        except:
            pass
        
        # Main frame with updated color
        main_frame = ThemedFrame(dialog, fg_color=view_config.theme.shape_colors.dark)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Form fields (changed "Name:" to "Title:")
        title_label, title_entry = FormBuilder.create_labeled_entry(
            main_frame, "Title:", default_value=prompt_data.get('name', ''), row=0
        )
        
        prompt_label, prompt_textbox = FormBuilder.create_labeled_textbox(
            main_frame, "Prompt:", height=200, row=2
        )
        prompt_textbox.insert("1.0", prompt_data.get('text', ''))
        
        # Buttons
        def save_changes():
            new_name = title_entry.get().strip()
            new_text = prompt_textbox.get("1.0", "end-1c").strip()
            
            # Validate inputs
            if not new_name:
                FormBuilder.show_temporary_message(main_frame, "Please enter a name for the prompt", 5)
                return
            
            if not new_text:
                FormBuilder.show_temporary_message(main_frame, "Please enter prompt text", 5)
                return
            
            # Try to save the changes
            if self.controller.edit_prompt(prompt_data['id'], new_name, new_text):
                dialog.destroy()
            else:
                FormBuilder.show_temporary_message(main_frame, "Failed to update prompt", 5)
        
        FormBuilder.create_button_row(
            main_frame,
            [
                {"text": view_config.theme.button_text.save_changes, "command": save_changes, "type": "primary"},
                {"text": view_config.theme.button_text.cancel, "command": dialog.destroy, "type": "danger"}
            ],
            row=4
        )

    def _delete_prompt(self, prompt_id: str) -> None:
        """Delete prompt with confirmation"""
        if self.show_delete_confirmation("this prompt"):
            self.controller.delete_prompt(prompt_id)

    # Simplified callback methods
    def on_prompts_updated(self, prompts: List[Dict[str, Any]]) -> None:
        """Handle prompts list updated"""
        self.display_prompts(prompts)

    def on_current_prompt_updated(self, prompt_id: str) -> None:
        """Handle current prompt updated"""
        self.display_prompts(self.controller.get_prompts())

    def on_status_update(self, message: str, is_error: bool = False) -> None:
        """Handle status updates - now simplified"""
        super().on_status_update(message, is_error) 