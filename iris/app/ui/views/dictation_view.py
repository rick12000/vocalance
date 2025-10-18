import tkinter as tk
from typing import Any, Dict, List

import customtkinter as ctk

from iris.app.ui.controls.dictation_control import DictationController
from iris.app.ui.utils.ui_icon_utils import set_window_icon_robust
from iris.app.ui.views.components.base_view import ViewHelper
from iris.app.ui.views.components.form_builder import FormBuilder
from iris.app.ui.views.components.list_builder import ButtonType, ListBuilder, ListItemColumn
from iris.app.ui.views.components.themed_components import ThemedFrame, TwoColumnTabLayout
from iris.app.ui.views.components.view_config import view_config


class DictationView(ViewHelper):
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
        form_builder = FormBuilder()
        form_builder.setup_form_grid(container)

        # Create form fields using form builder
        self.title_label, self.title_entry = form_builder.create_labeled_entry(
            container, "Prompt Title:", "e.g. 'Professional Email Cleanup'"
        )

        self.prompt_label, self.prompt_textbox = form_builder.create_labeled_textbox(
            container,
            "Prompt Instructions:",
            "e.g. Convert informal speech to professional writing while preserving all key information and maintaining clarity.",
            height=100,
        )

        # Make placeholder disappear on click
        def on_textbox_click(event):
            current_text = self.prompt_textbox.get("1.0", "end-1c")
            if current_text.startswith("e.g. "):
                self.prompt_textbox.delete("1.0", "end")

        self.prompt_textbox.bind("<Button-1>", on_textbox_click)

        # Add button
        form_builder.create_button_row(
            container, [{"text": view_config.theme.button_text.add_prompt, "command": self._add_prompt, "type": "primary"}]
        )

    def _setup_manage_prompts_panel(self) -> None:
        """Setup the manage prompts panel"""
        container = self.layout.right_content
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.prompts_container = ListBuilder.create_scrollable_list_container(
            container,
            row=0,
            column=0,
            pady=(0, view_config.theme.two_box_layout.last_element_bottom_padding),
        )

    def _add_prompt(self) -> None:
        """Add a new prompt with validation"""
        title = self.title_entry.get().strip()
        prompt_text = self.prompt_textbox.get("1.0", "end-1c").strip()

        # Remove example prefix if present
        if prompt_text.startswith("e.g. "):
            prompt_text = ""

        # Validate inputs
        if not title:
            form_builder = FormBuilder()
            form_builder.show_temporary_message(self.layout.left_content, "Please enter a title for the prompt.")
            return

        if not prompt_text:
            form_builder = FormBuilder()
            form_builder.show_temporary_message(self.layout.left_content, "Please enter prompt instructions.")
            return

        # Add prompt and clear form on success
        if self.controller.add_prompt(title, prompt_text):
            self.clear_form_fields(self.title_entry, self.prompt_textbox)
            # Restore example text
            example_text = "e.g. Convert informal speech to professional writing while preserving all key information and maintaining clarity."
            self.prompt_textbox.insert("1.0", example_text)

    def display_prompts(self, prompts: List[Dict[str, Any]]) -> None:
        """Display prompts using simplified tile creation"""
        ListBuilder.display_items(
            container=self.prompts_container,
            items=prompts,
            create_item_callback=self._create_prompt_tile,
        )

    def _create_prompt_tile(self, prompt_data: Dict[str, Any], row_idx: int) -> None:
        """Create a simplified prompt tile with 4-column layout using ListBuilder"""

        def on_radio_select():
            self.selected_prompt_var.set(prompt_data["id"])
            self.controller.select_prompt(prompt_data["id"])

        def create_radio(parent: ctk.CTkFrame) -> ctk.CTkRadioButton:
            """Factory function to create the radio button"""
            return ctk.CTkRadioButton(
                parent,
                text="",
                width=24,
                variable=self.selected_prompt_var,
                value=prompt_data["id"],
                command=on_radio_select,
                fg_color=view_config.theme.shape_colors.light,
                hover_color=view_config.theme.shape_colors.lightest,
                border_width_checked=4,
                border_width_unchecked=2,
            )

        ListBuilder.create_list_item(
            container=self.prompts_container,
            row_index=row_idx,
            columns=[
                ListItemColumn.widget(widget_factory=create_radio, weight=0),
                ListItemColumn.label(text=prompt_data.get("name", "Unnamed"), weight=1),
                ListItemColumn.button(
                    text=view_config.theme.button_text.edit,
                    command=lambda: self._edit_prompt(prompt_data),
                    button_type=ButtonType.PRIMARY,
                ),
                ListItemColumn.button(
                    text=view_config.theme.button_text.delete,
                    command=lambda: self._delete_prompt(prompt_data["id"]),
                    button_type=ButtonType.DANGER,
                ),
            ],
        )

        if prompt_data.get("is_current", False):
            self.selected_prompt_var.set(prompt_data["id"])

    def _edit_prompt(self, prompt_data: Dict[str, Any]) -> None:
        """Edit prompt with simplified dialog"""
        dialog = ctk.CTkToplevel(self.root_window)
        dialog.title(f"Edit: {prompt_data.get('name', 'Unnamed')}")
        # Increase height to fit all elements
        dialog.geometry(
            f"{view_config.theme.dimensions.dictation_view_dialog_width}x{view_config.theme.dimensions.dictation_view_dialog_height + 100}"
        )
        dialog.transient(self.root_window)
        dialog.grab_set()

        # Set dialog background color
        dialog.configure(fg_color=view_config.theme.shape_colors.darkest)

        # Set icon
        try:
            set_window_icon_robust(dialog)
        except Exception:
            pass

        # Main frame with updated color
        main_frame = ThemedFrame(dialog, fg_color=view_config.theme.shape_colors.dark)
        main_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=view_config.theme.two_box_layout.inner_content_padx,
            pady=view_config.theme.two_box_layout.inner_content_padx,
        )

        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Form fields (changed "Name:" to "Title:")
        form_builder = FormBuilder()
        title_label, title_entry = form_builder.create_labeled_entry(
            main_frame, "Prompt Title:", default_value=prompt_data.get("name", "")
        )

        prompt_label, prompt_textbox = form_builder.create_labeled_textbox(
            main_frame, "Prompt Instructions:", height=200, text_color=view_config.theme.text_colors.light
        )
        prompt_textbox.insert("1.0", prompt_data.get("text", ""))

        # Buttons
        def save_changes():
            new_name = title_entry.get().strip()
            new_text = prompt_textbox.get("1.0", "end-1c").strip()

            # Validate inputs
            if not new_name:
                form_builder.show_temporary_message(main_frame, "Please enter a title for the prompt")
                return

            if not new_text:
                form_builder.show_temporary_message(main_frame, "Please enter instructions for the prompt")
                return

            # Try to save the changes
            if self.controller.edit_prompt(prompt_data["id"], new_name, new_text):
                dialog.destroy()
            else:
                form_builder.show_temporary_message(main_frame, "Failed to update prompt")

        form_builder.create_button_row(
            main_frame,
            [
                {"text": view_config.theme.button_text.save_changes, "command": save_changes, "type": "primary"},
                {"text": view_config.theme.button_text.cancel, "command": dialog.destroy, "type": "danger"},
            ],
        )

    def _delete_prompt(self, prompt_id: str) -> None:
        """Delete prompt"""
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
