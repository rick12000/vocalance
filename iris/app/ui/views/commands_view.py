import customtkinter as ctk
import tkinter as tk
from typing import List, Callable
import logging

from iris.app.config.command_types import AutomationCommand
from iris.app.ui.views.command_dialog_view import CommandEditDialog
from iris.app.ui.views.components.base_view import BaseView
from iris.app.ui.views.components.form_builder import FormBuilder
from iris.app.ui.views.components.view_config import view_config
from iris.app.ui.views.components.themed_components import (
    TwoColumnTabLayout, ThemedScrollableFrame, BorderlessFrame, 
    PrimaryButton, DangerButton, ThemedLabel
)

class CommandsView(BaseView):
    """Simplified commands view using base components and form builder"""
    
    def __init__(self, parent, controller, root_window, logger: logging.Logger):
        super().__init__(parent, controller, root_window)
        self.logger = logger
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the main UI layout"""
        self.setup_main_layout()
        
        self.layout = TwoColumnTabLayout(self, "Add Command", "Manage Commands")
        self.layout.grid(row=0, column=0, sticky="nsew")
        
        self._setup_add_command_form()
        self._setup_commands_list_panel()
    
    def _setup_add_command_form(self) -> None:
        """Setup add command form using form builder"""
        container = self.layout.left_box
        FormBuilder.setup_form_grid(container, 6)
        
        # Form fields
        self.phrase_label, self.command_phrase_entry = FormBuilder.create_labeled_entry(
            container, "Command Phrase:", "Enter command phrase...", row=0
        )
        
        self.hotkey_label, self.hotkey_entry = FormBuilder.create_labeled_entry(
            container, "Hotkey:", "e.g., ctrl+alt+7", row=2
        )
        
        # Add button
        FormBuilder.create_button_row(
            container,
            [{"text": view_config.theme.button_text.add_command, "command": self._on_add_command_clicked, "type": "primary"}],
            row=4
        )
    
    def _setup_commands_list_panel(self) -> None:
        """Setup commands list panel"""
        container = self.layout.right_box
        container.grid_rowconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=0)
        container.grid_columnconfigure(0, weight=1)
        
        # Scrollable list
        self.command_list_container = ThemedScrollableFrame(container)
        self.command_list_container.grid(row=0, column=0, sticky="nsew", 
                                        padx=view_config.theme.spacing.small, 
                                        pady=(view_config.theme.spacing.small, 0))
        
        # Buttons
        FormBuilder.create_button_row(
            container,
            [
                {"text": view_config.theme.button_text.reset, "command": self._on_reset_to_defaults_clicked, "type": "danger"}
            ],
            row=1
        )
    
    def _on_add_command_clicked(self) -> None:
        """Handle add command button click"""
        command_phrase = self.command_phrase_entry.get().strip()
        hotkey_value = self.hotkey_entry.get().strip()
        self.controller.handle_add_command(command_phrase, hotkey_value)
    
    def clear_add_command_form(self) -> None:
        """Clear the add command form"""
        self.clear_form_fields(self.command_phrase_entry, self.hotkey_entry)
    
    def display_commands(self, commands: List[AutomationCommand]) -> None:
        """Display commands in a scrollable list"""
        self.logger.info(f"CommandsView: display_commands called with {len(commands)} commands")
        
        # Clear existing content
        if self.command_list_container:
            for widget in self.command_list_container.winfo_children():
                widget.destroy()
        
        self.command_list_container.grid_columnconfigure(0, weight=1)
        
        # Build command list items
        for i, command in enumerate(commands):
            self._create_command_item(command, i)
        
        self.logger.info(f"CommandsView: Finished displaying {len(commands)} commands")
    
    def _create_command_item(self, command: AutomationCommand, row_idx: int) -> None:
        """Create a streamlined command list item with only trigger word and change button"""
        command_frame = BorderlessFrame(self.command_list_container)
        command_frame.grid(row=row_idx, column=0, sticky="ew", 
                          padx=view_config.theme.spacing.tiny, 
                          pady=view_config.theme.list_layout.item_vertical_spacing)
        
        # Simplified grid configuration - only trigger word and change button
        command_frame.grid_columnconfigure(0, weight=1)  # Command phrase (expandable)
        command_frame.grid_columnconfigure(1, weight=0)  # Change button
        
        # Command phrase with color coding
        phrase_color = (
        view_config.theme.text_colors.light
        )

        ThemedLabel(
            command_frame,
            text=command.command_key,
            color=phrase_color,
            anchor="w"
        ).grid(row=0, column=0, sticky="ew", padx=(5, 15))
        
        # Change button
        PrimaryButton(command_frame, text=view_config.theme.button_text.change, compact=True,
                      command=lambda c=command: self.handle_change_command(c)).grid(
            row=0, column=1, padx=view_config.theme.spacing.tiny
        )
    
    def handle_change_command(self, command: AutomationCommand) -> None:
        """Handle changing a command"""
        dialog = CommandEditDialog(command, self.root_window)
        action, new_phrase = dialog.show()
        
        if action == "save" and new_phrase:
            self.controller.handle_change_command_phrase(command, new_phrase)
        elif action == "delete":
            self.controller.handle_delete_command(command)
    
    def handle_delete_command(self, command: AutomationCommand) -> None:
        """Handle deleting a command"""
        if self.show_delete_confirmation(f"command '{command.command_key}'"):
            self.controller.handle_delete_command(command)
    
    def _on_reset_to_defaults_clicked(self) -> None:
        """Handle reset to defaults button click"""
        if self.show_confirmation("Reset to Defaults", 
                                 "Are you sure you want to reset all commands to defaults? This will remove all custom commands."):
            self.controller.handle_reset_to_defaults()
    
    # Simplified callback methods - inherit most from BaseView
    def show_error_message(self, title: str, message: str) -> None:
        """Show an error message dialog"""
        self.show_error(title, message)
    
    def show_success_message(self, title: str, message: str) -> None:
        """Show a success message dialog"""
        self.show_info(title, message)
    
    def show_confirmation_dialog(self, title: str, message: str) -> bool:
        """Show a confirmation dialog and return the result"""
        return self.show_confirmation(title, message)
