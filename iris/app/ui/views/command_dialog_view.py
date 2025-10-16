"""
Command edit dialog with 2-tile structure for editing and deleting commands.
Replaces the simple text input with a comprehensive themed dialog.
"""

import logging
import tkinter as tk
from typing import Optional, Tuple

import customtkinter as ctk

from iris.app.config.command_types import AutomationCommand
from iris.app.ui import ui_theme
from iris.app.ui.utils.ui_icon_utils import set_window_icon_robust
from iris.app.ui.views.components import themed_dialogs as messagebox
from iris.app.ui.views.components.themed_components import (
    DangerButton,
    PrimaryButton,
    ThemedEntry,
    ThemedFrame,
    ThemedLabel,
    TransparentFrame,
)


class CommandEditDialog:
    """Dialog for editing command phrases"""

    def __init__(self, command: AutomationCommand, parent=None):
        self.command = command
        self.parent = parent
        self.result = None
        self.new_phrase = None
        self.entry = None
        self.logger = logging.getLogger("CommandEditDialog")

    def show(self) -> Tuple[Optional[str], Optional[str]]:
        """Show the dialog and return (action, new_phrase)"""
        try:
            self.logger.info(f"Creating command edit dialog for: '{self.command.command_key}'")

            # Create dialog window
            if self.parent:
                dialog = ctk.CTkToplevel(self.parent)
            else:
                # Fallback for testing
                dialog = ctk.CTk()

            dialog.title(f"Edit Command: {self.command.command_key}")
            dialog.geometry(f"{ui_theme.theme.dimensions.command_dialog_width}x{ui_theme.theme.dimensions.command_dialog_height}")

            if self.parent:
                dialog.transient(self.parent)
                dialog.grab_set()

            # Set icon on dialog
            try:
                set_window_icon_robust(dialog)
            except Exception:
                pass  # Silently fail if icon can't be set

            # Center the dialog
            dialog.update_idletasks()
            if self.parent:
                x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (dialog.winfo_width() // 2)
                y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (dialog.winfo_height() // 2)
            else:
                x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
                y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")

            # Set dialog background color
            dialog.configure(fg_color=ui_theme.theme.shape_colors.darkest)

            # Main frame - transparent to match dialog background
            main_frame = TransparentFrame(dialog)
            main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

            # Configure dialog grid
            dialog.grid_columnconfigure(0, weight=1)
            dialog.grid_rowconfigure(0, weight=1)

            # Configure main frame grid
            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_rowconfigure(0, weight=0)  # Command info
            main_frame.grid_rowconfigure(1, weight=0)  # Edit tile
            main_frame.grid_rowconfigure(2, weight=0)  # Delete tile
            main_frame.grid_rowconfigure(3, weight=1)  # Bottom frame

            # Command information at the top (centered)
            info_frame = TransparentFrame(main_frame)
            info_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 20))
            info_frame.grid_columnconfigure(0, weight=1)

            # Command description (with "Description:" prefix)
            description_text = self._get_command_description()
            description_label = ThemedLabel(
                info_frame, text=f"Description: {description_text}", color=ui_theme.theme.text_colors.light, justify="center"
            )
            description_label.grid(row=0, column=0, pady=(0, 0))

            # Edit tile
            edit_tile = ThemedFrame(main_frame, fg_color=ui_theme.theme.shape_colors.dark, border_width=0)
            edit_tile.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 15))
            edit_tile.grid_columnconfigure(0, weight=1)
            edit_tile.grid_rowconfigure(0, weight=0)
            edit_tile.grid_rowconfigure(1, weight=0)
            edit_tile.grid_rowconfigure(2, weight=0)

            # Edit title
            edit_title = ThemedLabel(edit_tile, text="Edit Command Phrase", bold=True, color=ui_theme.theme.text_colors.light)
            edit_title.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

            # Entry field (removed "New phrase:" label)
            self.entry = ThemedEntry(edit_tile, width=ui_theme.theme.dimensions.entry_width_large)
            self.entry.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 10))
            self.entry.insert(0, self.command.command_key)
            self.entry.select_range(0, tk.END)

            # Edit button frame
            edit_button_frame = TransparentFrame(edit_tile)
            edit_button_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))
            edit_button_frame.grid_columnconfigure(0, weight=1)

            # Save button
            save_btn = PrimaryButton(
                edit_button_frame, text=ui_theme.theme.button_text.save_changes, command=lambda: self._on_save(dialog)
            )
            save_btn.grid(row=0, column=0, sticky="ew")

            # Delete tile
            delete_tile = ThemedFrame(main_frame, fg_color=ui_theme.theme.shape_colors.dark, border_width=0)
            delete_tile.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 15))
            delete_tile.grid_columnconfigure(0, weight=1)

            # Delete title (removed trash emoji)
            delete_title = ThemedLabel(delete_tile, text="Delete Command", bold=True, color=ui_theme.theme.text_colors.light)
            delete_title.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 5))

            # Delete content based on command type
            if self.command.is_custom:
                # Custom command - can be deleted
                delete_desc = ThemedLabel(
                    delete_tile, text="This is a custom command and can be safely deleted.", color=ui_theme.theme.text_colors.light
                )
                delete_desc.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 10))

                # Delete button frame
                delete_button_frame = TransparentFrame(delete_tile)
                delete_button_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))
                delete_button_frame.grid_columnconfigure(0, weight=1)

                # Delete button
                delete_btn = DangerButton(
                    delete_button_frame, text=ui_theme.theme.button_text.delete_command, command=lambda: self._on_delete(dialog)
                )
                delete_btn.grid(row=0, column=0, sticky="ew")
            else:
                # Built-in command - cannot be deleted
                delete_desc = ThemedLabel(
                    delete_tile, text="This is a built-in command and cannot be deleted.", color=ui_theme.theme.text_colors.light
                )
                delete_desc.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 15))

            # Bottom frame for cancel button
            bottom_frame = TransparentFrame(main_frame)
            bottom_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 20))
            bottom_frame.grid_columnconfigure(0, weight=1)

            # Cancel button
            cancel_btn = PrimaryButton(
                bottom_frame, text=ui_theme.theme.button_text.cancel, command=lambda: self._on_cancel(dialog)
            )
            cancel_btn.grid(row=0, column=0, sticky="ew")

            # Focus the entry field
            self.entry.focus_set()

            # Bind Enter key to save
            dialog.bind("<Return>", lambda e: self._on_save(dialog))
            dialog.bind("<Escape>", lambda e: self._on_cancel(dialog))

            # Wait for dialog to close
            dialog.wait_window()

            self.logger.info(f"Dialog closed, returning: action={self.result}, new_phrase={self.new_phrase}")
            return self.result, self.new_phrase

        except Exception as e:
            self.logger.error(f"Error in CommandEditDialog.show(): {e}", exc_info=True)
            return None, None

    def _get_command_description(self) -> str:
        """Get a detailed description of what the command does"""
        # Use long description if available
        if self.command.long_description:
            return self.command.long_description
        else:
            # Fallback to generating description based on action type
            if self.command.action_type == "hotkey":
                return f"Triggers hotkey: {self.command.action_value or 'Not set'}"
            elif self.command.action_type == "key":
                return f"Simulates pressing the key: {self.command.action_value or 'Not set'}"
            elif self.command.action_type == "key_sequence":
                return f"Executes key sequence: {self.command.action_value or 'Not set'}"
            elif self.command.action_type == "click":
                return f"Performs a mouse click action: {self.command.action_value or 'Left click'}"
            elif self.command.action_type == "scroll":
                return f"Performs a scroll action: {self.command.action_value or 'Scroll'}"
            elif self.command.action_type == "type":
                return f"Types the text: {self.command.action_value or 'No text set'}"
            else:
                return f"Custom action: {self.command.action_value or 'No action defined'}"

    def _on_save(self, dialog):
        """Handle save button click"""
        try:
            new_phrase = self.entry.get().strip()
            if new_phrase and new_phrase != self.command.command_key:
                self.result = "save"
                self.new_phrase = new_phrase
                self.logger.info(f"Saving changes: '{self.command.command_key}' -> '{new_phrase}'")
            else:
                self.result = None
                self.new_phrase = None
                self.logger.info("No changes to save")

            dialog.destroy()
        except Exception as e:
            self.logger.error(f"Error saving changes: {e}", exc_info=True)

    def _on_delete(self, dialog):
        """Handle delete button click"""
        try:
            if not self.command.is_custom:
                self.logger.warning("Attempted to delete builtin command")
                return

            # Show confirmation
            confirm = messagebox.askyesno(
                "Confirm Delete",
                f"Are you sure you want to delete the command '{self.command.command_key}'?\n\nThis action cannot be undone.",
                parent=dialog,
            )

            if confirm:
                self.result = "delete"
                self.new_phrase = None
                self.logger.info(f"Confirmed deletion of command: '{self.command.command_key}'")
                dialog.destroy()
        except Exception as e:
            self.logger.error(f"Error deleting command: {e}", exc_info=True)

    def _on_cancel(self, dialog):
        """Handle cancel button click"""
        self.result = None
        self.new_phrase = None
        self.logger.info("Dialog cancelled")
        dialog.destroy()
