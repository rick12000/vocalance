"""
Command edit dialog with 2-tile structure for editing and deleting commands.
Replaces the simple text input with a comprehensive themed dialog.
"""

import logging
import tkinter as tk
from typing import Optional, Tuple

import customtkinter as ctk

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.ui import ui_theme
from vocalance.app.ui.utils.ui_icon_utils import set_window_icon_robust
from vocalance.app.ui.utils.window_positioning import center_window_on_parent
from vocalance.app.ui.views.components.themed_components import (
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

            if self.parent:
                dialog = ctk.CTkToplevel(self.parent)
            else:
                dialog = ctk.CTk()

            dialog.title(f"Edit Command: {self.command.command_key}")
            dialog.minsize(ui_theme.theme.dimensions.command_dialog_width, ui_theme.theme.dimensions.command_dialog_min_height)

            if self.parent:
                dialog.transient(self.parent)
                dialog.grab_set()

            dialog.configure(fg_color=ui_theme.theme.shape_colors.darkest)

            # Set icon immediately
            try:
                set_window_icon_robust(dialog)
            except Exception:
                pass

            # Reinforce icon after dialog is displayed to prevent CustomTkinter override
            dialog.after(50, lambda: self._reinforce_icon(dialog))
            dialog.after(200, lambda: self._reinforce_icon(dialog))

            main_frame = TransparentFrame(dialog)
            main_frame.grid(
                row=0,
                column=0,
                sticky="ew",
                padx=ui_theme.theme.two_box_layout.base_spacing,
                pady=ui_theme.theme.two_box_layout.base_spacing,
            )

            dialog.grid_columnconfigure(0, weight=1)

            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_rowconfigure(0, weight=0)
            main_frame.grid_rowconfigure(1, weight=0)
            main_frame.grid_rowconfigure(2, weight=0)
            main_frame.grid_rowconfigure(3, weight=1)

            description_tile = ThemedFrame(main_frame, fg_color=ui_theme.theme.shape_colors.dark, border_width=0)
            description_tile.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, ui_theme.theme.spacing.medium))
            description_tile.grid_columnconfigure(0, weight=1)
            description_tile.grid_rowconfigure(0, weight=0)
            description_tile.grid_rowconfigure(1, weight=0)

            description_title = ThemedLabel(
                description_tile, text="Description", bold=True, color=ui_theme.theme.text_colors.light
            )
            description_title.grid(
                row=0,
                column=0,
                sticky="w",
                padx=ui_theme.theme.two_box_layout.title_padx_left,
                pady=(ui_theme.theme.spacing.small, ui_theme.theme.spacing.tiny),
            )

            description_text = self._get_command_description()
            wrap_length = (
                ui_theme.theme.dimensions.command_dialog_width
                - (ui_theme.theme.two_box_layout.base_spacing * 2)
                - (ui_theme.theme.two_box_layout.title_padx_left * 2)
            )
            description_label = ThemedLabel(
                description_tile,
                text=description_text,
                color=ui_theme.theme.text_colors.light,
                justify="left",
                wraplength=wrap_length,
            )
            description_label.grid(
                row=1,
                column=0,
                sticky="w",
                padx=ui_theme.theme.two_box_layout.title_padx_left,
                pady=(0, ui_theme.theme.spacing.small),
            )

            main_frame.grid_rowconfigure(0, weight=0)
            main_frame.grid_rowconfigure(1, weight=0)
            main_frame.grid_rowconfigure(2, weight=0)
            main_frame.grid_rowconfigure(3, weight=1)

            edit_tile = ThemedFrame(main_frame, fg_color=ui_theme.theme.shape_colors.dark, border_width=0)
            edit_tile.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, ui_theme.theme.spacing.small))
            edit_tile.grid_columnconfigure(0, weight=1)
            edit_tile.grid_rowconfigure(0, weight=0)
            edit_tile.grid_rowconfigure(1, weight=0)
            edit_tile.grid_rowconfigure(2, weight=0)

            # Edit title
            edit_title = ThemedLabel(edit_tile, text="Edit Command Phrase", bold=True, color=ui_theme.theme.text_colors.light)
            edit_title.grid(
                row=0,
                column=0,
                sticky="w",
                padx=ui_theme.theme.two_box_layout.title_padx_left,
                pady=(ui_theme.theme.spacing.small, ui_theme.theme.spacing.tiny),
            )

            # Entry field (removed "New phrase:" label)
            self.entry = ThemedEntry(edit_tile, width=ui_theme.theme.dimensions.entry_width_large)
            self.entry.grid(
                row=1,
                column=0,
                sticky="ew",
                padx=ui_theme.theme.two_box_layout.title_padx_left,
                pady=(0, ui_theme.theme.spacing.small),
            )
            self.entry.insert(0, self.command.command_key)
            self.entry.select_range(0, tk.END)

            # Edit button frame
            edit_button_frame = TransparentFrame(edit_tile)
            edit_button_frame.grid(
                row=2,
                column=0,
                sticky="ew",
                padx=ui_theme.theme.two_box_layout.title_padx_left,
                pady=(0, ui_theme.theme.spacing.small),
            )
            edit_button_frame.grid_columnconfigure(0, weight=1)

            # Save button
            save_btn = PrimaryButton(
                edit_button_frame,
                text=ui_theme.theme.button_text.save_changes,
                command=lambda: self._on_save(dialog),
                compact=False,
            )
            save_btn.grid(row=0, column=0, sticky="ew")

            # Delete tile
            delete_tile = ThemedFrame(main_frame, fg_color=ui_theme.theme.shape_colors.dark, border_width=0)
            delete_tile.grid(row=2, column=0, sticky="ew", padx=0, pady=(0, ui_theme.theme.spacing.small))
            delete_tile.grid_columnconfigure(0, weight=1)

            # Delete title (removed trash emoji)
            delete_title = ThemedLabel(delete_tile, text="Delete Command", bold=True, color=ui_theme.theme.text_colors.light)
            delete_title.grid(
                row=0,
                column=0,
                sticky="w",
                padx=ui_theme.theme.two_box_layout.title_padx_left,
                pady=(ui_theme.theme.spacing.small, ui_theme.theme.spacing.tiny),
            )

            # Delete content based on command type
            if self.command.is_custom:
                # Custom command - can be deleted
                delete_desc = ThemedLabel(
                    delete_tile, text="This is a custom command and can be safely deleted.", color=ui_theme.theme.text_colors.light
                )
                delete_desc.grid(
                    row=1,
                    column=0,
                    sticky="w",
                    padx=ui_theme.theme.two_box_layout.title_padx_left,
                    pady=(0, ui_theme.theme.spacing.small),
                )

                # Delete button frame
                delete_button_frame = TransparentFrame(delete_tile)
                delete_button_frame.grid(
                    row=2,
                    column=0,
                    sticky="ew",
                    padx=ui_theme.theme.two_box_layout.title_padx_left,
                    pady=(0, ui_theme.theme.spacing.small),
                )
                delete_button_frame.grid_columnconfigure(0, weight=1)

                # Delete button
                delete_btn = DangerButton(
                    delete_button_frame,
                    text=ui_theme.theme.button_text.delete_command,
                    command=lambda: self._on_delete(dialog),
                    compact=False,
                )
                delete_btn.grid(row=0, column=0, sticky="ew")
            else:
                # Built-in command - cannot be deleted
                delete_desc = ThemedLabel(
                    delete_tile, text="This is a built-in command and cannot be deleted.", color=ui_theme.theme.text_colors.light
                )
                delete_desc.grid(
                    row=1,
                    column=0,
                    sticky="w",
                    padx=ui_theme.theme.two_box_layout.title_padx_left,
                    pady=(0, ui_theme.theme.spacing.small),
                )

            # Focus the entry field
            self.entry.focus_set()

            # Bind Enter key to save
            dialog.bind("<Return>", lambda e: self._on_save(dialog))

            # Center the dialog
            if self.parent:
                center_window_on_parent(dialog, self.parent)

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

            # Directly delete without confirmation
            self.result = "delete"
            self.new_phrase = None
            self.logger.info(f"Deleted command: '{self.command.command_key}'")
            dialog.destroy()
        except Exception as e:
            self.logger.error(f"Error deleting command: {e}", exc_info=True)

    def _on_cancel(self, dialog):
        """Handle cancel button click"""
        self.result = None
        self.new_phrase = None
        self.logger.info("Dialog cancelled")
        dialog.destroy()

    def _reinforce_icon(self, dialog):
        """Reinforce the icon setting to prevent CustomTkinter override."""
        if dialog and dialog.winfo_exists():
            try:
                set_window_icon_robust(dialog)
                dialog.update_idletasks()
            except Exception:
                pass
