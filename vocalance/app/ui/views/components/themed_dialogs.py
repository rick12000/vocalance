"""
Themed dialog components that match the application's dark theme.
Provides CustomTkinter-based replacements for standard tkinter messageboxes.

Thread Safety:
- All dialog operations must run in main tkinter thread
- Dialogs are modal and block until closed
- Safe to call from event handlers (already in event loop thread)
"""

from typing import Callable, List, Optional, Tuple

import customtkinter as ctk

from vocalance.app.ui import ui_theme
from vocalance.app.ui.utils.ui_icon_utils import set_window_icon_robust
from vocalance.app.ui.utils.window_positioning import center_window_on_parent
from vocalance.app.ui.views.components.themed_components import DangerButton, PrimaryButton


def _get_button_class(button_text: str):
    """Get the appropriate themed button class based on button text."""
    if button_text.lower() == ui_theme.theme.button_text.yes.lower():
        return DangerButton
    elif button_text.lower() in [ui_theme.theme.button_text.no.lower(), ui_theme.theme.button_text.cancel.lower()]:
        return PrimaryButton
    else:
        # Default to PrimaryButton for other button texts like OK
        return PrimaryButton


def _setup_dialog_window(dialog: ctk.CTkToplevel, parent=None) -> None:
    """Helper function to set up common dialog window properties including icon."""
    dialog.minsize(ui_theme.theme.dimensions.dialog_width, ui_theme.theme.dimensions.dialog_min_height)
    dialog.configure(fg_color=ui_theme.theme.shape_colors.darkest)

    try:
        set_window_icon_robust(dialog)
    except Exception:
        pass

    dialog.transient(parent)
    dialog.grab_set()


def _create_dialog_base(
    message: str,
    parent=None,
    buttons: Optional[List[Tuple[str, Callable]]] = None,
) -> Optional[bool]:
    """
    Base function for creating themed dialogs with configurable buttons.

    Thread Safety:
    - Must be called from main tkinter thread
    - Dialog is modal (blocks until closed)
    - Called from controllers via schedule_ui_update

    Args:
        message: Dialog message
        parent: Parent window
        buttons: List of (button_text, callback) tuples. Callback receives dialog as arg.

    Returns:
        Result from button callback if applicable, None otherwise
    """
    result = [None]

    dialog = ctk.CTkToplevel(parent)
    _setup_dialog_window(dialog, parent)

    dialog.grid_columnconfigure(0, weight=1)

    main_frame = ctk.CTkFrame(
        dialog, fg_color=ui_theme.theme.shape_colors.darkest, corner_radius=ui_theme.theme.border_radius.small
    )
    main_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)

    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=0)
    main_frame.grid_rowconfigure(1, weight=1)

    message_label = ctk.CTkLabel(
        main_frame,
        text=message,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light,
        wraplength=350,
    )
    message_label.grid(row=0, column=0, pady=(20, 20), sticky="ew")

    if buttons:
        if len(buttons) == 1:
            # Single button centered
            btn_text, btn_callback = buttons[0]
            button_class = _get_button_class(btn_text)
            btn = button_class(
                main_frame,
                text=btn_text,
                command=lambda: [result.__setitem__(0, btn_callback()), dialog.destroy()][1],
                compact=False,
            )
            btn.grid(row=1, column=0, pady=(0, 20))
            dialog.bind("<Return>", lambda e: [result.__setitem__(0, btn_callback()), dialog.destroy()])
            dialog.bind("<Escape>", lambda e: [result.__setitem__(0, btn_callback()), dialog.destroy()])
        else:
            # Multiple buttons in a row
            button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
            button_frame.grid(row=1, column=0, pady=(0, 20), sticky="ew")

            for i, (btn_text, btn_callback) in enumerate(buttons):
                button_frame.grid_columnconfigure(i, weight=1)
                button_class = _get_button_class(btn_text)
                btn = button_class(
                    button_frame,
                    text=btn_text,
                    command=lambda cb=btn_callback: [result.__setitem__(0, cb()), dialog.destroy()][1],
                    compact=False,
                )
                btn.grid(row=0, column=i, padx=5, sticky="ew")

            # Bind keyboard shortcuts to first two buttons
            if len(buttons) >= 2:
                first_callback = buttons[0][1]
                second_callback = buttons[1][1]
                dialog.bind("<Return>", lambda e: [result.__setitem__(0, first_callback()), dialog.destroy()])
                dialog.bind("<Escape>", lambda e: [result.__setitem__(0, second_callback()), dialog.destroy()])

    center_window_on_parent(dialog, parent)
    dialog.focus_force()
    dialog.wait_window()

    return result[0]


def askokcancel(message: str, parent=None) -> bool:
    """Show a themed OK/Cancel dialog and return True if OK was clicked."""
    result = _create_dialog_base(
        message=message,
        parent=parent,
        buttons=[
            (ui_theme.theme.button_text.ok, lambda: True),
            (ui_theme.theme.button_text.cancel, lambda: False),
        ],
    )
    return result if result is not None else False


def askyesno(message: str, parent=None) -> bool:
    """Show a themed Yes/No dialog and return True if Yes was clicked."""
    result = _create_dialog_base(
        message=message,
        parent=parent,
        buttons=[
            (ui_theme.theme.button_text.yes, lambda: True),
            (ui_theme.theme.button_text.no, lambda: False),
        ],
    )
    return result if result is not None else False


def showerror(message: str, parent=None):
    """Show a themed error dialog."""
    _create_dialog_base(
        message=message,
        parent=parent,
        buttons=[(ui_theme.theme.button_text.ok, lambda: None)],
    )


def showinfo(message: str, parent=None):
    """Show a themed info dialog."""
    _create_dialog_base(
        message=message,
        parent=parent,
        buttons=[(ui_theme.theme.button_text.ok, lambda: None)],
    )


def showwarning(message: str, parent=None):
    """Show a themed warning dialog."""
    _create_dialog_base(
        message=message,
        parent=parent,
        buttons=[(ui_theme.theme.button_text.ok, lambda: None)],
    )
