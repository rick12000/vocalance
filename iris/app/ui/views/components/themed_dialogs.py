"""
Themed dialog components that match the application's dark theme.
Provides CustomTkinter-based replacements for standard tkinter messageboxes.
"""

from typing import Callable, List, Optional, Tuple

import customtkinter as ctk

from iris.app.ui import ui_theme
from iris.app.ui.utils.ui_icon_utils import set_window_icon_robust


def _setup_dialog_window(dialog: ctk.CTkToplevel, title: str, parent=None) -> None:
    """Helper function to set up common dialog window properties including icon."""
    dialog.title(title)
    dialog.geometry(f"{ui_theme.theme.dimensions.dialog_width}x{ui_theme.theme.dimensions.dialog_height}")
    dialog.resizable(False, False)
    dialog.configure(fg_color=ui_theme.theme.shape_colors.darkest)

    # Set icon on dialog
    try:
        set_window_icon_robust(dialog)
    except Exception:
        pass  # Silently fail if icon can't be set

    # Center the dialog
    dialog.transient(parent)
    dialog.grab_set()

    # Center on parent
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (200)
    y = (dialog.winfo_screenheight() // 2) - (100)
    dialog.geometry(f"{ui_theme.theme.dimensions.dialog_width}x{ui_theme.theme.dimensions.dialog_height}+{x}+{y}")


def _create_dialog_base(
    title: str,
    message: str,
    parent=None,
    buttons: Optional[List[Tuple[str, Callable]]] = None,
) -> Optional[bool]:
    """
    Base function for creating themed dialogs with configurable buttons.

    Args:
        title: Dialog title
        message: Dialog message
        parent: Parent window
        buttons: List of (button_text, callback) tuples. Callback receives dialog as arg.

    Returns:
        Result from button callback if applicable, None otherwise
    """
    result = [None]

    dialog = ctk.CTkToplevel(parent)
    _setup_dialog_window(dialog, title, parent)

    dialog.grid_rowconfigure(0, weight=1)
    dialog.grid_columnconfigure(0, weight=1)

    main_frame = ctk.CTkFrame(
        dialog, fg_color=ui_theme.theme.shape_colors.darkest, corner_radius=ui_theme.theme.border_radius.small
    )
    main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=0)
    main_frame.grid_rowconfigure(1, weight=0)
    main_frame.grid_rowconfigure(2, weight=1)

    title_label = ctk.CTkLabel(
        main_frame,
        text=title,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light,
    )
    title_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")

    message_label = ctk.CTkLabel(
        main_frame,
        text=message,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light,
        wraplength=350,
    )
    message_label.grid(row=1, column=0, pady=(0, 20), sticky="ew")

    if buttons:
        if len(buttons) == 1:
            # Single button centered
            btn_text, btn_callback = buttons[0]
            btn = ctk.CTkButton(
                main_frame,
                text=btn_text,
                font=ui_theme.theme.font_family.get_button_font(),
                fg_color=ui_theme.theme.shape_colors.darkest,
                hover_color=ui_theme.theme.shape_colors.dark,
                text_color=ui_theme.theme.text_colors.light,
                width=80,
                corner_radius=ui_theme.theme.border_radius.rounded,
                command=lambda: [result.__setitem__(0, btn_callback()), dialog.destroy()][1],
            )
            btn.grid(row=2, column=0, pady=(0, 20))
            dialog.bind("<Return>", lambda e: [result.__setitem__(0, btn_callback()), dialog.destroy()])
            dialog.bind("<Escape>", lambda e: [result.__setitem__(0, btn_callback()), dialog.destroy()])
        else:
            # Multiple buttons in a row
            button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
            button_frame.grid(row=2, column=0, pady=(0, 20), sticky="ew")

            for i, (btn_text, btn_callback) in enumerate(buttons):
                button_frame.grid_columnconfigure(i, weight=1)
                btn = ctk.CTkButton(
                    button_frame,
                    text=btn_text,
                    font=ui_theme.theme.font_family.get_button_font(),
                    fg_color=ui_theme.theme.shape_colors.darkest,
                    hover_color=ui_theme.theme.shape_colors.dark,
                    text_color=ui_theme.theme.text_colors.light,
                    width=80,
                    corner_radius=ui_theme.theme.border_radius.rounded,
                    command=lambda cb=btn_callback: [result.__setitem__(0, cb()), dialog.destroy()][1],
                )
                btn.grid(row=0, column=i, padx=5, sticky="ew")

            # Bind keyboard shortcuts to first two buttons
            if len(buttons) >= 2:
                first_callback = buttons[0][1]
                second_callback = buttons[1][1]
                dialog.bind("<Return>", lambda e: [result.__setitem__(0, first_callback()), dialog.destroy()])
                dialog.bind("<Escape>", lambda e: [result.__setitem__(0, second_callback()), dialog.destroy()])

    dialog.focus_force()
    dialog.wait_window()

    return result[0]


def askokcancel(title: str, message: str, parent=None) -> bool:
    """Show a themed OK/Cancel dialog and return True if OK was clicked."""
    result = _create_dialog_base(
        title=title,
        message=message,
        parent=parent,
        buttons=[
            (ui_theme.theme.button_text.ok, lambda: True),
            (ui_theme.theme.button_text.cancel, lambda: False),
        ],
    )
    return result if result is not None else False


def askyesno(title: str, message: str, parent=None) -> bool:
    """Show a themed Yes/No dialog and return True if Yes was clicked."""
    result = _create_dialog_base(
        title=title,
        message=message,
        parent=parent,
        buttons=[
            (ui_theme.theme.button_text.yes, lambda: True),
            (ui_theme.theme.button_text.no, lambda: False),
        ],
    )
    return result if result is not None else False


def showerror(title: str, message: str, parent=None):
    """Show a themed error dialog."""
    _create_dialog_base(
        title=title,
        message=message,
        parent=parent,
        buttons=[(ui_theme.theme.button_text.ok, lambda: None)],
    )


def showinfo(title: str, message: str, parent=None):
    """Show a themed info dialog."""
    _create_dialog_base(
        title=title,
        message=message,
        parent=parent,
        buttons=[(ui_theme.theme.button_text.ok, lambda: None)],
    )


def showwarning(title: str, message: str, parent=None):
    """Show a themed warning dialog."""
    _create_dialog_base(
        title=title,
        message=message,
        parent=parent,
        buttons=[(ui_theme.theme.button_text.ok, lambda: None)],
    )
