"""
Themed dialog components that match the application's dark theme.
Provides CustomTkinter-based replacements for standard tkinter messageboxes.
"""

import customtkinter as ctk
from iris.app.ui import ui_theme
from iris.app.ui.utils.ui_icon_utils import set_window_icon_robust


def _get_dialog_font(size: int = None) -> tuple:
    """Get font configuration for dialogs using the font service"""
    if size is None:
        size = ui_theme.theme.font_sizes.medium
    font_family = ui_theme.theme.font_family.get_primary_font("regular")
    return (font_family, size)


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


def askokcancel(title: str, message: str, parent=None) -> bool:
    """Show a themed OK/Cancel dialog and return True if OK was clicked."""
    result = [False]  # Use list to allow modification in nested function
    
    # Create dialog window
    dialog = ctk.CTkToplevel(parent)
    _setup_dialog_window(dialog, title, parent)
    
    # Configure dialog grid
    dialog.grid_rowconfigure(0, weight=1)
    dialog.grid_columnconfigure(0, weight=1)
    
    # Main frame
    main_frame = ctk.CTkFrame(dialog, fg_color=ui_theme.theme.shape_colors.darkest, corner_radius=ui_theme.theme.border_radius.small)
    main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    
    # Configure main frame grid
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=0)  # Title
    main_frame.grid_rowconfigure(1, weight=0)  # Message
    main_frame.grid_rowconfigure(2, weight=1)  # Button frame (expandable)
    
    # Title label
    title_label = ctk.CTkLabel(
        main_frame,
        text=title,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light
    )
    title_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")
    
    # Message label
    message_label = ctk.CTkLabel(
        main_frame,
        text=message,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light,
        wraplength=350
    )
    message_label.grid(row=1, column=0, pady=(0, 20), sticky="ew")
    
    # Button frame
    button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    button_frame.grid(row=2, column=0, pady=(0, 20), sticky="ew")
    
    def on_ok():
        result[0] = True
        dialog.destroy()
    
    def on_cancel():
        result[0] = False
        dialog.destroy()
    
    # Configure button frame grid
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)
    
    # OK button
    ok_button = ctk.CTkButton(
        button_frame,
        text=ui_theme.theme.button_text.ok,
        font=ui_theme.theme.font_family.get_button_font(),
        fg_color=ui_theme.theme.shape_colors.darkest,
        hover_color=ui_theme.theme.shape_colors.dark,
        text_color=ui_theme.theme.text_colors.light,
        width=80,
        corner_radius=ui_theme.theme.border_radius.rounded,
        command=on_ok
    )
    ok_button.grid(row=0, column=0, padx=5, sticky="ew")
    
    # Cancel button
    cancel_button = ctk.CTkButton(
        button_frame,
        text=ui_theme.theme.button_text.cancel,
        font=ui_theme.theme.font_family.get_button_font(),
        fg_color=ui_theme.theme.shape_colors.darkest,
        hover_color=ui_theme.theme.shape_colors.dark,
        text_color=ui_theme.theme.text_colors.light,
        width=80,
        corner_radius=ui_theme.theme.border_radius.rounded,
        command=on_cancel
    )
    cancel_button.grid(row=0, column=1, padx=5, sticky="ew")
    
    # Bind keyboard shortcuts
    dialog.bind('<Return>', lambda e: on_ok())
    dialog.bind('<Escape>', lambda e: on_cancel())
    
    # Focus the dialog
    dialog.focus_force()
    
    # Wait for dialog to close
    dialog.wait_window()
    
    return result[0]


def askyesno(title: str, message: str, parent=None) -> bool:
    """Show a themed Yes/No dialog and return True if Yes was clicked."""
    result = [False]  # Use list to allow modification in nested function
    
    # Create dialog window
    dialog = ctk.CTkToplevel(parent)
    _setup_dialog_window(dialog, title, parent)
    
    # Configure dialog grid
    dialog.grid_rowconfigure(0, weight=1)
    dialog.grid_columnconfigure(0, weight=1)
    
    # Main frame
    main_frame = ctk.CTkFrame(dialog, fg_color=ui_theme.theme.shape_colors.darkest, corner_radius=ui_theme.theme.border_radius.small)
    main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    
    # Configure main frame grid
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=0)  # Title
    main_frame.grid_rowconfigure(1, weight=0)  # Message
    main_frame.grid_rowconfigure(2, weight=1)  # Button frame (expandable)
    
    # Title label
    title_label = ctk.CTkLabel(
        main_frame,
        text=title,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light
    )
    title_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")
    
    # Message label
    message_label = ctk.CTkLabel(
        main_frame,
        text=message,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light,
        wraplength=350
    )
    message_label.grid(row=1, column=0, pady=(0, 20), sticky="ew")
    
    # Button frame
    button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    button_frame.grid(row=2, column=0, pady=(0, 20), sticky="ew")
    
    def on_yes():
        result[0] = True
        dialog.destroy()
    
    def on_no():
        result[0] = False
        dialog.destroy()
    
    # Configure button frame grid
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)
    
    # Yes button
    yes_button = ctk.CTkButton(
        button_frame,
        text=ui_theme.theme.button_text.yes,
        font=ui_theme.theme.font_family.get_button_font(),
        fg_color=ui_theme.theme.shape_colors.darkest,
        hover_color=ui_theme.theme.shape_colors.dark,
        text_color=ui_theme.theme.text_colors.light,
        width=80,
        corner_radius=ui_theme.theme.border_radius.rounded,
        command=on_yes
    )
    yes_button.grid(row=0, column=0, padx=5, sticky="ew")
    
    # No button
    no_button = ctk.CTkButton(
        button_frame,
        text=ui_theme.theme.button_text.no,
        font=ui_theme.theme.font_family.get_button_font(),
        fg_color=ui_theme.theme.shape_colors.darkest,
        hover_color=ui_theme.theme.shape_colors.dark,
        text_color=ui_theme.theme.text_colors.light,
        width=80,
        corner_radius=ui_theme.theme.border_radius.rounded,
        command=on_no
    )
    no_button.grid(row=0, column=1, padx=5, sticky="ew")
    
    # Bind keyboard shortcuts
    dialog.bind('<Return>', lambda e: on_yes())
    dialog.bind('<Escape>', lambda e: on_no())
    
    # Focus the dialog
    dialog.focus_force()
    
    # Wait for dialog to close
    dialog.wait_window()
    
    return result[0]


def showerror(title: str, message: str, parent=None):
    """Show a themed error dialog."""
    # Create dialog window
    dialog = ctk.CTkToplevel(parent)
    _setup_dialog_window(dialog, title, parent)
    
    # Configure dialog grid
    dialog.grid_rowconfigure(0, weight=1)
    dialog.grid_columnconfigure(0, weight=1)
    
    # Main frame
    main_frame = ctk.CTkFrame(dialog, fg_color=ui_theme.theme.shape_colors.darkest, corner_radius=ui_theme.theme.border_radius.small)
    main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    
    # Configure main frame grid
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=0)  # Title
    main_frame.grid_rowconfigure(1, weight=0)  # Message
    main_frame.grid_rowconfigure(2, weight=1)  # OK button (expandable)
    
    # Title label with error styling
    title_label = ctk.CTkLabel(
        main_frame,
        text=title,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light
    )
    title_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")
    
    # Message label
    message_label = ctk.CTkLabel(
        main_frame,
        text=message,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light,
        wraplength=350
    )
    message_label.grid(row=1, column=0, pady=(0, 20), sticky="ew")
    
    # OK button
    ok_button = ctk.CTkButton(
        main_frame,
        text=ui_theme.theme.button_text.ok,
        font=ui_theme.theme.font_family.get_button_font(),
        fg_color=ui_theme.theme.shape_colors.darkest,
        hover_color=ui_theme.theme.shape_colors.dark,
        text_color=ui_theme.theme.text_colors.light,
        width=80,
        corner_radius=ui_theme.theme.border_radius.rounded,
        command=dialog.destroy
    )
    ok_button.grid(row=2, column=0, pady=(0, 20))
    
    # Bind keyboard shortcuts
    dialog.bind('<Return>', lambda e: dialog.destroy())
    dialog.bind('<Escape>', lambda e: dialog.destroy())
    
    # Focus the dialog
    dialog.focus_force()
    
    # Wait for dialog to close
    dialog.wait_window()


def showinfo(title: str, message: str, parent=None):
    """Show a themed info dialog."""
    # Create dialog window
    dialog = ctk.CTkToplevel(parent)
    _setup_dialog_window(dialog, title, parent)
    
    # Configure dialog grid
    dialog.grid_rowconfigure(0, weight=1)
    dialog.grid_columnconfigure(0, weight=1)
    
    # Main frame
    main_frame = ctk.CTkFrame(dialog, fg_color=ui_theme.theme.shape_colors.darkest, corner_radius=ui_theme.theme.border_radius.small)
    main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    
    # Configure main frame grid
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=0)  # Title
    main_frame.grid_rowconfigure(1, weight=0)  # Message
    main_frame.grid_rowconfigure(2, weight=1)  # OK button (expandable)
    
    # Title label with success styling
    title_label = ctk.CTkLabel(
        main_frame,
        text=title,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light
    )
    title_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")
    
    # Message label
    message_label = ctk.CTkLabel(
        main_frame,
        text=message,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light,
        wraplength=350
    )
    message_label.grid(row=1, column=0, pady=(0, 20), sticky="ew")
    
    # OK button
    ok_button = ctk.CTkButton(
        main_frame,
        text=ui_theme.theme.button_text.ok,
        font=ui_theme.theme.font_family.get_button_font(),
        fg_color=ui_theme.theme.shape_colors.darkest,
        hover_color=ui_theme.theme.shape_colors.dark,
        text_color=ui_theme.theme.text_colors.light,
        width=80,
        corner_radius=ui_theme.theme.border_radius.rounded,
        command=dialog.destroy
    )
    ok_button.grid(row=2, column=0, pady=(0, 20))
    
    # Bind keyboard shortcuts
    dialog.bind('<Return>', lambda e: dialog.destroy())
    dialog.bind('<Escape>', lambda e: dialog.destroy())
    
    # Focus the dialog
    dialog.focus_force()
    
    # Wait for dialog to close
    dialog.wait_window()


def showwarning(title: str, message: str, parent=None):
    """Show a themed warning dialog."""
    # Create dialog window
    dialog = ctk.CTkToplevel(parent)
    _setup_dialog_window(dialog, title, parent)
    
    # Configure dialog grid
    dialog.grid_rowconfigure(0, weight=1)
    dialog.grid_columnconfigure(0, weight=1)
    
    # Main frame
    main_frame = ctk.CTkFrame(dialog, fg_color=ui_theme.theme.shape_colors.darkest, corner_radius=ui_theme.theme.border_radius.small)
    main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    
    # Configure main frame grid
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=0)  # Title
    main_frame.grid_rowconfigure(1, weight=0)  # Message
    main_frame.grid_rowconfigure(2, weight=1)  # OK button (expandable)
    
    # Title label with warning styling
    title_label = ctk.CTkLabel(
        main_frame,
        text=title,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light
    )
    title_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")
    
    # Message label
    message_label = ctk.CTkLabel(
        main_frame,
        text=message,
        font=(ui_theme.theme.font_family.primary, ui_theme.theme.font_sizes.medium),
        text_color=ui_theme.theme.text_colors.light,
        wraplength=350
    )
    message_label.grid(row=1, column=0, pady=(0, 20), sticky="ew")
    
    # OK button
    ok_button = ctk.CTkButton(
        main_frame,
        text=ui_theme.theme.button_text.ok,
        font=ui_theme.theme.font_family.get_button_font(),
        fg_color=ui_theme.theme.shape_colors.darkest,
        hover_color=ui_theme.theme.shape_colors.dark,
        text_color=ui_theme.theme.text_colors.light,
        width=80,
        corner_radius=ui_theme.theme.border_radius.rounded,
        command=dialog.destroy
    )
    ok_button.grid(row=2, column=0, pady=(0, 20))
    
    # Bind keyboard shortcuts
    dialog.bind('<Return>', lambda e: dialog.destroy())
    dialog.bind('<Escape>', lambda e: dialog.destroy())
    
    # Focus the dialog
    dialog.focus_force()
    
    # Wait for dialog to close
    dialog.wait_window() 