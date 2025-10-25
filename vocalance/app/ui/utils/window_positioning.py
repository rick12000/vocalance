"""Window positioning utilities for centering dialogs on parent windows."""


def center_window_on_parent(window, parent):
    """
    Center a dialog window on its parent window.

    Uses both immediate centering and delayed reinforcement to ensure
    the dialog appears centered regardless of timing issues with
    CustomTkinter's window management.

    Args:
        window: Dialog window (CTkToplevel or tk.Toplevel)
        parent: Parent window (CTk, tk.Tk, or CTkToplevel)
    """
    if parent is None:
        return

    def do_center():
        """Perform the centering calculation and positioning."""
        try:
            # Ensure geometry is fully calculated
            window.update_idletasks()
            parent.update_idletasks()

            # Get parent absolute screen position
            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
            parent_width = parent.winfo_width()
            parent_height = parent.winfo_height()

            # Get actual rendered dialog size
            dialog_width = window.winfo_width()
            dialog_height = window.winfo_height()

            # Fallback to requested size if window hasn't been sized yet
            if dialog_width <= 1 or dialog_width == 200:
                dialog_width = window.winfo_reqwidth()
            if dialog_height <= 1 or dialog_height == 200:
                dialog_height = window.winfo_reqheight()

            # Calculate center position relative to parent
            x = parent_x + (parent_width - dialog_width) // 2
            y = parent_y + (parent_height - dialog_height) // 2

            # Set position
            window.geometry(f"+{x}+{y}")
        except Exception:
            # Silently ignore errors (window may be closing)
            pass

    # Immediate centering attempt
    do_center()

    # Reinforce centering with multiple delayed attempts to handle CustomTkinter timing
    # These ensure the dialog stays centered even if CustomTkinter repositions it
    if window.winfo_exists():
        window.after(10, do_center)
        window.after(50, do_center)
        window.after(100, do_center)
