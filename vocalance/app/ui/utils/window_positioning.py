"""Window positioning utilities for centering dialogs on parent windows."""


def center_window_on_parent(window, parent):
    """
    Center a dialog window on its parent window using event-driven approach.

    Uses the <Map> event to ensure centering happens after the window is fully
    rendered and displayed. This is more robust than timer-based approaches.

    Args:
        window: Dialog window (CTkToplevel or tk.Toplevel)
        parent: Parent window (CTk, tk.Tk, or CTkToplevel)

    Note:
        The <Map> event fires when a window becomes visible with all its content
        rendered, making it the ideal time to calculate and set the center position.
    """
    if parent is None:
        return

    def do_center(event=None):
        # Unbind immediately to prevent multiple calls
        try:
            window.unbind("<Map>", bind_id)
        except Exception:
            pass

        # Ensure geometry is fully calculated
        window.update_idletasks()

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

        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        # Set position
        window.geometry(f"+{x}+{y}")

    # Bind to Map event (fires when window is mapped/shown)
    bind_id = window.bind("<Map>", do_center, add="+")
