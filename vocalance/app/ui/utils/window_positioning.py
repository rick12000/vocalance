import logging


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
            window.update_idletasks()
            parent.update_idletasks()

            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
            parent_width = parent.winfo_width()
            parent_height = parent.winfo_height()

            dialog_width = window.winfo_width()
            dialog_height = window.winfo_height()

            if dialog_width <= 1 or dialog_width == 200:
                dialog_width = window.winfo_reqwidth()
            if dialog_height <= 1 or dialog_height == 200:
                dialog_height = window.winfo_reqheight()

            x = parent_x + (parent_width - dialog_width) // 2
            y = parent_y + (parent_height - dialog_height) // 2

            window.geometry(f"+{x}+{y}")
        except Exception:
            pass

    do_center()

    if window.winfo_exists():
        window.after(10, do_center)
        window.after(50, do_center)
        window.after(100, do_center)


def position_toplevel_window(window, target_x=None, target_y=None, width=None, height=None, position_type="center", parent=None):
    """
    Unified window positioning for all CTkToplevel windows.

    Handles:
    - Accurate size detection (winfo_reqwidth fallback)
    - Delayed reinforcement (multiple after() calls)
    - Different positioning strategies (center, bottom_left, center_left, custom)
    - Parent-relative or screen-relative positioning

    Args:
        window: CTkToplevel window to position
        target_x: Explicit x coordinate (or None for calculated)
        target_y: Explicit y coordinate (or None for calculated)
        width: Desired width (or None for requested width)
        height: Desired height (or None for requested height)
        position_type: "center", "bottom_left", "center_left", "custom"
        parent: Parent window for relative positioning (or None for screen)
    """

    def do_position():
        try:
            window.update_idletasks()

            # Get accurate window dimensions
            actual_width = window.winfo_width()
            actual_height = window.winfo_height()

            # Fallback to requested size if window not realized
            if actual_width <= 1 or actual_width == 200:
                actual_width = window.winfo_reqwidth() if width is None else width
            if actual_height <= 1 or actual_height == 200:
                actual_height = window.winfo_reqheight() if height is None else height

            # Get screen or parent dimensions
            if parent:
                parent.update_idletasks()
                ref_x = parent.winfo_rootx()
                ref_y = parent.winfo_rooty()
                ref_width = parent.winfo_width()
                ref_height = parent.winfo_height()
            else:
                # Use parent window for reliable screen dimensions
                screen_window = window.master
                ref_x = 0
                ref_y = 0
                ref_width = screen_window.winfo_screenwidth()
                ref_height = screen_window.winfo_screenheight()

            # Calculate position based on strategy
            if target_x is not None and target_y is not None:
                x, y = target_x, target_y
            elif position_type == "center":
                x = ref_x + (ref_width - actual_width) // 2
                y = ref_y + (ref_height - actual_height) // 2
            elif position_type == "bottom_left":
                x = 80  # WINDOW_MARGIN_X
                y = int(ref_height * 0.85) - actual_height
                y = max(0, y)
            elif position_type == "center_left":
                x = 80  # WINDOW_MARGIN_X
                y = (ref_height - actual_height) // 2
                y = max(0, min(y, ref_height - actual_height))
            else:
                # Default to center
                x = ref_x + (ref_width - actual_width) // 2
                y = ref_y + (ref_height - actual_height) // 2

            # Apply geometry
            if width and height:
                window.geometry(f"{width}x{height}+{x}+{y}")
            else:
                window.geometry(f"+{x}+{y}")

            logging.debug(f"Positioned window: type={position_type}, size={actual_width}x{actual_height}, pos=({x},{y})")

        except Exception as e:
            logging.warning(f"Window positioning failed: {e}")

    # Immediate + delayed reinforcement (proven pattern)
    do_position()
    if window.winfo_exists():
        window.after(10, do_position)
        window.after(50, do_position)
        window.after(100, do_position)
