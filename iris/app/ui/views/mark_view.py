# Mark View - UI-specific mark visualization implementation
import tkinter as tk
import logging
from typing import Dict, Tuple, Optional, Callable, Any, List
from iris.app.events.mark_events import MarkData
from iris.app.ui.views.components.view_config import view_config


class MarkView:
    """Thread-safe mark visualization view with simplified event handling."""
    
    def __init__(self, root: tk.Tk):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root = root

        # UI components
        self.overlay_window: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None
        self._is_active = False

        # Mark data
        self.marks: Dict[str, Tuple[int, int]] = {}

        # Controller callback reference
        self.controller_callback = None

        self.logger.info("MarkView initialized")

    def set_controller_callback(self, callback):
        """Set the controller callback for handling mark interactions."""
        self.controller_callback = callback

    def update_marks(self, marks_list: List[MarkData]) -> None:
        """Update the marks data from a list of MarkData objects."""
        self.marks = {mark.name: (mark.x, mark.y) for mark in marks_list}
        self.logger.debug(f"MarkView: Updated marks data with {len(self.marks)} marks")
        
        # Redraw if overlay is active
        if self._is_active and self.canvas:
            self._draw_marks()

    def update_marks_dict(self, marks: Dict[str, Tuple[int, int]]) -> None:
        """Update the marks data from a dictionary."""
        self.marks = marks.copy()
        self.logger.debug(f"MarkView: Updated marks data with {len(self.marks)} marks")
        
        # Redraw if overlay is active
        if self._is_active and self.canvas:
            self._draw_marks()

    def _draw_marks(self):
        """Draw marks on the canvas."""
        if not self.canvas:
            self.logger.error("Cannot draw marks: canvas is not available.")
            return
        
        self.canvas.delete("all")  # Clear existing marks
        
        mark_font = view_config.marks.mark_font
        
        for label, (x, y) in self.marks.items():
            radius = 5
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    fill=view_config.marks.themed_mark_fill_color, outline=view_config.marks.themed_mark_outline_color)
            # Draw the label near the mark
            self.canvas.create_text(x + 10, y - 10, text=label, fill="white", anchor="w",
                                    font=mark_font)
        self.logger.debug(f"Drew {len(self.marks)} marks on the overlay.")

    def _close_window_event(self, event=None):
        """Handle window close event."""
        self.hide()
        
    def is_active(self) -> bool:
        """Return whether the overlay is currently active and visible."""
        return self._is_active and self.overlay_window is not None
          
    def show(self):
        """Thread-safe show method that creates and displays the overlay window."""
        if not self.root:
            self.logger.error("Cannot show overlay: root is not available.")
            return
            
        if self.overlay_window is not None:
            self.logger.warning("MarkView show() called but window already exists. Aborting.")
            return

        try:
            self.logger.info("Showing MarkView overlay.")
            
            # Use Toplevel instead of creating a new Tk root
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.title("Active Marks")
            self.overlay_window.attributes("-topmost", True)  # Keep window on top
            self.overlay_window.attributes("-alpha", 0.8)  # Slight transparency
            self.overlay_window.geometry("+0+0") # Default position, can be adjusted
            self.overlay_window.attributes("-fullscreen", True) # Or use specific geometry
            self.overlay_window.configure(bg='grey15') # Background for visibility of labels

            # Remove window decorations (title bar, borders) for a true overlay feel
            self.overlay_window.overrideredirect(True) # Borderless window

            # Attempt to prevent taskbar icon on Windows
            try:
                self.overlay_window.attributes('-toolwindow', True)
            except tk.TclError:
                self.logger.warning("Failed to set -toolwindow attribute. May not be supported on this platform/Tk version.")

            # Configure overlay window grid
            self.overlay_window.grid_rowconfigure(0, weight=1)
            self.overlay_window.grid_columnconfigure(0, weight=1)
            
            self.canvas = tk.Canvas(self.overlay_window, bg=self.overlay_window.cget('bg'), highlightthickness=0)
            self.canvas.grid(row=0, column=0, sticky="nsew")

            self._draw_marks()

            # Close on Escape key
            self.overlay_window.bind("<Escape>", self._close_window_event)

            # Set up proper window close handling
            self.overlay_window.protocol("WM_DELETE_WINDOW", self._close_window_event)

            # Show the overlay window and give it focus
            self.overlay_window.deiconify()
            self.overlay_window.lift()
            self.overlay_window.focus_force()

            self._is_active = True

            # Notify controller of successful show
            if self.controller_callback:
                self.controller_callback.on_mark_visualization_shown()
            
            self.logger.debug("MarkView.show() completed successfully.")
            
        except tk.TclError as e:
            self.logger.error(f"TclError during overlay show: {e}")
            self._cleanup_failed_show()
        except Exception as e:
            self.logger.error(f"Unexpected error during overlay show: {e}")
            self._cleanup_failed_show()

    def _cleanup_failed_show(self):
        """Clean up resources after a failed show attempt."""
        try:
            if self.overlay_window and self.overlay_window.winfo_exists():
                self.overlay_window.destroy()
        except tk.TclError:
            pass  # Window may already be destroyed
        finally:
            self.overlay_window = None
            self.canvas = None
            self._is_active = False
            
            # Notify controller of failed show
            if self.controller_callback:
                self.controller_callback.on_mark_visualization_failed("Failed to create overlay window")
    
    def hide(self):
        """Thread-safe hide method that properly cleans up resources."""
        if not self.overlay_window:
            self.logger.debug("MarkView.hide() called but window is already None.")
            return
            
        self.logger.info("Hiding MarkView overlay.")
        self._is_active = False

        # Clean up references before destroying
        self.canvas = None
        
        # Destroy the window safely
        try:
            if self.overlay_window.winfo_exists():
                self.overlay_window.destroy()
                self.logger.debug("Window destroyed successfully.")
        except tk.TclError as e:
            self.logger.warning(f"TclError during window destruction: {e}")
        finally:
            self.overlay_window = None
            
        # Notify controller of hide
        if self.controller_callback:
            self.controller_callback.on_mark_visualization_hidden()
    
    def withdraw(self):
        """Thread-safe withdraw method that hides the overlay window without destroying it."""
        if not self.overlay_window:
            self.logger.warning("MarkView withdraw() called but window doesn't exist.")
            return
            
        self.logger.info("Withdrawing MarkView overlay.")
        self._is_active = False

        try:
            self.overlay_window.withdraw()
            self.logger.debug("MarkView overlay withdrawn successfully.")
        except tk.TclError as e:
            self.logger.error(f"TclError during overlay withdraw: {e}")

    def cleanup(self) -> None:
        """Clean up resources when mark view is destroyed."""
        self.hide()
        self.logger.info("MarkView cleaned up") 