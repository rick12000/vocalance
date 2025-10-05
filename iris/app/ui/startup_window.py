"""
Streamlined Startup Window for Iris Application
"""

import customtkinter as ctk
import tkinter as tk
from typing import Optional, Union
import logging
import threading

from iris.app.ui import ui_theme
from iris.app.ui.utils.ui_thread_utils import schedule_ui_update_immediate, is_main_thread
from iris.app.ui.utils.ui_icon_utils import set_window_icon_with_parent_inheritance
from iris.app.ui.utils.logo_service import logo_service


class StartupWindow:
    """Thread-safe startup window with progress tracking"""
    
    def __init__(self, logger: logging.Logger, main_root: Union[tk.Tk, ctk.CTk]):
        self.logger = logger
        self.main_root = main_root
        self.window: Optional[ctk.CTkToplevel] = None
        self.progress_bar: Optional[ctk.CTkProgressBar] = None
        self.status_label: Optional[ctk.CTkLabel] = None
        self.is_closed = False
        
    def show(self) -> None:
        """Show the startup window in a thread-safe manner."""
        if not is_main_thread():
            schedule_ui_update_immediate(self._show_impl)
        else:
            self._show_impl()
    
    def _show_impl(self) -> None:
        """Internal implementation of show that must run in main thread."""
        if self.window is not None:
            return
        
        try:
            self.main_root.update_idletasks()
            
            # Create window
            self.window = ctk.CTkToplevel(self.main_root)
            self.window.title("Iris")
            self.window.geometry(f"{ui_theme.theme.dimensions.startup_width}x{ui_theme.theme.dimensions.startup_height}")
            self.window.resizable(False, False)
            self.window.attributes("-topmost", True)
            self.window.configure(fg_color=ui_theme.theme.shape_colors.darkest)
            
            # Center window
            self._center_window()
            
            # Set icon
            from iris.app.ui.utils.ui_icon_utils import track_window_for_icon_management
            track_window_for_icon_management(self.window)
            set_window_icon_with_parent_inheritance(self.window, self.main_root)
            
            # Prevent closing during startup
            self.window.protocol("WM_DELETE_WINDOW", lambda: None)
            
            # Create UI components
            self._create_ui()
            
            # Show window
            self.window.update_idletasks()
            self.window.lift()
            self.window.focus_force()
            
            self.logger.info("Startup window displayed")
            
        except Exception as e:
            self.logger.error(f"Error creating startup window: {e}", exc_info=True)
    
    def _center_window(self) -> None:
        """Center the window on screen"""
        try:
            self.window.update_idletasks()
            width = ui_theme.theme.dimensions.startup_width
            height = ui_theme.theme.dimensions.startup_height
            x = (self.window.winfo_screenwidth() // 2) - (width // 2)
            y = (self.window.winfo_screenheight() // 2) - (height // 2)
            self.window.geometry(f"{width}x{height}+{x}+{y}")
        except Exception as e:
            self.logger.warning(f"Could not center window: {e}")
    
    def _create_ui(self) -> None:
        """Create the UI components"""
        # Configure window grid
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Main container
        main_frame = ctk.CTkFrame(
            self.window,
            fg_color=ui_theme.theme.shape_colors.darkest,
            corner_radius=0
        )
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Configure main frame grid
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=0)  # Logo
        main_frame.grid_rowconfigure(1, weight=0)  # Progress bar
        main_frame.grid_rowconfigure(2, weight=1)  # Status label
        
        # Logo with automatic image/text fallback
        self.logo_label = logo_service.create_logo_widget(
            main_frame,
            max_size=ui_theme.theme.dimensions.startup_logo_size,
            context="startup",
            text_fallback="IRIS"
        )
        self.logo_label.grid(row=0, column=0, pady=(10, 20), sticky="ew")
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            main_frame,
            width=ui_theme.theme.dimensions.progress_bar_width,
            height=ui_theme.theme.dimensions.progress_bar_height,
            progress_color=ui_theme.theme.shape_colors.lightest,
            fg_color=ui_theme.theme.shape_colors.light
        )
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, pady=(0, 15), sticky="ew")
        
        # Status label
        font_family = ui_theme.theme.font_family.get_primary_font("regular")
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Starting up...",
            font=(font_family, ui_theme.theme.font_sizes.small),
            text_color=ui_theme.theme.text_colors.dark,
            justify="center"
        )
        self.status_label.grid(row=2, column=0, pady=(0, 10), padx=10, sticky="new")
    

    
    def update_progress(self, progress: float, status: str) -> None:
        """Update progress in a thread-safe manner."""
        schedule_ui_update_immediate(self._update_progress_impl, progress, status)
    
    def _update_progress_impl(self, progress: float, status: str) -> None:
        """Internal implementation of progress update."""
        if self.is_closed or not self.window:
            return
        
        try:
            progress = max(0.0, min(1.0, progress))
            
            if self.progress_bar:
                self.progress_bar.set(progress)
            
            if self.status_label and status:
                self.status_label.configure(text=status)
            
            if self.window:
                self.window.update_idletasks()
                
        except Exception as e:
            self.logger.error(f"Error updating startup progress: {e}")
    
    def close(self) -> None:
        """Close the startup window in a thread-safe manner."""
        if not is_main_thread():
            schedule_ui_update_immediate(self._close_impl)
        else:
            self._close_impl()
    
    def _close_impl(self) -> None:
        """Internal implementation of close that must run in main thread."""
        if self.is_closed or not self.window:
            return
        
        try:
            self.is_closed = True
            self.window.destroy()
            self.window = None
            self.logger.info("Startup window closed")
        except Exception as e:
            self.logger.error(f"Error closing startup window: {e}")
    
    def is_visible(self) -> bool:
        """Check if the startup window is visible."""
        return self.window is not None and not self.is_closed


class StartupProgressTracker:
    """Startup progress tracker with sub-step support"""
    
    def __init__(self, startup_window: StartupWindow, total_steps: int):
        self.startup_window = startup_window
        self.total_steps = total_steps
        self.current_step = 0
        self.current_step_name = ""
        self.sub_step_progress = 0.0
        
    def start_step(self, step_name: str) -> None:
        """Start a new step"""
        self.current_step += 1
        self.current_step_name = step_name
        self.sub_step_progress = 0.0
        self._update_display(step_name)
        
    def update_sub_step(self, sub_step_name: str, progress: float = 0.5) -> None:
        """Update sub-step progress within current step"""
        self.sub_step_progress = max(0.0, min(1.0, progress))
        self._update_display(sub_step_name)
        
    def complete_step(self, step_name: str = "") -> None:
        """Complete the current step"""
        self.sub_step_progress = 1.0
        self._update_display(step_name or f"{self.current_step_name} completed")
        
    def _update_display(self, status: str) -> None:
        """Update the startup window display"""
        if self.total_steps > 0:
            # Base progress from completed steps
            base_progress = (self.current_step - 1) / self.total_steps
            # Add current step progress
            current_step_contribution = self.sub_step_progress / self.total_steps
            progress = base_progress + current_step_contribution
        else:
            progress = 0.0
            
        progress = min(1.0, progress)
        self.startup_window.update_progress(progress, status)
        
    def finish(self) -> None:
        """Finish progress tracking and close window"""
        self.startup_window.update_progress(1.0, "Ready!")
        
        def delayed_close():
            import time
            time.sleep(0.8)
            self.startup_window.close()
            
        import threading
        threading.Thread(target=delayed_close, daemon=True).start()