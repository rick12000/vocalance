import logging
import tkinter as tk
from typing import Any, Callable, Optional

import customtkinter as ctk

from vocalance.app.ui.views.components import themed_dialogs as messagebox
from vocalance.app.ui.views.components.view_config import view_config


class ViewHelper(ctk.CTkFrame):
    """Base class for all view components with common functionality"""

    def __init__(self, parent, controller=None, root_window=None):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.root_window = root_window
        self.logger = logging.getLogger(self.__class__.__name__)

        if controller:
            controller.set_view_callback(self)

    def show_error(self, title: str, message: str) -> None:
        """Show standardized error dialog"""
        messagebox.showerror(message, parent=self.root_window)

    def show_info(self, title: str, message: str) -> None:
        """Show standardized info dialog"""
        messagebox.showinfo(message, parent=self.root_window)

    def show_confirmation(self, title: str, message: str) -> bool:
        """Show standardized confirmation dialog"""
        return messagebox.askyesno(message, parent=self.root_window)

    def show_delete_all_confirmation(self, items_name: str) -> bool:
        """Show standardized delete all confirmation dialog"""
        message = view_config.messages.delete_all_confirmation_template.format(items=items_name)
        return self.show_confirmation(view_config.messages.confirm_delete_title, message)

    def clear_form_fields(self, *fields) -> None:
        """Clear multiple form fields at once"""
        for field in fields:
            if hasattr(field, "delete"):
                if isinstance(field, ctk.CTkEntry):
                    field.delete(0, tk.END)
                elif isinstance(field, ctk.CTkTextbox):
                    field.delete("1.0", tk.END)

    def setup_main_layout(self) -> None:
        """Setup standard main layout grid configuration"""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def safe_widget_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Safely execute widget operations with error handling"""
        try:
            return operation(*args, **kwargs)
        except (tk.TclError, AttributeError) as e:
            self.logger.debug(f"Widget operation failed (widget may be destroyed): {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in widget operation: {e}")
            return None

    def schedule_delayed_action(self, action: Callable, delay_ms: Optional[int] = None) -> None:
        """Schedule an action to run after a delay"""
        if delay_ms is None:
            delay_ms = view_config.timings.refresh_list_delay_ms
        self.after(delay_ms, action)

    # Standard callback methods that can be overridden
    def on_status_update(self, message: str, is_error: bool = False) -> None:
        """Handle status updates - override in subclasses"""
        if is_error:
            self.logger.error(f"Status error: {message}")
        else:
            self.logger.info(f"Status: {message}")

    def on_data_updated(self, data: Any) -> None:
        """Handle data updates - override in subclasses"""

    def on_validation_error(self, title: str, message: str) -> None:
        """Handle validation errors - override in subclasses"""
        self.show_error(title or view_config.messages.validation_error_title, message)

    def on_save_success(self, message: str) -> None:
        """Handle save success - override in subclasses"""
        self.show_info(view_config.messages.save_success_title, message)

    def on_save_error(self, message: str) -> None:
        """Handle save errors - override in subclasses"""
        self.show_error(view_config.messages.save_error_title, message)
