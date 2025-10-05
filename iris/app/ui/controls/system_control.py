import logging
import tkinter as tk
import asyncio
from iris.app.ui.controls.base_control import BaseController
from iris.app.events.ui_events import GetMainWindowHandleRequest, GetMainWindowHandleResponse
from iris.app.events.core_events import CommandExecutedStatusEvent


class SystemController(BaseController):
    """Controller for general system functionality."""
    
    def __init__(self, event_bus, tk_root: tk.Tk, event_loop, logger):
        super().__init__(event_bus, event_loop, logger, "SystemController")
        self.tk_root = tk_root
        
        self.subscribe_to_events([
            (GetMainWindowHandleRequest, self._handle_get_main_window_handle_request),
            (CommandExecutedStatusEvent, self._handle_command_executed_status),
        ])

    async def _handle_get_main_window_handle_request(self, event_data) -> None:
        """Handle main window handle request."""
        hwnd = None
        error_message = None
        
        if self.tk_root:
            try:
                hwnd = self.tk_root.winfo_id()
            except Exception as e:
                error_message = f"Error getting window handle: {e}"
                self.logger.error(error_message)
        else:
            error_message = "tk_root is not available"
            
        response = GetMainWindowHandleResponse(hwnd=hwnd, error_message=error_message)
        self.publish_event(response)

    async def _handle_command_executed_status(self, event_data) -> None:
        """Handle command execution status event."""
        message = event_data.message
        if not message:
            command_type = event_data.command.get('command_type', 'Unknown Command')
            status = "executed successfully" if event_data.success else "failed"
            message = f"Command {command_type} {status}."
        
        self.notify_status(message, not event_data.success)

    def cleanup(self) -> None:
        """Clean up resources when controller is destroyed."""
        super().cleanup() 