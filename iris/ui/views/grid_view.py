# Grid View - UI-specific grid overlay implementation
import tkinter as tk
import math
import logging
import time
import uuid
import asyncio
from typing import Optional, Dict, Any, Tuple, List, Callable

import pyautogui
from iris.services.grid.click_tracker_service import prioritize_grid_rects
from iris.events.core_events import PerformMouseClickEventData
from iris.events.grid_events import RequestClickCountsForGridEventData, ClickCountsForGridEventData
from iris.event_bus import EventBus
from iris.utils.event_utils import ThreadSafeEventPublisher, EventSubscriptionManager
from iris.ui.views.components.view_config import view_config


class GridView:
    """Thread-safe grid overlay view with simplified event handling."""
    
    # Constants
    DEFAULT_NUM_RECTS = 500

    def __init__(self, root: tk.Tk, event_bus: EventBus, default_num_rects: Optional[int] = None, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        # Initialize logging and core attributes
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root = root
        self.event_bus = event_bus
        self.event_loop = event_loop or asyncio.get_event_loop()
        self._event_loop = None
        self._is_active = False
        
        self.default_num_rects = default_num_rects or self.DEFAULT_NUM_RECTS
        
        # Initialize UI components
        self.overlay_window: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None
        self.current_num_rects_displayed: Optional[int] = None
        self.ui_to_rect_data_map: Dict[int, Dict[str, Any]] = {}
        
        # Event handling - use provided event_loop if available
        self.event_publisher = ThreadSafeEventPublisher(event_bus, self.event_loop)
        self.subscription_manager = EventSubscriptionManager(event_bus, "GridView")
        self._pending_renders: Dict[str, Dict[str, Any]] = {}
        
        # Screen dimensions
        self._update_screen_dimensions()
        
        # Setup subscriptions
        self.setup_event_subscriptions()
        
        # Controller callback reference
        self.controller_callback = None
        
        self.logger.info(f"GridView initialized. Screen: {self.screen_width}x{self.screen_height}")

    def set_controller_callback(self, callback):
        """Set the controller callback for handling grid interactions."""
        self.controller_callback = callback

    def _get_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the current event loop or find running loop."""
        if self._event_loop and not self._event_loop.is_closed():
            return self._event_loop
        
        try:
            loop = asyncio.get_running_loop()
            self._event_loop = loop
            return loop
        except RuntimeError:
            self.logger.warning("No running event loop found")
            return None
    
    def _schedule_in_tk_thread(self, callback: Callable, *args) -> None:
        """Schedule a callback to run in the Tkinter main thread."""
        if self.root:
            try:
                self.root.after(0, callback, *args)
            except (RuntimeError, tk.TclError):
                pass  # Silently ignore scheduling errors

    def _update_screen_dimensions(self) -> None:
        """Update screen dimensions from root window."""
        if self.root:
            self.screen_width = self.root.winfo_screenwidth()
            self.screen_height = self.root.winfo_screenheight()
        else:
            self.screen_width = 1920  # Fallback
            self.screen_height = 1080

    def setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for grid view."""
        self.subscription_manager.subscribe(
            ClickCountsForGridEventData, 
            self._handle_click_counts_received
        )

    def cleanup(self) -> None:
        """Clean up resources when grid view is destroyed."""
        self.hide()
        self.subscription_manager.unsubscribe_all()
        self._pending_renders.clear()
        if self.overlay_window:
            try:
                self.overlay_window.destroy()
            except tk.TclError:
                pass
        self.overlay_window = None
        self._is_active = False

    async def _handle_click_counts_received(self, event_data: ClickCountsForGridEventData) -> None:
        """Handle received click counts and draw the grid."""
        self.logger.info(f"[GridView] _handle_click_counts_received called with request_id={event_data.request_id}")
        self.logger.debug(f"[GridView] Received {len(event_data.processed_rects_with_clicks)} processed rectangles")
        
        request_id = event_data.request_id
        pending_context = self._pending_renders.pop(request_id, None)
        
        if not pending_context:
            self.logger.warning(f"[GridView] No pending render context for request_id: {request_id}")
            return
        
        self.logger.info(f"[GridView] Found pending context for request_id: {request_id}")
        
        # Process and prioritize rectangles
        self.logger.info(f"[GridView] Prioritizing {len(event_data.processed_rects_with_clicks)} rectangles")
        weighted_rects = prioritize_grid_rects(event_data.processed_rects_with_clicks)
        self.logger.info(f"[GridView] Prioritization returned {len(weighted_rects)} rectangles")
        
        # Schedule drawing in Tk thread
        self.logger.info(f"[GridView] Scheduling grid drawing with cell_w={pending_context['cell_w']}, cell_h={pending_context['cell_h']}")
        self._schedule_in_tk_thread(
            self._draw_grid_elements,
            weighted_rects,
            pending_context['cell_w'],
            pending_context['cell_h']
        )

    def _draw_grid_elements(self, weighted_rects: List[Dict[str, Any]], cell_w: float, cell_h: float) -> None:
        self.logger.info(f"[GridView] Drawing {len(weighted_rects)} grid elements with cell_w={cell_w}, cell_h={cell_h}")
        if not self._validate_ui_state():
            self.logger.error("[GridView] UI state invalid for drawing")
            return

        self.ui_to_rect_data_map.clear()
        self.canvas.delete("all")

        # Debug canvas state
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.logger.debug(f"[GridView] Canvas dimensions: {canvas_width}x{canvas_height}")
        
        font_tuple = view_config.grid.get_font_tuple(cell_h)
        self.logger.debug(f"[GridView] Using font: {font_tuple}")

        # Get colors with transparency applied
        fill_color = view_config.grid.get_fill_color_with_alpha()
        outline_color = view_config.grid.get_outline_color_with_alpha()
        text_color = view_config.grid.get_text_color_with_alpha()

        for ui_number, weighted_rect_info in enumerate(weighted_rects, 1):
            rect_data = weighted_rect_info['data']
            if ui_number <= 5:  # Only log first 5 for brevity
                self.logger.debug(f"[GridView] Drawing rect #{ui_number}: {rect_data}")
            self.ui_to_rect_data_map[ui_number] = rect_data

            x0, y0 = rect_data['x'], rect_data['y']
            x1, y1 = x0 + rect_data['w'], y0 + rect_data['h']
            
            # Create rectangle
            rect_kwargs = {
                'outline': outline_color,
                'width': 2  # Make outline more visible
            }
            # Only add fill if fill_color is not empty (for transparency)
            if fill_color:
                rect_kwargs['fill'] = fill_color
            
            rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, 
                **rect_kwargs
            )
            
            # Create text
            text_id = self.canvas.create_text(
                rect_data['center_x'], rect_data['center_y'], 
                text=str(ui_number), 
                fill=text_color, 
                font=font_tuple
            )
            
            if ui_number <= 3:  # Log first few items for debugging
                self.logger.debug(f"[GridView] Created rect {rect_id} and text {text_id} for cell {ui_number}")
        
        # Force canvas update before making window visible
        self.canvas.update_idletasks()
        self.canvas.update()
        
        self._ensure_window_visible()
        self._is_active = True
        self.logger.info(f"[GridView] Grid displayed with {len(self.ui_to_rect_data_map)} rectangles")

    def _validate_ui_state(self) -> bool:
        """Validate that UI components are ready for drawing."""
        if not self.overlay_window or not self.overlay_window.winfo_exists():
            self.logger.error("Overlay window not available for drawing")
            return False
        if not self.canvas:
            self.logger.error("Canvas not available for drawing")
            return False
        return True

    def _ensure_window_visible(self) -> None:
        """Ensure the overlay window is visible."""
        if self.overlay_window:
            if not self.overlay_window.winfo_viewable():
                self.overlay_window.deiconify()
            # Ensure window is on top and visible
            self.overlay_window.lift()
            self.overlay_window.update()
            self.logger.debug("[GridView] Overlay window made visible and brought to front")

    def _calculate_grid_layout(self, num_rects_requested: int) -> Tuple[List[Dict[str, Any]], float, float]:
        """Calculate grid layout that creates approximately the requested number of cells while filling the screen."""
        if not self.root:
            raise RuntimeError("Tk root not available for grid layout calculation")

        if not self.screen_width or not self.screen_height:
            self._update_screen_dimensions()
            if not self.screen_width or not self.screen_height:
                self.logger.error("Cannot get screen dimensions")
                return [], 0, 0
        
        if num_rects_requested <= 0:
            return [], 0, 0

        screen_aspect_ratio = self.screen_width / self.screen_height
        
        # Calculate cell size based on dividing screen area by requested cells
        total_screen_area = self.screen_width * self.screen_height
        target_cell_area = total_screen_area / num_rects_requested
        
        # Calculate cell dimensions that maintain screen aspect ratio
        # If cell has same aspect ratio as screen: cell_w / cell_h = screen_aspect_ratio
        # And cell_w * cell_h = target_cell_area
        # So: cell_h = sqrt(target_cell_area / screen_aspect_ratio)
        # And: cell_w = cell_h * screen_aspect_ratio
        
        cell_h = math.sqrt(target_cell_area / screen_aspect_ratio)
        cell_w = cell_h * screen_aspect_ratio
        
        # Calculate how many cells fit in each dimension
        num_cols = math.floor(self.screen_width / cell_w)
        num_rows = math.floor(self.screen_height / cell_h)
        
        # Ensure we have at least 1 cell in each dimension
        num_cols = max(1, num_cols)
        num_rows = max(1, num_rows)
        
        # Recalculate actual cell dimensions to fill the screen exactly
        rect_w = self.screen_width / num_cols
        rect_h = self.screen_height / num_rows
        
        # Calculate total cells that will be created
        actual_cells_to_create = num_cols * num_rows

        rect_definitions = []
        for i in range(actual_cells_to_create):
            row_idx = i // num_cols
            col_idx = i % num_cols

            if row_idx >= num_rows:
                break

            x = col_idx * rect_w
            y = row_idx * rect_h
            center_x = x + rect_w / 2
            center_y = y + rect_h / 2
            
            rect_definitions.append({
                'id': i, 'x': x, 'y': y, 'w': rect_w, 'h': rect_h,
                'center_x': center_x, 'center_y': center_y,
                'col': col_idx, 'row': row_idx
            })
        
        self.logger.debug(f"Layout: {num_cols}x{num_rows}, {len(rect_definitions)} rects (requested: {num_rects_requested}, actual: {actual_cells_to_create})")
        return rect_definitions, rect_w, rect_h

    def show(self, num_rects: Optional[int] = None) -> None:
        """Show the grid overlay."""
        self.logger.info(f"[GridView] show() called with num_rects={num_rects}")
        if not self.root:
            self.logger.error("[GridView] Cannot show grid: Tk root not available")
            return
        
        self._schedule_in_tk_thread(self._show_tkinter_elements, num_rects)

    def _show_tkinter_elements(self, num_rects: Optional[int] = None) -> None:
        self.logger.info(f"[GridView] _show_tkinter_elements called with num_rects={num_rects}")
        if not self.root:
            self.logger.error("[GridView] Tk root not available in _show_tkinter_elements")
            return

        # Create overlay window if needed
        self._ensure_overlay_window()
        
        num_rects_to_display = num_rects if num_rects and num_rects > 0 else self.default_num_rects
        self.logger.info(f"[GridView] Calculating grid layout for {num_rects_to_display} rectangles")
        self.current_num_rects_displayed = num_rects_to_display

        # Calculate layout
        rect_definitions, cell_w, cell_h = self._calculate_grid_layout(num_rects_to_display)
        if not rect_definitions:
            self.logger.error("[GridView] No rectangles to display")
            return

        # Generate request for click counts
        request_id = uuid.uuid4().hex
        self._pending_renders[request_id] = {
            "cell_w": cell_w,
            "cell_h": cell_h,
            "num_rects_to_display": num_rects_to_display
        }

        # Request click counts
        event_data = RequestClickCountsForGridEventData(
            rect_definitions=rect_definitions,
            request_id=request_id
        )        
        self.logger.info(f"[GridView] Publishing RequestClickCountsForGridEvent with request_id={request_id} and {len(rect_definitions)} rects")
        
        # Use ThreadSafeEventPublisher to publish the event
        try:
            self.event_publisher.publish(event_data)
            self.logger.info("[GridView] Successfully queued RequestClickCountsForGridEvent to event bus")
        except Exception as e:
            self.logger.error(f"[GridView] Failed to request click counts: {e}")
            self._pending_renders.pop(request_id, None)

    def _ensure_overlay_window(self) -> None:
        """Ensure overlay window and canvas exist."""
        if not self.overlay_window or not self.overlay_window.winfo_exists():
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.attributes('-fullscreen', True)
            self.overlay_window.attributes('-alpha', view_config.grid.window_alpha)
            self.overlay_window.overrideredirect(True)
            self.overlay_window.wm_attributes('-topmost', True)
            
            try:
                self.overlay_window.attributes('-toolwindow', True)
            except tk.TclError:
                self.logger.warning("Failed to set -toolwindow attribute")
            
            # Configure overlay window grid
            self.overlay_window.grid_rowconfigure(0, weight=1)
            self.overlay_window.grid_columnconfigure(0, weight=1)
            
            self.canvas = tk.Canvas(
                self.overlay_window, 
                highlightthickness=0,
                bg='black',  # Set background color to ensure visibility
                bd=0,  # Remove border
                relief='flat'  # Remove relief
            )
            self.canvas.grid(row=0, column=0, sticky="nsew")
            
            # Force window to update geometry
            self.overlay_window.update_idletasks()
            
            # Set active flag when window is created
            self._is_active = True
            self.logger.debug("[GridView] Overlay window created and set to active")
        elif self.canvas:
            self.canvas.delete("all")

    def hide(self) -> None:
        """Hide the grid overlay."""
        if not self.is_active():
            return
        
        if not self.root:
            self.logger.error("Cannot hide grid: Tk root not available")
            return
        
        self._schedule_in_tk_thread(self._hide_tkinter_elements)

    def _hide_tkinter_elements(self) -> None:
        """Hide grid elements in the Tkinter thread."""
        if self.overlay_window:
            self.overlay_window.withdraw()
            if self.canvas:
                self.canvas.delete("all")
        
        self.current_num_rects_displayed = None
        self._is_active = False
        self.logger.info("Grid overlay hidden")

    def refresh_display(self) -> None:
        """Refresh the grid display with current number of rectangles."""
        if self.current_num_rects_displayed is not None:
            self.show(self.current_num_rects_displayed)
        else:
            self.logger.warning("Cannot refresh: no previous rectangle count")

    def is_active(self) -> bool:
        """Check if grid overlay is currently active and visible."""
        try:
            return (self.overlay_window is not None 
                    and self.overlay_window.winfo_exists() 
                    and self.overlay_window.winfo_viewable())
        except tk.TclError:
            # Window may have been destroyed
            self.overlay_window = None
            self._is_active = False
            return False

    def handle_selection(self, selection_key: str) -> bool:
        """Handle grid cell selection."""
        self.logger.info(f"[GridView] handle_selection called with selection_key='{selection_key}'")
        self.logger.debug(f"[GridView] Grid active: {self.is_active()}, Root available: {self.root is not None}")
        self.logger.debug(f"[GridView] Available rectangles: {list(self.ui_to_rect_data_map.keys())}")
        
        if not self.is_active() or not self.root:
            self.logger.warning(f"[GridView] Grid not active or root not available for selection '{selection_key}'")
            return False

        try:
            selected_number = int(selection_key)
            self.logger.debug(f"[GridView] Parsed selection_key '{selection_key}' as number: {selected_number}")
        except ValueError:
            self.logger.warning(f"[GridView] Invalid selection key: '{selection_key}'")
            return False

        if selected_number not in self.ui_to_rect_data_map:
            self.logger.warning(f"[GridView] Selection '{selected_number}' not in available rectangles: {list(self.ui_to_rect_data_map.keys())}")
            return False

        rect_data = self.ui_to_rect_data_map[selected_number]
        center_x, center_y = rect_data['center_x'], rect_data['center_y']
        self.logger.info(f"[GridView] Found rectangle data for cell {selected_number}: center=({center_x}, {center_y})")
        
        try:
            # Hide grid first to prevent interference
            self.logger.debug(f"[GridView] Hiding grid before click")
            self.hide()
            
            # Small delay to ensure grid is fully hidden before clicking
            time.sleep(0.1)
            
            # Perform click
            self.logger.debug(f"[GridView] Performing click at ({center_x}, {center_y})")
            pyautogui.click(center_x, center_y)
            
            # Log click event
            event_data = PerformMouseClickEventData(
                x=int(center_x), 
                y=int(center_y), 
                source="grid_selection"
            )
            self.event_publisher.publish(event_data)
            
            self.logger.info(f"[GridView] Grid cell {selected_number} clicked at ({center_x}, {center_y})")
            
            # Notify controller of successful selection
            if self.controller_callback:
                self.logger.debug(f"[GridView] Notifying controller of successful selection")
                self.controller_callback.on_grid_selection_success(selected_number, center_x, center_y)
            else:
                self.logger.warning(f"[GridView] No controller callback set")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[GridView] Failed to handle selection {selected_number}: {e}", exc_info=True)
            # Notify controller of failed selection
            if self.controller_callback:
                self.logger.debug(f"[GridView] Notifying controller of failed selection")
                self.controller_callback.on_grid_selection_failed(selected_number, str(e))
            else:
                self.logger.warning(f"[GridView] No controller callback set for failed selection")
            return False

    def update_config(self, config_changes: Dict[str, Any]) -> None:
        """Update configuration at runtime."""
        for key, value in config_changes.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated {key} to {value}")
            else:
                self.logger.warning(f"Unknown config key: {key}") 