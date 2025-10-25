# Grid View - UI-specific grid overlay implementation
import asyncio
import logging
import math
import threading
import time
import tkinter as tk
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyautogui

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.services.grid.click_tracker_service import prioritize_grid_rects
from vocalance.app.services.storage.storage_models import GridClicksData
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.ui.views.components.view_config import view_config
from vocalance.app.utils.event_utils import EventSubscriptionManager, ThreadSafeEventPublisher


class GridView:
    """
    Thread-safe grid overlay view with simplified event handling.

    Thread Safety:
    - UI operations (show/hide/draw) run in main tkinter thread
    - Event handlers run in GUI event loop thread
    - Shared state protected by _state_lock (RLock for reentrancy)
    - Click cache operations are atomic with lock protection
    """

    # Constants
    DEFAULT_NUM_RECTS = 500

    def __init__(
        self,
        root: tk.Tk,
        event_bus: EventBus,
        default_num_rects: Optional[int] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        storage: Optional[StorageService] = None,
    ):
        # Initialize logging and core attributes
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root = root
        self.event_bus = event_bus
        self.event_loop = event_loop or asyncio.get_event_loop()
        self._storage = storage

        self.default_num_rects = default_num_rects or self.DEFAULT_NUM_RECTS

        # Thread safety: RLock protects all shared state
        # RLock (not Lock) allows reentrant access from same thread
        self._state_lock = threading.RLock()

        # Protected state variables (all access must be under lock)
        self._is_active = False
        self.current_num_rects_displayed: Optional[int] = None
        self.ui_to_rect_data_map: Dict[int, Dict[str, Any]] = {}
        self._cached_clicks: List[Dict[str, Any]] = []
        self._click_cache_timestamp: float = 0.0
        self._cache_loaded = False

        # Initialize UI components (main thread only)
        self.overlay_window: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None

        # Event handling - use provided event_loop if available
        self.event_publisher = ThreadSafeEventPublisher(event_bus, self.event_loop)
        self.subscription_manager = EventSubscriptionManager(event_bus, "GridView")

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

    async def initialize_click_cache(self) -> None:
        """Load historical click data from storage into cache. Thread-safe."""
        # Check both conditions under lock atomically
        with self._state_lock:
            if self._cache_loaded:
                return

        if not self._storage:
            return

        try:
            self.logger.info("[GridView] Loading historical click data from storage...")
            clicks_data = await self._storage.read(model_type=GridClicksData)

            with self._state_lock:
                if clicks_data.clicks:
                    # Convert GridClickEvent objects to dictionaries for compatibility
                    self._cached_clicks = [click.model_dump() for click in clicks_data.clicks]
                    self._click_cache_timestamp = time.time()
                    self._cache_loaded = True
                    self.logger.info(f"[GridView] Loaded {len(clicks_data.clicks)} historical clicks into cache")
                else:
                    self.logger.info("[GridView] No historical click data found, starting fresh")
                    self._cache_loaded = True

        except Exception as e:
            self.logger.error(f"[GridView] Failed to load historical clicks: {e}", exc_info=True)
            with self._state_lock:
                self._cache_loaded = True

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
        self.subscription_manager.subscribe(PerformMouseClickEventData, self._handle_click_logged_for_cache)

    def cleanup(self) -> None:
        """Clean up resources when grid view is destroyed. Thread-safe."""
        self.hide()
        self.subscription_manager.unsubscribe_all()

        with self._state_lock:
            self._cached_clicks.clear()
            self.ui_to_rect_data_map.clear()
            self._is_active = False

        if self.overlay_window:
            try:
                self.overlay_window.destroy()
            except tk.TclError:
                pass
        self.overlay_window = None

    async def _handle_click_logged_for_cache(self, event_data: PerformMouseClickEventData) -> None:
        """Cache click data for instant grid display. Thread-safe."""
        click_data = {"x": event_data.x, "y": event_data.y, "timestamp": time.time(), "source": event_data.source}

        with self._state_lock:
            self._cached_clicks.append(click_data)
            self._click_cache_timestamp = time.time()

            max_cache_size = 10000
            if len(self._cached_clicks) > max_cache_size:
                self._cached_clicks = self._cached_clicks[-max_cache_size:]

            cache_size = len(self._cached_clicks)

        self.logger.debug(f"[GridView] Cached click at ({event_data.x}, {event_data.y}), total cached: {cache_size}")

    def _calculate_click_counts_sync(self, rect_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate click counts synchronously using cached data for instant display. Thread-safe."""
        processed_rects = []

        # Take snapshot of clicks under lock to avoid race conditions
        with self._state_lock:
            cached_clicks_snapshot = list(self._cached_clicks)

        for rect_def in rect_definitions:
            try:
                rect_x, rect_y = int(rect_def["x"]), int(rect_def["y"])
                rect_w, rect_h = int(rect_def["w"]), int(rect_def["h"])

                count = sum(1 for click in cached_clicks_snapshot if self._is_click_in_rect(click, rect_x, rect_y, rect_w, rect_h))

                processed_rects.append({"data": rect_def, "clicks": count})

            except (KeyError, ValueError, TypeError):
                processed_rects.append({"data": rect_def, "clicks": 0})

        return processed_rects

    def _is_click_in_rect(self, click: Dict[str, Any], rect_x: int, rect_y: int, rect_w: int, rect_h: int) -> bool:
        """Check if click falls within rectangle bounds."""
        try:
            click_x, click_y = click.get("x", 0), click.get("y", 0)
            return rect_x <= click_x <= rect_x + rect_w and rect_y <= click_y <= rect_y + rect_h
        except (TypeError, ValueError):
            return False

    def _draw_grid_elements(self, weighted_rects: List[Dict[str, Any]], cell_w: float, cell_h: float) -> None:
        """Draw grid elements. Must be called from main thread. Thread-safe state access."""
        self.logger.info(f"[GridView] Drawing {len(weighted_rects)} grid elements with cell_w={cell_w}, cell_h={cell_h}")
        if not self._validate_ui_state():
            self.logger.error("[GridView] UI state invalid for drawing")
            return

        with self._state_lock:
            self.ui_to_rect_data_map.clear()

        self.canvas.delete("all")

        font_tuple = view_config.grid.get_font_tuple(cell_h)
        self.logger.debug(f"[GridView] Using font: {font_tuple}")

        # Get colors with transparency applied
        fill_color = view_config.grid.get_fill_color_with_alpha()
        outline_color = view_config.grid.get_outline_color_with_alpha()
        text_color = view_config.grid.get_text_color_with_alpha()

        # Draw all grid elements while window is hidden
        for ui_number, weighted_rect_info in enumerate(weighted_rects, 1):
            rect_data = weighted_rect_info["data"]

            with self._state_lock:
                self.ui_to_rect_data_map[ui_number] = rect_data

            x0, y0 = rect_data["x"], rect_data["y"]
            x1, y1 = x0 + rect_data["w"], y0 + rect_data["h"]

            rect_kwargs = {"outline": outline_color, "width": 2}
            if fill_color:
                rect_kwargs["fill"] = fill_color

            self.canvas.create_rectangle(x0, y0, x1, y1, **rect_kwargs)
            self.canvas.create_text(
                rect_data["center_x"], rect_data["center_y"], text=str(ui_number), fill=text_color, font=font_tuple
            )

        # Update canvas with all content ready
        self.canvas.update_idletasks()

        # Now show window with all content ready - instant display
        self._show_window_with_content()

        with self._state_lock:
            self._is_active = True
            rect_count = len(self.ui_to_rect_data_map)

        self.logger.info(f"[GridView] Grid displayed instantly with {rect_count} rectangles")

    def _validate_ui_state(self) -> bool:
        """Validate that UI components are ready for drawing."""
        if not self.overlay_window or not self.overlay_window.winfo_exists():
            self.logger.error("Overlay window not available for drawing")
            return False
        if not self.canvas:
            self.logger.error("Canvas not available for drawing")
            return False
        return True

    def _show_window_with_content(self) -> None:
        """Show the overlay window with content ready - single atomic operation."""
        if self.overlay_window:
            self.overlay_window.deiconify()
            self.overlay_window.lift()
            self.overlay_window.focus_force()
            self.logger.debug("[GridView] Overlay window shown with content ready")

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

            rect_definitions.append(
                {
                    "id": i,
                    "x": x,
                    "y": y,
                    "w": rect_w,
                    "h": rect_h,
                    "center_x": center_x,
                    "center_y": center_y,
                    "col": col_idx,
                    "row": row_idx,
                }
            )

        self.logger.debug(
            f"Layout: {num_cols}x{num_rows}, {len(rect_definitions)} rects (requested: {num_rects_requested}, actual: {actual_cells_to_create})"
        )
        return rect_definitions, rect_w, rect_h

    def show(self, num_rects: Optional[int] = None) -> None:
        """Show the grid overlay."""
        self.logger.info(f"[GridView] show() called with num_rects={num_rects}")
        if not self.root:
            self.logger.error("[GridView] Cannot show grid: Tk root not available")
            return

        self._schedule_in_tk_thread(self._show_tkinter_elements, num_rects)

    def _show_tkinter_elements(self, num_rects: Optional[int] = None) -> None:
        """Show grid elements in tkinter. Must be called from main thread. Thread-safe state access."""
        self.logger.info(f"[GridView] _show_tkinter_elements called with num_rects={num_rects}")
        if not self.root:
            self.logger.error("[GridView] Tk root not available in _show_tkinter_elements")
            return

        # Create overlay window hidden
        self._ensure_overlay_window_hidden()

        num_rects_to_display = num_rects if num_rects and num_rects > 0 else self.default_num_rects
        self.logger.info(f"[GridView] Calculating grid layout for {num_rects_to_display} rectangles")

        with self._state_lock:
            self.current_num_rects_displayed = num_rects_to_display

        # Calculate layout
        rect_definitions, cell_w, cell_h = self._calculate_grid_layout(num_rects_to_display)
        if not rect_definitions:
            self.logger.error("[GridView] No rectangles to display")
            return

        # Calculate click counts synchronously using cached data for instant display
        with self._state_lock:
            cache_size = len(self._cached_clicks)
        self.logger.debug(f"[GridView] Calculating click counts synchronously with {cache_size} cached clicks")
        processed_rects = self._calculate_click_counts_sync(rect_definitions)
        weighted_rects = prioritize_grid_rects(processed_rects)

        # Draw grid immediately with final prioritized order - no background updates
        self._draw_grid_elements(weighted_rects, cell_w, cell_h)

    def _close_window_event(self, event=None):
        """Handle window close event."""
        self.hide()

    def _ensure_overlay_window_hidden(self) -> None:
        """Ensure overlay window and canvas exist, but keep hidden until content is ready."""
        if not self.overlay_window or not self.overlay_window.winfo_exists():
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.attributes("-fullscreen", True)
            self.overlay_window.attributes("-alpha", view_config.grid.window_alpha)
            self.overlay_window.overrideredirect(True)
            self.overlay_window.wm_attributes("-topmost", True)

            try:
                self.overlay_window.attributes("-toolwindow", True)
            except tk.TclError:
                self.logger.warning("Failed to set -toolwindow attribute")

            # Configure overlay window grid
            self.overlay_window.grid_rowconfigure(0, weight=1)
            self.overlay_window.grid_columnconfigure(0, weight=1)

            self.canvas = tk.Canvas(self.overlay_window, highlightthickness=0, bg="black", bd=0, relief="flat")
            self.canvas.grid(row=0, column=0, sticky="nsew")

            # Close on Escape key
            self.overlay_window.bind("<Escape>", self._close_window_event)

            # Set up proper window close handling
            self.overlay_window.protocol("WM_DELETE_WINDOW", self._close_window_event)

            # Keep window hidden initially
            self.overlay_window.withdraw()

            # Force geometry update while hidden
            self.overlay_window.update_idletasks()

            self.logger.debug("[GridView] Overlay window created (hidden, awaiting content)")
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
        """Hide grid elements in the Tkinter thread. Thread-safe state access."""
        if self.overlay_window:
            self.overlay_window.withdraw()
            if self.canvas:
                self.canvas.delete("all")

        with self._state_lock:
            self.current_num_rects_displayed = None
            self._is_active = False

        self.logger.info("Grid overlay hidden")

    def refresh_display(self) -> None:
        """Refresh the grid display with current number of rectangles. Thread-safe."""
        with self._state_lock:
            num_rects = self.current_num_rects_displayed

        if num_rects is not None:
            self.show(num_rects)
        else:
            self.logger.warning("Cannot refresh: no previous rectangle count")

    def is_active(self) -> bool:
        """Check if grid overlay is currently active and visible. Thread-safe."""
        try:
            is_window_visible = (
                self.overlay_window is not None and self.overlay_window.winfo_exists() and self.overlay_window.winfo_viewable()
            )
            return is_window_visible
        except tk.TclError:
            # Window may have been destroyed
            self.overlay_window = None
            with self._state_lock:
                self._is_active = False
            return False

    def handle_selection(self, selection_key: str) -> bool:
        """Handle grid cell selection. Thread-safe."""
        self.logger.info(f"[GridView] handle_selection called with selection_key='{selection_key}'")
        self.logger.debug(f"[GridView] Grid active: {self.is_active()}, Root available: {self.root is not None}")

        with self._state_lock:
            available_keys = list(self.ui_to_rect_data_map.keys())

        self.logger.debug(f"[GridView] Available rectangles: {available_keys}")

        if not self.is_active() or not self.root:
            self.logger.warning(f"[GridView] Grid not active or root not available for selection '{selection_key}'")
            return False

        try:
            selected_number = int(selection_key)
            self.logger.debug(f"[GridView] Parsed selection_key '{selection_key}' as number: {selected_number}")
        except ValueError:
            self.logger.warning(f"[GridView] Invalid selection key: '{selection_key}'")
            return False

        with self._state_lock:
            if selected_number not in self.ui_to_rect_data_map:
                self.logger.warning(
                    f"[GridView] Selection '{selected_number}' not in available rectangles: {list(self.ui_to_rect_data_map.keys())}"
                )
                return False

            rect_data = self.ui_to_rect_data_map[selected_number]

        center_x, center_y = rect_data["center_x"], rect_data["center_y"]
        self.logger.info(f"[GridView] Found rectangle data for cell {selected_number}: center=({center_x}, {center_y})")

        try:
            # Hide grid first to prevent interference
            self.logger.debug("[GridView] Hiding grid before click")
            self.hide()

            # Small delay to ensure grid is fully hidden before clicking
            time.sleep(0.1)

            # Perform click
            self.logger.debug(f"[GridView] Performing click at ({center_x}, {center_y})")
            pyautogui.click(center_x, center_y)

            # Log click event
            event_data = PerformMouseClickEventData(x=int(center_x), y=int(center_y), source="grid_selection")
            self.event_publisher.publish(event_data)

            self.logger.info(f"[GridView] Grid cell {selected_number} clicked at ({center_x}, {center_y})")

            # Notify controller of successful selection
            if self.controller_callback:
                self.logger.debug("[GridView] Notifying controller of successful selection")
                self.controller_callback.on_grid_selection_success(selected_number, center_x, center_y)
            else:
                self.logger.warning("[GridView] No controller callback set")

            return True

        except Exception as e:
            self.logger.error(f"[GridView] Failed to handle selection {selected_number}: {e}", exc_info=True)
            # Notify controller of failed selection
            if self.controller_callback:
                self.logger.debug("[GridView] Notifying controller of failed selection")
                self.controller_callback.on_grid_selection_failed(selected_number, str(e))
            else:
                self.logger.warning("[GridView] No controller callback set for failed selection")
            return False

    def update_config(self, config_changes: Dict[str, Any]) -> None:
        """Update configuration at runtime."""
        for key, value in config_changes.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated {key} to {value}")
            else:
                self.logger.warning(f"Unknown config key: {key}")
