import asyncio
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pyautogui
from PySide6.QtCore import Q_ARG, QMetaObject, QRect, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.services.grid.click_tracker_service import prioritize_grid_rects
from vocalance.app.ui.qt_theme import theme

# Drag mode: PyAutoGUI only interpolates pointer motion when duration > 0.1s. A short settle after the
# final move helps many Windows targets register the drop at the cell center before button-up.
_GRID_DRAG_MOVE_MIN_S = 0.22
_GRID_DRAG_MOVE_MAX_S = 0.85
_GRID_DRAG_MOVE_DIST_DIVISOR = 2200.0
_GRID_DRAG_SETTLE_S = 0.05


class QtGridView(QWidget):
    """Thread-safe grid overlay for PySide6.

    Key principles:
    - Simple window management without complex event filters
    - Deferred focus to avoid blocking
    - Thread-safe click cache
    - Direct keyboard handling
    """

    DEFAULT_NUM_RECTS = 500

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        click_tracker_service,
        config: GlobalAppConfig,
        default_num_rects: Optional[int] = None,
    ):
        """Initialize grid view.

        Args:
            event_bus: Event bus for publishing events
            event_loop: Async event loop
            click_tracker_service: ClickTrackerService for getting click data
            config: Global app configuration
            default_num_rects: Default number of rectangles to display
        """
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        self.event_loop = event_loop
        self._click_tracker = click_tracker_service
        self.config = config

        self.default_num_rects = default_num_rects or self.DEFAULT_NUM_RECTS

        # Thread safety
        self._state_lock = threading.RLock()

        # Grid state
        self._is_active = False
        self.current_num_rects_displayed: Optional[int] = None
        self.ui_to_rect_data_map: Dict[int, Dict[str, Any]] = {}
        self._current_click_mode: str = "click"
        self._drag_origin: Optional[Tuple[int, int]] = None

        # Cached layout data for fast painting
        self._cached_num_cols: int = 0
        self._cached_num_rows: int = 0
        self._cached_cell_w: float = 0
        self._cached_cell_h: float = 0

        self._is_preparing = False
        self._layout_device_pixel_ratio: float = 1.0

        # Focus management - track pending focus timers
        self._focus_timers: List[QTimer] = []

        # Setup window as frameless overlay - stays on top of everything
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        # Enable translucent background for proper alpha rendering
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Set focus policy early - BEFORE window is ever shown
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Visual properties - ALL FULLY OPAQUE for reliable rendering
        # Background layer - semi-transparent overlay
        self.background_color = QColor(theme.config.shapes.dark)
        self.background_color.setAlpha(180)  # Semi-transparent background

        # Grid lines - FULLY OPAQUE
        self.grid_line_color = QColor(theme.config.shapes.medium)
        self.grid_line_color.setAlpha(255)  # 100% opaque

        # Text - FULLY OPAQUE
        self.text_color = QColor(theme.config.blue.blue_2)
        self.text_color.setAlpha(255)  # 100% opaque

        self.font = QFont(theme.config.font_family_primary, theme.config.fonts.small, QFont.Weight.Bold)

        # Adaptive font scaling
        self.current_font_size = theme.config.fonts.small
        self.min_font_size = 7
        self.max_font_size = theme.config.fonts.large
        self.default_font_size = theme.config.fonts.small  # Base font size for proportionality calculation

        # Controller callback
        self.controller_callback = None

        # Screen dimensions will be set dynamically when grid is shown
        self.screen_width = 1920
        self.screen_height = 1080

        self.logger.info("QtGridView initialized")

    def set_controller_callback(self, callback) -> None:
        """Set the controller callback."""
        self.controller_callback = callback

    def _calculate_click_counts_sync(self, rect_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate click counts from click tracker service synchronously.

        OPTIMIZATION: Uses spatial bucketing to reduce O(N*M) to O(N+M) complexity.
        For 2025 cells and 1000 clicks: reduces from 2,025,000 to 3,025 operations!
        """
        # Get all clicks from click tracker service (fast, in-memory)
        if not self._click_tracker:
            # No click tracker, return zero counts
            return [{"data": rect_def, "clicks": 0} for rect_def in rect_definitions]

        all_clicks = self._click_tracker.get_all_clicks_sync()

        # OPTIMIZATION: Build spatial index of clicks using grid bucketing
        # This reduces complexity from O(N*M) to O(N+M)
        if not all_clicks:
            return [{"data": rect_def, "clicks": 0} for rect_def in rect_definitions]

        # Build spatial buckets: group clicks by cell they fall into
        # Use first rect to determine bucket size (all rects have same dimensions)
        if not rect_definitions:
            return []

        first_rect = rect_definitions[0]
        bucket_w = int(first_rect["w"])
        bucket_h = int(first_rect["h"])

        # Create spatial hash map: (bucket_x, bucket_y) -> [clicks]
        click_buckets = {}
        for click in all_clicks:
            try:
                click_x = click.get("x", 0)
                click_y = click.get("y", 0)
                bucket_x = int(click_x // bucket_w)
                bucket_y = int(click_y // bucket_h)
                key = (bucket_x, bucket_y)
                if key not in click_buckets:
                    click_buckets[key] = []
                click_buckets[key].append(click)
            except (TypeError, ValueError):
                continue

        # Now count clicks per rect by checking only relevant buckets
        processed_rects = []
        for rect_def in rect_definitions:
            try:
                rect_x, rect_y = int(rect_def["x"]), int(rect_def["y"])
                rect_w, rect_h = int(rect_def["w"]), int(rect_def["h"])

                # Determine which bucket(s) this rect overlaps
                bucket_x = int(rect_x // bucket_w)
                bucket_y = int(rect_y // bucket_h)

                # Check clicks in this bucket and adjacent buckets (for edge cases)
                count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        key = (bucket_x + dx, bucket_y + dy)
                        if key in click_buckets:
                            for click in click_buckets[key]:
                                if self._is_click_in_rect(click, rect_x, rect_y, rect_w, rect_h):
                                    count += 1

                processed_rects.append({"data": rect_def, "clicks": count})

            except (KeyError, ValueError, TypeError):
                processed_rects.append({"data": rect_def, "clicks": 0})

        return processed_rects

    def _is_click_in_rect(self, click: Dict[str, Any], rect_x: int, rect_y: int, rect_w: int, rect_h: int) -> bool:
        """Check if click is in rectangle."""
        try:
            click_x, click_y = click.get("x", 0), click.get("y", 0)
            return rect_x <= click_x <= rect_x + rect_w and rect_y <= click_y <= rect_y + rect_h
        except (TypeError, ValueError):
            return False

    def _calculate_adaptive_font_size(self, num_rects: int) -> int:
        """Calculate adaptive font size based on number of grid cells.

        Uses inverse proportionality: fewer cells = larger font, more cells = smaller font.
        Scales from max_font_size (few cells) down to min_font_size (many cells).

        Args:
            num_rects: Number of grid cells to display

        Returns:
            Font size in points, clamped between min and max
        """
        if num_rects <= 0:
            return self.default_font_size

        # Use inverse proportionality: font_size ∝ 1/sqrt(num_cells)
        # This ensures reasonable scaling: fewer cells get progressively larger fonts
        # sqrt is used to compress the scaling range to reasonable values
        reference_rects = 100  # Reference point for default font size
        scale_factor = math.sqrt(reference_rects / num_rects)
        calculated_size = int(self.default_font_size * scale_factor)

        # Clamp between min and max
        clamped_size = max(self.min_font_size, min(self.max_font_size, calculated_size))

        self.logger.debug(
            f"Font size calculation: {num_rects} cells -> scale={scale_factor:.2f} -> {calculated_size}pt -> clamped={clamped_size}pt"
        )

        return clamped_size

    def _calculate_grid_layout(self, num_rects_requested: int) -> Tuple[List[Dict[str, Any]], float, float]:
        """Calculate grid layout that perfectly fills the screen with no partial cells.

        The grid will have cells of exactly equal dimensions that cover the entire screen.
        The actual number of cells may differ from the requested number - we round to the
        nearest number that fits perfectly with no remainder.

        Args:
            num_rects_requested: Target number of cells (actual may differ)

        Returns:
            Tuple of (rect_definitions, cell_width, cell_height)
        """
        if num_rects_requested <= 0:
            return [], 0, 0

        # Calculate the ideal cell dimensions based on requested count
        screen_aspect_ratio = self.screen_width / self.screen_height
        total_screen_area = self.screen_width * self.screen_height
        target_cell_area = total_screen_area / num_rects_requested

        # Calculate ideal cell dimensions maintaining screen aspect ratio
        ideal_cell_h = math.sqrt(target_cell_area / screen_aspect_ratio)
        ideal_cell_w = ideal_cell_h * screen_aspect_ratio

        # Calculate how many columns and rows would fit
        # Round to nearest integer (not floor) to get closest match to requested count
        num_cols = max(1, round(self.screen_width / ideal_cell_w))
        num_rows = max(1, round(self.screen_height / ideal_cell_h))

        # Calculate EXACT cell dimensions that perfectly divide the screen
        # Using integer division ensures no partial cells or gaps
        rect_w = self.screen_width / num_cols
        rect_h = self.screen_height / num_rows

        actual_cells_to_create = num_cols * num_rows

        # Generate cell definitions
        rect_definitions = []
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                # Calculate exact pixel positions
                x = col_idx * rect_w
                y = row_idx * rect_h
                center_x = x + rect_w / 2
                center_y = y + rect_h / 2

                rect_definitions.append(
                    {
                        "x": x,
                        "y": y,
                        "w": rect_w,
                        "h": rect_h,
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                )

        self.logger.info(
            f"Grid layout: requested={num_rects_requested}, actual={actual_cells_to_create} "
            f"({num_cols}x{num_rows}), cell_size={rect_w:.1f}x{rect_h:.1f}"
        )

        return rect_definitions, rect_w, rect_h

    def paintEvent(self, event) -> None:
        """Paint grid overlay with four distinct layers:

        LAYER 1 (Bottom): Full-screen semi-transparent background
        LAYER 1.5 (Below grid): Colored cell fills for top 50 cells (vibrant, less transparent)
        LAYER 2 (Middle): Opaque grid lines (vertical + horizontal)
        LAYER 3 (Top): Opaque text labels centered in each cell
        """
        if self._is_preparing:
            return

        if not self._is_active:
            return

        with self._state_lock:
            rect_map = dict(self.ui_to_rect_data_map)
            num_cols = self._cached_num_cols
            num_rows = self._cached_num_rows
            cell_w = self._cached_cell_w
            cell_h = self._cached_cell_h

        if not rect_map or num_cols == 0 or num_rows == 0 or cell_w == 0 or cell_h == 0:
            return

        dpr = self._layout_device_pixel_ratio

        # Get the window size in logical pixels (this is the actual drawable area)
        window_width = self.width()
        window_height = self.height()

        # Grid starts at (0, 0) and fills the entire window
        min_x = 0
        min_y = 0
        max_x = window_width
        max_y = window_height

        # Begin painting
        painter = QPainter(self)

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 1: Full-screen semi-transparent background (BOTTOM LAYER)
        # ═══════════════════════════════════════════════════════════════════
        painter.fillRect(self.rect(), self.background_color)

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 1.5: Colored cell fills for top 50 cells (vibrant gradient)
        # ═══════════════════════════════════════════════════════════════════
        top_n_cells = min(50, len(rect_map))
        for ui_number in range(1, top_n_cells + 1):
            if ui_number not in rect_map:
                continue

            rect_data = rect_map[ui_number]

            # Calculate cell position
            raw_x = rect_data["x"] / dpr
            raw_y = rect_data["y"] / dpr
            col = round((raw_x - min_x) / cell_w)
            row = round((raw_y - min_y) / cell_h)
            cell_x = min_x + col * cell_w
            cell_y = min_y + row * cell_h

            # Calculate color based on priority (1 = highest priority, 50 = lowest of top 50)
            # Lower numbers get more vibrant and less transparent colors
            # Use blue color as base (matching the theme)
            base_color = QColor(theme.config.blue.blue_2)  # #8a8ac8

            # Calculate alpha: cell 1 gets highest alpha (220), cell 50 gets lowest (60)
            # Linear interpolation from 220 to 60
            alpha = int(220 - ((ui_number - 1) / (top_n_cells - 1) * 160)) if top_n_cells > 1 else 220

            # Calculate color intensity: cell 1 gets full intensity, cell 50 gets reduced
            # This makes lower numbers more vibrant
            intensity_factor = 1.0 - ((ui_number - 1) / (top_n_cells - 1) * 0.4) if top_n_cells > 1 else 1.0

            # Apply intensity to RGB values
            r = int(base_color.red() * intensity_factor)
            g = int(base_color.green() * intensity_factor)
            b = int(base_color.blue() * intensity_factor)

            fill_color = QColor(r, g, b, alpha)

            # Fill the cell with the calculated color
            cell_rect = QRect(cell_x, cell_y, cell_w, cell_h)
            painter.fillRect(cell_rect, fill_color)

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 2: Grid lines - opaque vertical and horizontal lines
        # ═══════════════════════════════════════════════════════════════════
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        grid_pen = QPen(self.grid_line_color, 1, Qt.PenStyle.SolidLine)
        painter.setPen(grid_pen)

        # Draw all vertical lines (num_cols + 1 lines)
        for col in range(num_cols + 1):
            x = min_x + col * cell_w
            painter.drawLine(x, min_y, x, max_y)

        # Draw all horizontal lines (num_rows + 1 lines)
        for row in range(num_rows + 1):
            y = min_y + row * cell_h
            painter.drawLine(min_x, y, max_x, y)

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 3: Text labels - opaque, centered in each cell (TOP LAYER)
        # ═══════════════════════════════════════════════════════════════════
        # IMPORTANT: Calculate cell positions using the SAME formula as grid lines
        # to ensure perfect alignment. Derive row/col from stored coordinates.
        painter.setPen(QPen(self.text_color))
        painter.setFont(self.font)

        for ui_number, rect_data in rect_map.items():
            # Derive row and column index from stored coordinates
            # This ensures text is positioned exactly between the grid lines
            raw_x = rect_data["x"] / dpr
            raw_y = rect_data["y"] / dpr

            # Calculate which column and row this cell belongs to
            col = round((raw_x - min_x) / cell_w)
            row = round((raw_y - min_y) / cell_h)

            # Calculate cell bounds using the SAME formula as grid lines
            cell_x = min_x + col * cell_w
            cell_y = min_y + row * cell_h

            # Create cell rect that exactly matches the grid cell boundaries
            cell_rect = QRect(cell_x, cell_y, cell_w, cell_h)
            painter.drawText(cell_rect, Qt.AlignmentFlag.AlignCenter, str(ui_number))

        painter.end()

        self.logger.debug(f"Drew grid: {num_cols}x{num_rows} = {len(rect_map)} cells")

    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            # Schedule hide on next event loop iteration to avoid blocking
            QTimer.singleShot(0, self._do_hide)
            event.accept()
        elif event.key() >= Qt.Key.Key_0 and event.key() <= Qt.Key.Key_9:
            digit = event.key() - Qt.Key.Key_0
            # Schedule number input handling to avoid blocking
            QTimer.singleShot(0, lambda: self._handle_number_input(digit))
            event.accept()
        else:
            super().keyPressEvent(event)

    def _handle_number_input(self, digit: int) -> None:
        """Handle number input for grid cell selection."""
        if digit in self.ui_to_rect_data_map:
            self.handle_selection(str(digit), self._current_click_mode)

    def is_active(self) -> bool:
        """Check if overlay is active."""
        return self._is_active and self.isVisible()

    @Slot()
    def show(self, num_rects: Optional[int] = None, click_mode: str = "click") -> None:
        """Show the grid overlay - thread-safe via QMetaObject.invokeMethod."""
        # Use BlockingQueuedConnection for immediate execution when called from main thread
        # This eliminates the event loop delay while maintaining thread safety
        from PySide6.QtCore import QThread

        connection_type = (
            Qt.ConnectionType.DirectConnection
            if QThread.currentThread() == self.thread()
            else Qt.ConnectionType.BlockingQueuedConnection
        )

        QMetaObject.invokeMethod(
            self,
            "_do_show",
            connection_type,
            Q_ARG(int, num_rects if num_rects is not None else self.default_num_rects),
            Q_ARG(str, click_mode),
        )

    @Slot(int, str)
    def _do_show(self, num_rects: int, click_mode: str) -> None:
        """Internal show implementation - MUST run on main Qt thread."""
        with self._state_lock:
            if self._is_active or self._is_preparing:
                self.logger.warning("Grid already active or preparing - ignoring show request")
                return
            self._is_preparing = True

        try:
            start_time = time.perf_counter()
            self.logger.info(f"Showing grid: num_rects={num_rects}, mode={click_mode}")

            with self._state_lock:
                self._current_click_mode = click_mode
                if click_mode == "drag":
                    pos = pyautogui.position()
                    self._drag_origin = (round(pos[0]), round(pos[1]))
                else:
                    self._drag_origin = None

            # Get PRIMARY screen for physical pixel calculations
            primary = QApplication.primaryScreen()
            if primary:
                geometry = primary.geometry()
                device_pixel_ratio = primary.devicePixelRatio()
                self._layout_device_pixel_ratio = float(device_pixel_ratio)

                # Calculate PHYSICAL pixels (what pyautogui uses)
                # Qt geometry() returns logical pixels, multiply by DPR for physical
                self.screen_width = int(geometry.width() * device_pixel_ratio)
                self.screen_height = int(geometry.height() * device_pixel_ratio)

                self.logger.info(
                    f"Screen - Logical: {geometry.width()}x{geometry.height()}, "
                    f"DPR: {device_pixel_ratio}, "
                    f"Physical: {self.screen_width}x{self.screen_height}"
                )

                # OPTIMIZATION: Set window geometry IMMEDIATELY for instant display
                self.setGeometry(geometry)

                logical_width = geometry.width()
                logical_height = geometry.height()
            else:
                self.logger.warning("No primary screen found, using fallback geometry")
                self._layout_device_pixel_ratio = 1.0
                self.screen_width = 1920
                self.screen_height = 1080
                device_pixel_ratio = 1.0
                self.setGeometry(0, 0, 1920, 1080)

                logical_width = 1920
                logical_height = 1080

            # Calculate grid layout using PHYSICAL pixels
            t1 = time.perf_counter()
            rect_definitions, cell_w, cell_h = self._calculate_grid_layout(num_rects)
            t2 = time.perf_counter()
            self.logger.debug(f"Grid layout calculation: {(t2-t1)*1000:.1f}ms")

            if not rect_definitions:
                self.logger.error("No rectangles generated")
                with self._state_lock:
                    self._drag_origin = None
                self._is_preparing = False
                return

            self.logger.debug(f"Calculating click counts for {len(rect_definitions)} cells")
            rects_with_clicks = self._calculate_click_counts_sync(rect_definitions)
            t3 = time.perf_counter()
            self.logger.debug(f"Click counting: {(t3-t2)*1000:.1f}ms")

            weighted_rects = prioritize_grid_rects(rects_with_clicks)
            t4 = time.perf_counter()
            self.logger.debug(f"Prioritization: {(t4-t3)*1000:.1f}ms")

            logical_cell_w = cell_w / device_pixel_ratio
            logical_cell_h = cell_h / device_pixel_ratio
            num_cols = max(1, round(logical_width / logical_cell_w))
            num_rows = max(1, round(logical_height / logical_cell_h))

            logical_cell_w = logical_width / num_cols
            logical_cell_h = logical_height / num_rows

            self.current_font_size = self._calculate_adaptive_font_size(len(weighted_rects))
            self.font.setPointSize(self.current_font_size)

            self.setUpdatesEnabled(False)

            with self._state_lock:
                self.ui_to_rect_data_map.clear()
                for ui_number, weighted_rect_info in enumerate(weighted_rects, 1):
                    self.ui_to_rect_data_map[ui_number] = weighted_rect_info["data"]
                self.current_num_rects_displayed = len(weighted_rects)

                self._cached_num_cols = num_cols
                self._cached_num_rows = num_rows
                self._cached_cell_w = logical_cell_w
                self._cached_cell_h = logical_cell_h

                self._is_active = True

            super().show()
            self.raise_()
            self.activateWindow()
            self.setFocus(Qt.FocusReason.PopupFocusReason)

            self._is_preparing = False

            self.setUpdatesEnabled(True)
            self.update()

            # Schedule deferred focus attempts to overcome Windows focus stealing
            self._schedule_robust_focus()

            end_time = time.perf_counter()
            total_ms = (end_time - start_time) * 1000
            self.logger.info(f"Grid displayed with {len(weighted_rects)} cells in {total_ms:.1f}ms")

        except Exception as e:
            with self._state_lock:
                self._is_preparing = False
                self._is_active = False
                self._drag_origin = None
            self.logger.error(f"Error showing grid: {e}", exc_info=True)

    def _schedule_robust_focus(self) -> None:
        """Schedule deferred focus attempts to overcome Windows focus stealing."""
        self._cancel_focus_timers()

        # Critical delays to overcome taskbar interference
        delays = [50, 200]

        for delay in delays:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._ensure_focus)
            timer.start(delay)
            self._focus_timers.append(timer)

    def _cancel_focus_timers(self) -> None:
        """Cancel all pending focus timers."""
        for timer in self._focus_timers:
            if timer.isActive():
                timer.stop()
            timer.deleteLater()
        self._focus_timers.clear()

    def _ensure_focus(self) -> None:
        """Reinforce focus and window stacking."""
        if self._is_active and not self.isHidden():
            self.raise_()
            self.activateWindow()
            self.setFocus(Qt.FocusReason.PopupFocusReason)

    @Slot()
    def hide(self) -> None:
        """Hide the grid overlay - thread-safe via QMetaObject.invokeMethod."""
        # Ensure this runs on the main Qt thread
        QMetaObject.invokeMethod(self, "_do_hide", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_hide(self) -> None:
        """Internal hide implementation - MUST run on main Qt thread."""
        if not self._is_active:
            return

        try:
            self.logger.info("Hiding grid")

            self._cancel_focus_timers()
            self._is_preparing = False

            self.clearFocus()
            super().hide()
            self._is_active = False

            with self._state_lock:
                self.ui_to_rect_data_map.clear()
                self.current_num_rects_displayed = None
                self._cached_num_cols = 0
                self._cached_num_rows = 0
                self._cached_cell_w = 0
                self._cached_cell_h = 0
                self._drag_origin = None

            self.logger.info("Grid hidden")

        except Exception as e:
            self.logger.error(f"Error hiding grid: {e}", exc_info=True)

    def _publish_grid_mouse_click(self, x: int, y: int) -> None:
        """Log a grid-originated click at physical coordinates (click tracking)."""
        ev = PerformMouseClickEventData(x=x, y=y, source="grid")
        asyncio.run_coroutine_threadsafe(self.event_bus.publish(ev), self.event_loop)

    def _execute_grid_pointer_action(
        self,
        click_mode: str,
        center_x: float,
        center_y: float,
        drag_origin: Optional[Tuple[int, int]],
    ) -> None:
        """Perform pyautogui action for grid cell (physical pixel coordinates)."""
        cx, cy = int(center_x), int(center_y)
        if click_mode == "click":
            pyautogui.click(cx, cy)
            self._publish_grid_mouse_click(cx, cy)
        elif click_mode == "hover":
            pyautogui.moveTo(cx, cy)
        elif click_mode == "drag":
            if drag_origin is None:
                raise RuntimeError("Drag mode requires a recorded start position")
            ox, oy = drag_origin
            tx, ty = round(center_x), round(center_y)
            dist = math.hypot(float(tx - ox), float(ty - oy))
            duration = min(
                _GRID_DRAG_MOVE_MAX_S,
                max(_GRID_DRAG_MOVE_MIN_S, dist / _GRID_DRAG_MOVE_DIST_DIVISOR),
            )
            pyautogui.moveTo(ox, oy, duration=0.0, _pause=False)
            pyautogui.mouseDown(ox, oy, button="left", _pause=False)
            try:
                pyautogui.moveTo(tx, ty, duration=duration, _pause=False)
                time.sleep(_GRID_DRAG_SETTLE_S)
            finally:
                pyautogui.mouseUp(tx, ty, button="left", _pause=False)
            self._publish_grid_mouse_click(tx, ty)
        else:
            raise ValueError(f"Unsupported grid click_mode: {click_mode!r}")

    def handle_selection(self, selection_key: str, click_mode: str = "click") -> bool:
        """Handle grid cell selection.

        IMPORTANT: The overlay must be hidden BEFORE clicking, otherwise the click
        lands on the overlay instead of the underlying screen content.
        """
        try:
            selected_number = int(selection_key)
        except ValueError:
            self.logger.warning(f"Invalid selection: {selection_key}")
            return False

        with self._state_lock:
            if selected_number not in self.ui_to_rect_data_map:
                self.logger.warning(f"Selection {selected_number} not in grid")
                return False

            rect_data = self.ui_to_rect_data_map[selected_number]
            drag_origin = self._drag_origin if click_mode == "drag" else None

        if click_mode == "drag" and drag_origin is None:
            self.logger.error("Drag mode selection failed: no recorded start position")
            return False

        # Coordinates are already in physical pixels (matching pyautogui)
        center_x = rect_data["center_x"]
        center_y = rect_data["center_y"]

        self.logger.info(f"Grid cell {selected_number} selected at physical coords ({center_x}, {center_y})")

        # CRITICAL: Hide the overlay FIRST, then schedule the click after a delay
        # This ensures the overlay is fully gone before we click on the screen
        self.hide()

        # Schedule the actual click/hover action using threading.Timer (not QTimer)
        # QTimer requires Qt event loop which may not be available from calling thread
        def perform_action():
            try:
                # Small delay to ensure overlay is fully hidden
                time.sleep(0.05)  # 50ms delay

                self._execute_grid_pointer_action(click_mode, center_x, center_y, drag_origin)

                self.logger.info(f"Action '{click_mode}' performed at ({center_x}, {center_y})")

                if self.controller_callback:
                    self.controller_callback.on_grid_selection_success(selected_number, center_x, center_y)

            except Exception as e:
                self.logger.error(f"Error performing action: {e}", exc_info=True)
                if self.controller_callback:
                    self.controller_callback.on_grid_selection_failed(selected_number, str(e))

        # Use threading.Timer for cross-thread compatibility
        action_thread = threading.Timer(0.001, perform_action)  # Start almost immediately but in background
        action_thread.daemon = True
        action_thread.start()

        return True

    def cleanup(self) -> None:
        """Clean up resources."""
        self._cancel_focus_timers()
        self.hide()

        with self._state_lock:
            self.ui_to_rect_data_map.clear()
            self._is_active = False

        self.logger.debug("QtGridView cleanup completed")
