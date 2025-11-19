"""Qt-based grid visualization view - SIMPLIFIED AND THREAD-SAFE.

Frameless overlay window for displaying numbered grid cells with smart prioritization.
Uses proper PySide6 patterns for cross-thread GUI operations.
"""

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
from vocalance.app.services.storage.storage_models import GridClicksData
from vocalance.app.services.storage.storage_service import StorageService


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
        storage: Optional[StorageService],
        config: GlobalAppConfig,
        default_num_rects: Optional[int] = None,
    ):
        """Initialize grid view."""
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        self.event_loop = event_loop
        self._storage = storage
        self.config = config

        self.default_num_rects = default_num_rects or self.DEFAULT_NUM_RECTS

        # Thread safety
        self._state_lock = threading.RLock()

        # Grid state
        self._is_active = False
        self.current_num_rects_displayed: Optional[int] = None
        self.ui_to_rect_data_map: Dict[int, Dict[str, Any]] = {}

        # Click tracking
        self._cached_clicks: List[Dict[str, Any]] = []
        self._click_cache_timestamp: float = 0.0
        self._cache_loaded = False
        self._current_click_mode: str = "click"

        # Setup window as frameless overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Visual properties
        self.fill_color = QColor("#000000")
        self.fill_color.setAlpha(0)  # Transparent
        self.outline_color = QColor("#FFFFFF")
        self.text_color = QColor("#FFFFFF")
        self.font = QFont("Arial", 10, QFont.Weight.Bold)

        # Controller callback
        self.controller_callback = None

        # Subscribe to click events
        self.event_bus.subscribe(PerformMouseClickEventData, self._handle_click_logged_for_cache)

        # Get screen dimensions
        screen = self.screen()
        if screen:
            screen_geometry = screen.geometry()
            self.screen_width = screen_geometry.width()
            self.screen_height = screen_geometry.height()
        else:
            primary = QApplication.primaryScreen()
            if primary:
                screen_geometry = primary.geometry()
                self.screen_width = screen_geometry.width()
                self.screen_height = screen_geometry.height()
            else:
                self.screen_width = 1920
                self.screen_height = 1080

        self.logger.info(f"QtGridView initialized. Screen: {self.screen_width}x{self.screen_height}")

    def set_controller_callback(self, callback) -> None:
        """Set the controller callback."""
        self.controller_callback = callback

    async def initialize_click_cache(self) -> None:
        """Load historical click data from storage. Thread-safe."""
        with self._state_lock:
            if self._cache_loaded:
                return

        if not self._storage:
            return

        try:
            self.logger.info("Loading historical click data...")
            clicks_data = await self._storage.read(model_type=GridClicksData)

            with self._state_lock:
                if clicks_data.clicks:
                    self._cached_clicks = [click.model_dump() for click in clicks_data.clicks]
                    self._click_cache_timestamp = time.time()
                    self._cache_loaded = True
                    self.logger.info(f"Loaded {len(clicks_data.clicks)} clicks into cache")
                else:
                    self.logger.info("No historical click data found")
                    self._cache_loaded = True

        except Exception as e:
            self.logger.error(f"Failed to load historical clicks: {e}", exc_info=True)
            with self._state_lock:
                self._cache_loaded = True

    async def _handle_click_logged_for_cache(self, event_data: PerformMouseClickEventData) -> None:
        """Cache click data. Thread-safe."""
        click_data = {"x": event_data.x, "y": event_data.y, "timestamp": time.time(), "source": event_data.source}

        with self._state_lock:
            self._cached_clicks.append(click_data)
            self._click_cache_timestamp = time.time()

            max_cache_size = 10000
            if len(self._cached_clicks) > max_cache_size:
                self._cached_clicks = self._cached_clicks[-max_cache_size:]

            cache_size = len(self._cached_clicks)

        self.logger.debug(f"Cached click, total: {cache_size}")

    def _calculate_click_counts_sync(self, rect_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate click counts synchronously. Thread-safe."""
        processed_rects = []

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
        """Check if click is in rectangle."""
        try:
            click_x, click_y = click.get("x", 0), click.get("y", 0)
            return rect_x <= click_x <= rect_x + rect_w and rect_y <= click_y <= rect_y + rect_h
        except (TypeError, ValueError):
            return False

    def _calculate_grid_layout(self, num_rects_requested: int) -> Tuple[List[Dict[str, Any]], float, float]:
        """Calculate grid layout."""
        if num_rects_requested <= 0:
            return [], 0, 0

        screen_aspect_ratio = self.screen_width / self.screen_height
        total_screen_area = self.screen_width * self.screen_height
        target_cell_area = total_screen_area / num_rects_requested

        cell_h = math.sqrt(target_cell_area / screen_aspect_ratio)
        cell_w = cell_h * screen_aspect_ratio

        num_cols = max(1, math.floor(self.screen_width / cell_w))
        num_rows = max(1, math.floor(self.screen_height / cell_h))

        rect_w = self.screen_width / num_cols
        rect_h = self.screen_height / num_rows

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
                    "x": x,
                    "y": y,
                    "w": rect_w,
                    "h": rect_h,
                    "center_x": center_x,
                    "center_y": center_y,
                }
            )

        self.logger.info(f"Calculated grid: {num_cols}x{num_rows} = {len(rect_definitions)} cells")

        return rect_definitions, rect_w, rect_h

    def paintEvent(self, event) -> None:
        """Paint grid cells."""
        if not self._is_active:
            return

        with self._state_lock:
            rect_map = dict(self.ui_to_rect_data_map)

        if not rect_map:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for ui_number, rect_data in rect_map.items():
            x0, y0 = rect_data["x"], rect_data["y"]
            x1, y1 = x0 + rect_data["w"], y0 + rect_data["h"]

            # Draw rectangle
            painter.setPen(QPen(self.outline_color, 2))
            if self.fill_color.alpha() > 0:
                painter.setBrush(self.fill_color)
            painter.drawRect(QRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0)))

            # Draw number
            painter.setPen(QPen(self.text_color))
            painter.setFont(self.font)
            painter.drawText(int(rect_data["center_x"]), int(rect_data["center_y"]), str(ui_number))

        self.logger.debug(f"Drew {len(rect_map)} grid cells")

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
        # Ensure this runs on the main Qt thread
        QMetaObject.invokeMethod(
            self,
            "_do_show",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(int, num_rects if num_rects is not None else self.default_num_rects),
            Q_ARG(str, click_mode),
        )

    @Slot(int, str)
    def _do_show(self, num_rects: int, click_mode: str) -> None:
        """Internal show implementation - MUST run on main Qt thread."""
        if self._is_active:
            self.logger.warning("Grid already active")
            return

        try:
            self.logger.info(f"Showing grid: num_rects={num_rects}, mode={click_mode}")

            self._current_click_mode = click_mode

            # Calculate grid layout
            rect_definitions, cell_w, cell_h = self._calculate_grid_layout(num_rects)

            if not rect_definitions:
                self.logger.error("No rectangles generated")
                return

            # Calculate click counts
            rects_with_clicks = self._calculate_click_counts_sync(rect_definitions)

            # Prioritize rectangles
            weighted_rects = prioritize_grid_rects(rects_with_clicks)

            # Update map
            with self._state_lock:
                self.ui_to_rect_data_map.clear()
                for ui_number, weighted_rect_info in enumerate(weighted_rects, 1):
                    self.ui_to_rect_data_map[ui_number] = weighted_rect_info["data"]
                self.current_num_rects_displayed = len(weighted_rects)

            # Get PRIMARY screen geometry only (for mirrored/extended displays)
            # Following legacy Tkinter approach: fullscreen on primary monitor
            primary = QApplication.primaryScreen()
            if primary:
                geometry = primary.geometry()
                self.logger.info(
                    f"Primary screen geometry: x={geometry.x()}, y={geometry.y()}, w={geometry.width()}, h={geometry.height()}"
                )
                self.setGeometry(geometry)
            else:
                # Fallback geometry
                self.logger.warning("No primary screen found, using fallback geometry")
                self.setGeometry(0, 0, 1920, 1080)

            # Show window
            super().show()
            self.raise_()
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.activateWindow()  # Activate window to bring to foreground
            self.setFocus()  # Set focus immediately to capture keyboard input

            self._is_active = True

            # Schedule another focus attempt shortly after to ensure it sticks
            QTimer.singleShot(10, self._ensure_focus)

            self.logger.info(f"Grid displayed with {len(weighted_rects)} cells and focus set")

        except Exception as e:
            self.logger.error(f"Error showing grid: {e}", exc_info=True)

    def _ensure_focus(self) -> None:
        """Ensure focus is maintained after show."""
        if self._is_active and not self.isHidden():
            self.setFocus()
            self.logger.debug("Focus re-asserted on grid")

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

            self.clearFocus()
            super().hide()
            self._is_active = False

            with self._state_lock:
                self.ui_to_rect_data_map.clear()
                self.current_num_rects_displayed = None

            self.logger.info("Grid hidden")

        except Exception as e:
            self.logger.error(f"Error hiding grid: {e}", exc_info=True)

    def refresh_display(self) -> None:
        """Refresh the grid display."""
        if self._is_active and self.current_num_rects_displayed:
            self.update()

    def handle_selection(self, selection_key: str, click_mode: str = "click") -> bool:
        """Handle grid cell selection."""
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

        center_x = rect_data["center_x"]
        center_y = rect_data["center_y"]

        try:
            if click_mode == "click":
                pyautogui.click(center_x, center_y)
            elif click_mode == "hover":
                pyautogui.moveTo(center_x, center_y)

            self.logger.info(f"Grid cell {selected_number} selected at ({center_x}, {center_y})")

            if self.controller_callback:
                self.controller_callback.on_grid_selection_success(selected_number, center_x, center_y)

            self.hide()

            return True

        except Exception as e:
            self.logger.error(f"Error handling selection: {e}", exc_info=True)

            if self.controller_callback:
                self.controller_callback.on_grid_selection_failed(selected_number, str(e))

            return False

    def cleanup(self) -> None:
        """Clean up resources."""
        self.hide()
        self.event_bus.unsubscribe(PerformMouseClickEventData, self._handle_click_logged_for_cache)

        with self._state_lock:
            self._cached_clicks.clear()
            self.ui_to_rect_data_map.clear()
            self._is_active = False

        self.logger.debug("QtGridView cleanup completed")
