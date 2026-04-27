import asyncio
import logging
import math
import threading
import time
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import pyautogui
from PySide6.QtCore import QMetaObject, QRect, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QWidget

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import PerformMouseClickEventData
from vocalance.app.events.grid_events import GridStateEvent
from vocalance.app.lifecycle import schedule_on_loop
from vocalance.app.services.grid.click_tracker_service import prioritize_grid_rects, rects_with_click_counts
from vocalance.app.ui.qt_theme import theme

GRID_DRAG_MOVE_MIN_S = 0.22
GRID_DRAG_MOVE_MAX_S = 0.85
GRID_DRAG_MOVE_DIST_DIVISOR = 2200.0
GRID_DRAG_SETTLE_S = 0.05


class QtGridView(QWidget):
    DEFAULT_NUM_RECTS = 500

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        gui_event_loop: asyncio.AbstractEventLoop,
        input_service: Any,
        default_num_rects: Optional[int] = None,
    ):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        self.config = config
        self.gui_loop = gui_event_loop
        self.input_service = input_service

        self.default_num_rects = default_num_rects or self.DEFAULT_NUM_RECTS

        self.state_lock = threading.RLock()
        self.clicks_snapshot: List[Dict[str, Any]] = []
        self.pending_clicks_snapshot: Optional[List[Dict[str, Any]]] = None

        self.overlay_active = False
        self.current_num_rects_displayed: Optional[int] = None
        self.ui_to_rect_data_map: Dict[int, Dict[str, Any]] = {}
        self.current_click_mode: str = "click"
        self.drag_origin: Optional[Tuple[int, int]] = None
        self.layout_num_rects_requested: Optional[int] = None

        self.cached_num_cols: int = 0
        self.cached_num_rows: int = 0
        self.cached_cell_w: float = 0
        self.cached_cell_h: float = 0

        self.overlay_preparing = False
        self.layout_device_pixel_ratio: float = 1.0

        self.focus_timers: List[QTimer] = []

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.background_color = QColor(theme.config.shapes.dark)
        self.background_color.setAlpha(180)

        self.grid_line_color = QColor(theme.config.shapes.medium)
        self.grid_line_color.setAlpha(255)

        self.text_color = QColor(theme.config.blue.blue_2)
        self.text_color.setAlpha(255)

        self.font = QFont(theme.config.font_family_primary, theme.config.fonts.small, QFont.Weight.Bold)

        self.current_font_size = theme.config.fonts.small
        self.min_font_size = 7
        self.max_font_size = theme.config.fonts.large
        self.default_font_size = theme.config.fonts.small

        self.screen_width = 1920
        self.screen_height = 1080

    def set_clicks_snapshot(self, clicks: List[Dict[str, Any]]) -> None:
        self.pending_clicks_snapshot = [dict(c) for c in clicks]
        QMetaObject.invokeMethod(self, "apply_clicks_snapshot", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def apply_clicks_snapshot(self) -> None:
        pending = self.pending_clicks_snapshot
        if pending is None:
            return
        self.pending_clicks_snapshot = None
        with self.state_lock:
            self.clicks_snapshot = pending
        self.refresh_click_labels_if_active()

    def calculate_click_counts_sync(self, rect_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self.state_lock:
            snap = list(self.clicks_snapshot)
        return rects_with_click_counts(rect_definitions, snap)

    def schedule_bus_coroutine(self, coro: Coroutine[Any, Any, Any]) -> None:
        async def _log_and_run() -> None:
            try:
                await coro
            except Exception as exc:
                self.logger.error("Grid overlay bus publish failed", exc_info=exc)

        schedule_on_loop(self.gui_loop, _log_and_run())

    def calculate_adaptive_font_size(self, num_rects: int) -> int:
        if num_rects <= 0:
            return self.default_font_size

        reference_rects = 100
        scale_factor = math.sqrt(reference_rects / num_rects)
        calculated_size = int(self.default_font_size * scale_factor)

        clamped_size = max(self.min_font_size, min(self.max_font_size, calculated_size))

        self.logger.debug(
            f"Font size calculation: {num_rects} cells -> scale={scale_factor:.2f} -> {calculated_size}pt -> clamped={clamped_size}pt"
        )

        return clamped_size

    def calculate_grid_layout(self, num_rects_requested: int) -> Tuple[List[Dict[str, Any]], float, float]:
        if num_rects_requested <= 0:
            return [], 0, 0

        screen_aspect_ratio = self.screen_width / self.screen_height
        total_screen_area = self.screen_width * self.screen_height
        target_cell_area = total_screen_area / num_rects_requested

        ideal_cell_h = math.sqrt(target_cell_area / screen_aspect_ratio)
        ideal_cell_w = ideal_cell_h * screen_aspect_ratio

        num_cols = max(1, round(self.screen_width / ideal_cell_w))
        num_rows = max(1, round(self.screen_height / ideal_cell_h))

        rect_w = self.screen_width / num_cols
        rect_h = self.screen_height / num_rows

        actual_cells_to_create = num_cols * num_rows

        rect_definitions = []
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
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

        self.logger.debug(
            "Grid layout: requested=%s, actual=%s (%sx%s), cell_size=%.1fx%.1f",
            num_rects_requested,
            actual_cells_to_create,
            num_cols,
            num_rows,
            rect_w,
            rect_h,
        )

        return rect_definitions, rect_w, rect_h

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        if self.overlay_preparing:
            return

        if not self.overlay_active:
            return

        with self.state_lock:
            rect_map = dict(self.ui_to_rect_data_map)
            num_cols = self.cached_num_cols
            num_rows = self.cached_num_rows
            cell_w = self.cached_cell_w
            cell_h = self.cached_cell_h

        if not rect_map or num_cols == 0 or num_rows == 0 or cell_w == 0 or cell_h == 0:
            return

        dpr = self.layout_device_pixel_ratio

        window_width = self.width()
        window_height = self.height()

        min_x = 0
        min_y = 0
        max_x = window_width
        max_y = window_height

        painter = QPainter(self)

        painter.fillRect(self.rect(), self.background_color)

        top_n_cells = min(50, len(rect_map))
        for ui_number in range(1, top_n_cells + 1):
            if ui_number not in rect_map:
                continue

            rect_data = rect_map[ui_number]

            raw_x = rect_data["x"] / dpr
            raw_y = rect_data["y"] / dpr
            col = round((raw_x - min_x) / cell_w)
            row = round((raw_y - min_y) / cell_h)
            cell_x = min_x + col * cell_w
            cell_y = min_y + row * cell_h

            base_color = QColor(theme.config.blue.blue_2)

            alpha = int(220 - ((ui_number - 1) / (top_n_cells - 1) * 160)) if top_n_cells > 1 else 220

            intensity_factor = 1.0 - ((ui_number - 1) / (top_n_cells - 1) * 0.4) if top_n_cells > 1 else 1.0

            r = int(base_color.red() * intensity_factor)
            g = int(base_color.green() * intensity_factor)
            b = int(base_color.blue() * intensity_factor)

            fill_color = QColor(r, g, b, alpha)

            cell_rect = QRect(cell_x, cell_y, cell_w, cell_h)
            painter.fillRect(cell_rect, fill_color)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        grid_pen = QPen(self.grid_line_color, 1, Qt.PenStyle.SolidLine)
        painter.setPen(grid_pen)

        for col in range(num_cols + 1):
            x = min_x + col * cell_w
            painter.drawLine(x, min_y, x, max_y)

        for row in range(num_rows + 1):
            y = min_y + row * cell_h
            painter.drawLine(min_x, y, max_x, y)

        painter.setPen(QPen(self.text_color))
        painter.setFont(self.font)

        for ui_number, rect_data in rect_map.items():
            raw_x = rect_data["x"] / dpr
            raw_y = rect_data["y"] / dpr

            col = round((raw_x - min_x) / cell_w)
            row = round((raw_y - min_y) / cell_h)

            cell_x = min_x + col * cell_w
            cell_y = min_y + row * cell_h

            cell_rect = QRect(cell_x, cell_y, cell_w, cell_h)
            painter.drawText(cell_rect, Qt.AlignmentFlag.AlignCenter, str(ui_number))

        self.logger.debug("Drew grid: %sx%s = %s cells", num_cols, num_rows, len(rect_map))

    def keyPressEvent(self, key_event: QKeyEvent) -> None:
        """Handle key press events."""
        if key_event.key() == Qt.Key.Key_Escape:
            # Schedule hide on next event loop iteration to avoid blocking
            QTimer.singleShot(0, self.do_hide)
            key_event.accept()
        elif key_event.key() >= Qt.Key.Key_0 and key_event.key() <= Qt.Key.Key_9:
            digit = key_event.key() - Qt.Key.Key_0
            # Schedule number input handling to avoid blocking
            QTimer.singleShot(0, lambda: self.handle_number_input(digit))
            key_event.accept()
        else:
            super().keyPressEvent(key_event)

    def handle_number_input(self, digit: int) -> None:
        """Handle number input for grid cell selection."""
        if digit in self.ui_to_rect_data_map:
            self.handle_selection(str(digit), self.current_click_mode)

    def is_active(self) -> bool:
        """Check if overlay is active."""
        return self.overlay_active and self.isVisible()

    @Slot()
    def show(self, num_rects: Optional[int] = None, click_mode: str = "click") -> None:
        """Show the grid overlay. Callers are expected to be on the GUI thread."""
        self.do_show(num_rects if num_rects is not None else self.default_num_rects, click_mode)

    @Slot(int, str)
    def do_show(self, num_rects: int, click_mode: str) -> None:
        """Internal show implementation - MUST run on main Qt thread."""
        with self.state_lock:
            if self.overlay_active or self.overlay_preparing:
                self.logger.warning("Grid already active or preparing - ignoring show request")
                return
            self.overlay_preparing = True

        try:
            start_time = time.perf_counter()
            self.logger.debug("Showing grid: num_rects=%s, mode=%s", num_rects, click_mode)

            with self.state_lock:
                self.current_click_mode = click_mode
                if click_mode == "drag":
                    pos = pyautogui.position()
                    self.drag_origin = (round(pos[0]), round(pos[1]))
                else:
                    self.drag_origin = None

            # Get PRIMARY screen for physical pixel calculations
            primary = QApplication.primaryScreen()
            if primary:
                geometry = primary.geometry()
                device_pixel_ratio = primary.devicePixelRatio()
                self.layout_device_pixel_ratio = float(device_pixel_ratio)

                # Calculate PHYSICAL pixels (what pyautogui uses)
                # Qt geometry() returns logical pixels, multiply by DPR for physical
                self.screen_width = int(geometry.width() * device_pixel_ratio)
                self.screen_height = int(geometry.height() * device_pixel_ratio)

                self.logger.debug(
                    "Screen logical=%sx%s dpr=%s physical=%sx%s",
                    geometry.width(),
                    geometry.height(),
                    device_pixel_ratio,
                    self.screen_width,
                    self.screen_height,
                )

                # OPTIMIZATION: Set window geometry IMMEDIATELY for instant display
                self.setGeometry(geometry)

                logical_width = geometry.width()
                logical_height = geometry.height()
            else:
                self.logger.warning("No primary screen found, using fallback geometry")
                self.layout_device_pixel_ratio = 1.0
                self.screen_width = 1920
                self.screen_height = 1080
                device_pixel_ratio = 1.0
                self.setGeometry(0, 0, 1920, 1080)

                logical_width = 1920
                logical_height = 1080

            # Calculate grid layout using PHYSICAL pixels
            t1 = time.perf_counter()
            rect_definitions, cell_w, cell_h = self.calculate_grid_layout(num_rects)
            t2 = time.perf_counter()
            self.logger.debug(f"Grid layout calculation: {(t2 - t1) * 1000:.1f}ms")

            if not rect_definitions:
                self.logger.error("No rectangles generated")
                with self.state_lock:
                    self.drag_origin = None
                return

            self.logger.debug(f"Calculating click counts for {len(rect_definitions)} cells")
            rects_with_clicks = self.calculate_click_counts_sync(rect_definitions)
            t3 = time.perf_counter()
            self.logger.debug(f"Click counting: {(t3 - t2) * 1000:.1f}ms")

            weighted_rects = prioritize_grid_rects(rects_with_clicks)
            t4 = time.perf_counter()
            self.logger.debug(f"Prioritization: {(t4 - t3) * 1000:.1f}ms")

            logical_cell_w = cell_w / device_pixel_ratio
            logical_cell_h = cell_h / device_pixel_ratio
            num_cols = max(1, round(logical_width / logical_cell_w))
            num_rows = max(1, round(logical_height / logical_cell_h))

            logical_cell_w = logical_width / num_cols
            logical_cell_h = logical_height / num_rows

            self.current_font_size = self.calculate_adaptive_font_size(len(weighted_rects))
            self.font.setPointSize(self.current_font_size)

            self.setUpdatesEnabled(False)

            with self.state_lock:
                self.ui_to_rect_data_map.clear()
                for ui_number, weighted_rect_info in enumerate(weighted_rects, 1):
                    self.ui_to_rect_data_map[ui_number] = weighted_rect_info["data"]
                self.current_num_rects_displayed = len(weighted_rects)

                self.cached_num_cols = num_cols
                self.cached_num_rows = num_rows
                self.cached_cell_w = logical_cell_w
                self.cached_cell_h = logical_cell_h
                self.layout_num_rects_requested = num_rects

                self.overlay_active = True

            super().show()
            self.raise_()
            self.activateWindow()
            self.setFocus(Qt.FocusReason.PopupFocusReason)

            self.setUpdatesEnabled(True)
            self.update()

            # Schedule deferred focus attempts to overcome Windows focus stealing
            self.schedule_robust_focus()

            end_time = time.perf_counter()
            total_ms = (end_time - start_time) * 1000
            self.logger.debug("Grid displayed with %s cells in %.1fms", len(weighted_rects), total_ms)

        finally:
            with self.state_lock:
                self.overlay_preparing = False

    def schedule_robust_focus(self) -> None:
        """Schedule deferred focus attempts to overcome Windows focus stealing."""
        self.cancel_focus_timers()

        # Critical delays to overcome taskbar interference
        delays = [50, 200]

        for delay in delays:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.ensure_focus)
            timer.start(delay)
            self.focus_timers.append(timer)

    def cancel_focus_timers(self) -> None:
        """Cancel all pending focus timers."""
        for timer in self.focus_timers:
            if timer.isActive():
                timer.stop()
            timer.deleteLater()
        self.focus_timers.clear()

    def ensure_focus(self) -> None:
        """Reinforce focus and window stacking."""
        if self.overlay_active and not self.isHidden():
            self.raise_()
            self.activateWindow()
            self.setFocus(Qt.FocusReason.PopupFocusReason)

    @Slot()
    def hide(self) -> None:
        """Hide the grid overlay - thread-safe via QMetaObject.invokeMethod."""
        # Ensure this runs on the main Qt thread
        QMetaObject.invokeMethod(self, "do_hide", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def do_hide(self) -> None:
        """Internal hide implementation - MUST run on main Qt thread."""
        if not self.overlay_active:
            return

        self.logger.debug("Hiding grid")

        self.cancel_focus_timers()
        self.overlay_preparing = False

        self.clearFocus()
        super().hide()
        self.overlay_active = False

        with self.state_lock:
            self.ui_to_rect_data_map.clear()
            self.current_num_rects_displayed = None
            self.cached_num_cols = 0
            self.cached_num_rows = 0
            self.cached_cell_w = 0
            self.cached_cell_h = 0
            self.drag_origin = None
            self.layout_num_rects_requested = None

        self.logger.debug("Grid hidden")

    def refresh_click_labels_if_active(self) -> None:
        """Recompute cell numbering from click history while the overlay stays visible."""
        with self.state_lock:
            if not self.overlay_active:
                return
            num_rects = self.layout_num_rects_requested
        if num_rects is None:
            return

        primary = QApplication.primaryScreen()
        if primary:
            geometry = primary.geometry()
            device_pixel_ratio = primary.devicePixelRatio()
            self.layout_device_pixel_ratio = float(device_pixel_ratio)
            self.screen_width = int(geometry.width() * device_pixel_ratio)
            self.screen_height = int(geometry.height() * device_pixel_ratio)
            logical_width = geometry.width()
            logical_height = geometry.height()
        else:
            self.layout_device_pixel_ratio = 1.0
            self.screen_width = 1920
            self.screen_height = 1080
            device_pixel_ratio = 1.0
            logical_width = 1920
            logical_height = 1080

        rect_definitions, cell_w, cell_h = self.calculate_grid_layout(num_rects)
        if not rect_definitions:
            return

        rects_with_clicks = self.calculate_click_counts_sync(rect_definitions)
        weighted_rects = prioritize_grid_rects(rects_with_clicks)

        logical_cell_w = cell_w / device_pixel_ratio
        logical_cell_h = cell_h / device_pixel_ratio
        num_cols = max(1, round(logical_width / logical_cell_w))
        num_rows = max(1, round(logical_height / logical_cell_h))
        logical_cell_w = logical_width / num_cols
        logical_cell_h = logical_height / num_rows

        self.current_font_size = self.calculate_adaptive_font_size(len(weighted_rects))
        self.font.setPointSize(self.current_font_size)

        with self.state_lock:
            if not self.overlay_active:
                return
            self.ui_to_rect_data_map.clear()
            for ui_number, weighted_rect_info in enumerate(weighted_rects, 1):
                self.ui_to_rect_data_map[ui_number] = weighted_rect_info["data"]
            self.current_num_rects_displayed = len(weighted_rects)
            self.cached_num_cols = num_cols
            self.cached_num_rows = num_rows
            self.cached_cell_w = logical_cell_w
            self.cached_cell_h = logical_cell_h

        self.update()

    def publish_grid_pointer_recorded(self, x: int, y: int) -> None:
        self.schedule_bus_coroutine(self.event_bus.publish(PerformMouseClickEventData(x=x, y=y, source="grid_overlay")))

    def execute_grid_pointer_action(
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
            self.publish_grid_pointer_recorded(cx, cy)
        elif click_mode == "hover":
            pyautogui.moveTo(cx, cy)
        elif click_mode == "drag":
            if drag_origin is None:
                raise RuntimeError("Drag mode requires a recorded start position")
            ox, oy = drag_origin
            tx, ty = round(center_x), round(center_y)
            dist = math.hypot(float(tx - ox), float(ty - oy))
            duration = min(
                GRID_DRAG_MOVE_MAX_S,
                max(GRID_DRAG_MOVE_MIN_S, dist / GRID_DRAG_MOVE_DIST_DIVISOR),
            )
            pyautogui.moveTo(ox, oy, duration=0.0, _pause=False)
            pyautogui.mouseDown(ox, oy, button="left", _pause=False)
            try:
                pyautogui.moveTo(tx, ty, duration=duration, _pause=False)
                time.sleep(GRID_DRAG_SETTLE_S)
            finally:
                pyautogui.mouseUp(tx, ty, button="left", _pause=False)
            self.publish_grid_pointer_recorded(tx, ty)
        else:
            raise ValueError(f"Unsupported grid click_mode: {click_mode!r}")

    def execute_delayed_grid_action(
        self,
        selected_number: int,
        center_x: float,
        center_y: float,
        click_mode: str,
        drag_origin: Optional[Tuple[int, int]],
    ) -> None:
        time.sleep(0.05)
        self.execute_grid_pointer_action(click_mode, center_x, center_y, drag_origin)
        self.logger.debug("Action '%s' performed at (%s, %s)", click_mode, center_x, center_y)
        self.schedule_bus_coroutine(self.publish_grid_interaction_success(selected_number, center_x, center_y))

    async def publish_grid_interaction_success(self, selected_number: int, center_x: float, center_y: float) -> None:
        await self.event_bus.publish(
            GridStateEvent(
                state="interaction_success",
                config={
                    "selected_number": selected_number,
                    "center_x": center_x,
                    "center_y": center_y,
                },
            )
        )

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

        with self.state_lock:
            if selected_number not in self.ui_to_rect_data_map:
                self.logger.warning(f"Selection {selected_number} not in grid")
                return False

            rect_data = self.ui_to_rect_data_map[selected_number]
            drag_origin = self.drag_origin if click_mode == "drag" else None

        if click_mode == "drag" and drag_origin is None:
            self.logger.error("Drag mode selection failed: no recorded start position")
            return False

        # Coordinates are already in physical pixels (matching pyautogui)
        center_x = rect_data["center_x"]
        center_y = rect_data["center_y"]

        self.logger.debug("Grid cell %s selected at physical coords (%s, %s)", selected_number, center_x, center_y)

        # CRITICAL: Hide the overlay FIRST, then schedule the click after a delay
        # This ensures the overlay is fully gone before we click on the screen
        self.hide()

        async def _run_action() -> None:
            try:
                await self.input_service.run(
                    self.execute_delayed_grid_action,
                    selected_number,
                    center_x,
                    center_y,
                    click_mode,
                    drag_origin,
                )
            except Exception as exc:
                self.logger.error("Grid input action failed", exc_info=exc)

        asyncio.create_task(_run_action())

        return True

    def shutdown(self) -> None:
        """Cancel pending timers, hide the overlay, and clear cached state."""
        self.cancel_focus_timers()
        self.hide()

        with self.state_lock:
            self.ui_to_rect_data_map.clear()
            self.overlay_active = False
            self.layout_num_rects_requested = None
            self.clicks_snapshot = []
            self.pending_clicks_snapshot = None

        self.logger.debug("QtGridView shutdown completed")
