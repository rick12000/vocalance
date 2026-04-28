import logging
import threading
from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QPainter, QPainterPath, QPaintEvent, QPalette
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from vocalance.app.config.app_config import AssetPathsConfig
from vocalance.app.lifecycle.lifecycle import AppLifecycle
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.qt_assets import QtAssetCache
from vocalance.app.ui.utils.qt_logo_service import QtLogoService
from vocalance.app.ui.utils.window_icon_manager import WindowIconManager


class StartupSignals(QObject):
    """Qt signals for cross-thread startup UI updates."""

    progress_updated = Signal(float, str, bool)
    close_requested = Signal()


class StartupWindow(QDialog):
    """Frameless startup dialog with progress and optional spinner."""

    def __init__(
        self,
        logger: logging.Logger,
        asset_paths_config: AssetPathsConfig,
        lifecycle: Optional[AppLifecycle] = None,
        icon_manager: Optional[WindowIconManager] = None,
    ) -> None:
        super().__init__()

        self.logger = logger
        self.lifecycle = lifecycle
        self.icon_manager = icon_manager
        self.is_closed = False
        self._state_lock = threading.Lock()
        self._closing_after_init = False

        self.is_animating = False
        self.animation_base_text = ""
        self.spinner_animation = None
        self.animation_timer = None
        self.animation_frame = 0
        self.animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        self.asset_cache = QtAssetCache(asset_paths_config=asset_paths_config)
        self.logo_service = QtLogoService(self.asset_cache)

        self.signals = StartupSignals()
        self.signals.progress_updated.connect(self._apply_progress_update)
        self.signals.close_requested.connect(self._finalize_close)

        self._build_widgets()

        if self.icon_manager and self.icon_manager.is_icon_loaded():
            self.icon_manager.apply_to_dialog(self)

        self.logger.debug("Startup window initialized")

    def _build_widgets(self) -> None:
        self.setWindowTitle("Vocalance")
        self.setFixedSize(
            theme.config.components.startup_width,
            theme.config.components.startup_height,
        )
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.border_radius = theme.config.radius.rounded

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.darkest))
        self.setPalette(palette)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large
        )
        main_layout.setSpacing(theme.config.spacing.small)

        self.logo_label = self.logo_service.create_logo_widget(
            self,
            max_size=theme.config.components.startup_logo_size,
            context="startup",
            text_fallback="VOCALANCE",
            logo_type="full",
        )
        main_layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("StartupProgressBar")
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(2)
        self.progress_bar.setFixedWidth(460)

        progress_container = QWidget(self)
        progress_layout = QHBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.addStretch()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addStretch()
        main_layout.addWidget(progress_container)

        status_outer_container = QWidget(self)
        status_outer_layout = QHBoxLayout(status_outer_container)
        status_outer_layout.setContentsMargins(0, 0, 0, 0)
        status_outer_layout.setSpacing(0)

        status_outer_layout.addStretch()

        status_container = QWidget(self)
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(theme.config.spacing.tiny)

        self.text_label = QLabel("Starting up", self)
        font = theme.get_font(size=theme.config.fonts.small)
        self.text_label.setFont(font)
        palette = self.text_label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.shapes.light))
        self.text_label.setPalette(palette)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.text_label)

        self.spinner_label = QLabel(self.animation_frames[0], self)
        spinner_font = theme.get_font(size=theme.config.fonts.large)
        self.spinner_label.setFont(spinner_font)
        spinner_palette = self.spinner_label.palette()
        spinner_palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.blue.blue_2))
        self.spinner_label.setPalette(spinner_palette)
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.spinner_label.setFixedWidth(30)
        status_layout.addWidget(self.spinner_label)

        status_outer_layout.addWidget(status_container)

        status_outer_layout.addStretch()

        main_layout.addWidget(status_outer_container)

        main_layout.addStretch()

        self._center_on_primary_screen()

    def _center_on_primary_screen(self) -> None:
        screen = self.screen()
        if screen:
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 2
            self.move(x, y)

    def paintEvent(self, paint_event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self.border_radius, self.border_radius)

        painter.fillPath(path, QColor(theme.config.shapes.darkest))

    def show(self) -> None:
        """Show and activate the dialog if it has not been closed."""
        if not self.is_closed:
            super().show()
            self.raise_()
            self.activateWindow()
            self.logger.debug("Startup window displayed")

    def update_progress(self, progress: float, status: str, animate: bool = False) -> None:
        """Queue a thread-safe progress update (``progress`` in ``0.0``–``1.0``)."""
        self.signals.progress_updated.emit(progress, status, animate)

    def _apply_progress_update(self, progress: float, status: str, animate: bool) -> None:
        with self._state_lock:
            if self.is_closed:
                return
            try:
                progress_value = int(progress * 100)
                progress_value = max(0, min(100, progress_value))
                self.progress_bar.setValue(progress_value)

                if self.text_label and self.spinner_label and status:
                    self.text_label.setText(status.rstrip("."))

                    if animate:
                        self._start_spinner()
                    else:
                        self._stop_spinner()

            except RuntimeError:
                self.logger.debug("Startup progress update skipped (widget teardown)")

    def _start_spinner(self) -> None:
        if self.is_closed or self.is_animating:
            return

        self.is_animating = True
        self.animation_frame = 0

        if self.spinner_label:
            self.animation_timer = QTimer(self)
            self.animation_timer.timeout.connect(self._advance_spinner_frame)
            self.animation_timer.start(80)

    def _advance_spinner_frame(self) -> None:
        if not self.is_animating or self.is_closed or not self.spinner_label:
            return

        try:
            self.spinner_label.setText(self.animation_frames[self.animation_frame])
            self.animation_frame = (self.animation_frame + 1) % len(self.animation_frames)
        except RuntimeError:
            self.is_animating = False

    def _stop_spinner(self) -> None:
        if not self.is_animating:
            return

        self.is_animating = False
        self.animation_base_text = ""

        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None

        if self.spinner_label:
            self.spinner_label.setText(self.animation_frames[0])

    def _finalize_close(self) -> None:
        with self._state_lock:
            if self.is_closed:
                return
            try:
                self._stop_spinner()
                self.is_closed = True
                self.accept()

                if not self._closing_after_init and self.lifecycle is not None and not self.lifecycle.is_shutdown_requested():
                    self.lifecycle.request_shutdown(
                        reason="User closed startup window",
                        source="startup_window",
                    )

            except RuntimeError:
                self.logger.debug("Startup close skipped (widget teardown)")

    def close(self) -> None:
        """Request a thread-safe close (user gesture path)."""
        self.signals.close_requested.emit()

    def close_after_initialization(self) -> None:
        """Close after successful init without requesting application shutdown."""
        self._closing_after_init = True
        self.close()

    def is_visible(self) -> bool:
        """Return False once the window has been closed."""
        with self._state_lock:
            return not self.is_closed

    def closeEvent(self, close_event: QCloseEvent) -> None:
        self._finalize_close()
        close_event.accept()


class StartupProgressTracker:
    """Maps discrete startup steps to ``StartupWindow`` progress."""

    def __init__(self, startup_window: StartupWindow, total_steps: int) -> None:
        self.startup_window = startup_window
        self.total_steps = total_steps
        self.current_step = 0
        self.current_step_name = ""
        self.sub_step_progress = 0.0
        self._tracker_lock = threading.Lock()

    def start_step(self, step_name: str) -> None:
        with self._tracker_lock:
            self.current_step += 1
            self.current_step_name = step_name
            self.sub_step_progress = 0.0
        self._push_progress_to_window(step_name, animate=True)

    def update_sub_step(self, sub_step_name: str, progress: float = 0.5) -> None:
        with self._tracker_lock:
            self.sub_step_progress = max(0.0, min(1.0, progress))
        self._push_progress_to_window(sub_step_name, animate=True)

    def update_status_animated(self, status: str, progress: float = 0.5) -> None:
        with self._tracker_lock:
            self.sub_step_progress = max(0.0, min(1.0, progress))
        self._push_progress_to_window(status, animate=True)

    def update_status_static(self, status: str, progress: float = 0.5) -> None:
        with self._tracker_lock:
            self.sub_step_progress = max(0.0, min(1.0, progress))
        self._push_progress_to_window(status, animate=False)

    def complete_step(self, step_name: str = "") -> None:
        with self._tracker_lock:
            self.sub_step_progress = 1.0
        message = step_name or f"{self.current_step_name} completed"
        self._push_progress_to_window(message, animate=True)

    def _push_progress_to_window(self, status: str, animate: bool = False) -> None:
        with self._tracker_lock:
            if self.total_steps > 0:
                base_progress = (self.current_step - 1) / self.total_steps
                step_contribution = self.sub_step_progress / self.total_steps
                progress = base_progress + step_contribution
            else:
                progress = 0.0

            progress = min(1.0, progress)

        self.startup_window.update_progress(progress, status, animate=animate)

    def finish(self) -> None:
        self.startup_window.update_progress(1.0, "Ready!", animate=False)
