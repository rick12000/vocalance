"""Qt-based startup window with progress tracking.

Provides a modern startup/splash screen using QDialog with thread-safe
progress updates via Qt signals.
"""

import logging
import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.qt_assets import QtAssetCache
from vocalance.app.ui.utils.qt_logo_service import QtLogoService


class StartupSignals(QObject):
    """Signal container for thread-safe startup window updates."""

    progress_updated = Signal(float, str, bool)  # progress, status, animate
    close_requested = Signal()


class StartupWindow(QDialog):
    """Thread-safe startup window with progress bar and spinner animation.

    Displays application logo, progress bar, and status text during initialization.
    Uses Qt signals for thread-safe updates from any thread.

    Attributes:
        signals: StartupSignals instance for cross-thread communication.
        progress_bar: QProgressBar widget for visual progress.
        is_closed: Whether window has been closed.
    """

    def __init__(
        self,
        logger: logging.Logger,
        asset_paths_config,
        shutdown_coordinator=None,
    ):
        """Initialize startup window.

        Args:
            logger: Logger instance.
            asset_paths_config: Asset paths configuration.
            shutdown_coordinator: Optional shutdown coordinator reference.
        """
        super().__init__()

        self.logger = logger
        self.shutdown_coordinator = shutdown_coordinator
        self.is_closed = False
        self._lock = threading.Lock()
        self._programmatic_close = False

        # Animation state
        self.is_animating = False
        self.animation_base_text = ""
        self.spinner_animation = None
        self.animation_timer = None
        self.animation_frame = 0
        # Spinner animation frames - rotating dots
        self.animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        # Asset services
        self.asset_cache = QtAssetCache(asset_paths_config=asset_paths_config)
        self.logo_service = QtLogoService(self.asset_cache)

        # Setup signals for thread-safe updates
        self.signals = StartupSignals()
        self.signals.progress_updated.connect(self._update_progress_impl)
        self.signals.close_requested.connect(self._close_impl)

        # Setup UI
        self._setup_ui()

        self.logger.info("Startup window initialized")

    def _setup_ui(self) -> None:
        """Build UI components."""
        # Window configuration
        self.setWindowTitle("Vocalance")
        self.setFixedSize(
            theme.config.components.startup_width,
            theme.config.components.startup_height,
        )
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        # Enable transparency for rounded corners effect
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Store border radius for painting
        self.border_radius = theme.config.radius.rounded

        # Apply theme colors programmatically (for content inside rounded box)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.darkest))
        self.setPalette(palette)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large
        )
        main_layout.setSpacing(theme.config.spacing.small)  # Reduced spacing between progress bar and status text

        # Logo
        self.logo_label = self.logo_service.create_logo_widget(
            self,
            max_size=theme.config.components.startup_logo_size,
            context="startup",
            text_fallback="VOCALANCE",
            logo_type="full",
        )
        main_layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Progress bar - minimal, modern aesthetic
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(2)  # Very thin bar
        self.progress_bar.setFixedWidth(460)  # Fixed width to prevent horizontal shifts

        # Style progress bar with theme colors
        progress_stylesheet = f"""
        QProgressBar {{
            background-color: {theme.config.shapes.dark};
            border: none;
            border-radius: 1px;
        }}
        QProgressBar::chunk {{
            background-color: {theme.config.blue.blue_2};
            border-radius: 1px;
        }}
        """
        self.progress_bar.setStyleSheet(progress_stylesheet)

        # Center the progress bar horizontally
        progress_container = QWidget(self)
        progress_layout = QHBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.addStretch()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addStretch()
        main_layout.addWidget(progress_container)

        # Status container (text + spinner) - truly centered
        status_outer_container = QWidget(self)
        status_outer_layout = QHBoxLayout(status_outer_container)
        status_outer_layout.setContentsMargins(0, 0, 0, 0)
        status_outer_layout.setSpacing(0)

        # Add stretch before
        status_outer_layout.addStretch()

        # Inner container - no fixed width, content determines size
        status_container = QWidget(self)
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(theme.config.spacing.tiny)  # Minimal spacing between text and spinner

        # Status text - centered alignment
        self.text_label = QLabel("Starting up", self)
        font = theme.get_font(size=theme.config.fonts.small)
        self.text_label.setFont(font)
        palette = self.text_label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.shapes.light))
        self.text_label.setPalette(palette)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        # Let text size naturally, no minimum width constraint
        status_layout.addWidget(self.text_label)

        # Spinner - using rotating braille animation frames
        self.spinner_label = QLabel(self.animation_frames[0], self)  # Start with first frame visible
        spinner_font = theme.get_font(size=theme.config.fonts.large)
        self.spinner_label.setFont(spinner_font)
        spinner_palette = self.spinner_label.palette()
        spinner_palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.blue.blue_2))
        self.spinner_label.setPalette(spinner_palette)
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.spinner_label.setFixedWidth(30)  # Fixed width to prevent shifts
        status_layout.addWidget(self.spinner_label)

        status_outer_layout.addWidget(status_container)

        # Add stretch after
        status_outer_layout.addStretch()

        main_layout.addWidget(status_outer_container)

        # Add vertical stretch
        main_layout.addStretch()

        # Center window on screen
        self._center_window()

    def _center_window(self) -> None:
        """Center window on screen."""
        screen = self.screen()
        if screen:
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 2
            self.move(x, y)

    def paintEvent(self, event) -> None:
        """Paint rounded rectangle background for transparent window effect."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self.border_radius, self.border_radius)

        # Fill with background color
        painter.fillPath(path, QColor(theme.config.shapes.darkest))

        # Don't call super().paintEvent() to avoid double painting
        # The layout will handle drawing the content

    def show(self) -> None:
        """Display the startup window."""
        if not self.is_closed:
            super().show()
            self.raise_()
            self.activateWindow()
            self.logger.info("Startup window displayed")

    def update_progress(self, progress: float, status: str, animate: bool = False) -> None:
        """Update progress - thread-safe.

        Can be called from any thread. Updates are queued via signals.

        Args:
            progress: Progress value (0.0 to 1.0).
            status: Status text to display.
            animate: Whether to show animated spinner.
        """
        self.signals.progress_updated.emit(progress, status, animate)

    def _update_progress_impl(self, progress: float, status: str, animate: bool) -> None:
        """Update progress bar and status text (runs in main thread).

        Args:
            progress: Progress value (0.0 to 1.0).
            status: Status text to display.
            animate: Whether to show animated spinner.
        """
        with self._lock:
            if self.is_closed:
                return

            try:
                # Update progress bar
                progress_value = int(progress * 100)
                progress_value = max(0, min(100, progress_value))
                self.progress_bar.setValue(progress_value)

                # Update status text
                if self.text_label and self.spinner_label and status:
                    self.text_label.setText(status.rstrip("."))

                    if animate:
                        self._start_animation()
                    else:
                        self._stop_animation()

            except Exception as e:
                self.logger.error(f"Error updating progress: {e}")

    def _start_animation(self) -> None:
        """Begin spinner animation using rotating braille frames."""
        if self.is_closed or self.is_animating:
            return

        self.is_animating = True
        self.animation_frame = 0

        # Create timer-based animation for rotating spinner
        if self.spinner_label:
            self.animation_timer = QTimer(self)
            self.animation_timer.timeout.connect(self._update_animation_frame)
            self.animation_timer.start(80)  # 80ms per frame = smooth rotation

    def _update_animation_frame(self) -> None:
        """Update spinner to next animation frame."""
        if not self.is_animating or self.is_closed or not self.spinner_label:
            return

        try:
            self.spinner_label.setText(self.animation_frames[self.animation_frame])
            self.animation_frame = (self.animation_frame + 1) % len(self.animation_frames)
        except Exception as e:
            self.logger.error(f"Error updating animation: {e}")
            self.is_animating = False

    def _stop_animation(self) -> None:
        """Stop spinner animation."""
        if not self.is_animating:
            return

        self.is_animating = False
        self.animation_base_text = ""

        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None

        # Keep spinner visible but static at first frame
        if self.spinner_label:
            self.spinner_label.setText(self.animation_frames[0])

    def _close_impl(self) -> None:
        """Close window (must run in main thread).

        Only triggers shutdown if:
        1. User manually closed the window (not programmatic close after initialization).
        2. Shutdown hasn't already been requested.
        """
        with self._lock:
            if self.is_closed:
                return

            try:
                self._stop_animation()
                self.is_closed = True

                # Close the dialog
                self.accept()

                self.logger.info("Startup window closed")

                # Trigger shutdown if user closed (not programmatic)
                if (
                    not self._programmatic_close
                    and self.shutdown_coordinator
                    and not self.shutdown_coordinator.is_shutdown_requested()
                ):
                    self.shutdown_coordinator.request_shutdown(
                        reason="User closed startup window",
                        source="startup_window",
                    )

            except Exception as e:
                self.logger.error(f"Error closing window: {e}")

    def close(self) -> None:
        """Close window - thread-safe. Used when user manually closes window."""
        self.signals.close_requested.emit()

    def close_after_initialization(self) -> None:
        """Close window programmatically after successful initialization.

        This does NOT trigger shutdown - it's the normal close after initialization completes.
        """
        self._programmatic_close = True
        self.close()

    def is_visible(self) -> bool:
        """Check if window is visible.

        Returns:
            True if window is not closed, False otherwise.
        """
        with self._lock:
            return not self.is_closed

    def closeEvent(self, event) -> None:
        """Handle window close event.

        Args:
            event: Close event.
        """
        self._close_impl()
        event.accept()


class StartupProgressTracker:
    """Track and display progress during startup."""

    def __init__(self, startup_window: StartupWindow, total_steps: int):
        """Initialize progress tracker.

        Args:
            startup_window: StartupWindow instance.
            total_steps: Total number of initialization steps.
        """
        self.startup_window = startup_window
        self.total_steps = total_steps
        self.current_step = 0
        self.current_step_name = ""
        self.sub_step_progress = 0.0
        self._lock = threading.Lock()

    def start_step(self, step_name: str) -> None:
        """Start a new initialization step.

        Args:
            step_name: Name of the step.
        """
        with self._lock:
            self.current_step += 1
            self.current_step_name = step_name
            self.sub_step_progress = 0.0
        self._update_display(step_name, animate=True)

    def update_sub_step(self, sub_step_name: str, progress: float = 0.5) -> None:
        """Update status within current step.

        Args:
            sub_step_name: Name of the sub-step.
            progress: Progress within step (0.0 to 1.0).
        """
        with self._lock:
            self.sub_step_progress = max(0.0, min(1.0, progress))
        self._update_display(sub_step_name, animate=True)

    def update_status_animated(self, status: str, progress: float = 0.5) -> None:
        """Update status (animated).

        Args:
            status: Status message.
            progress: Progress within step (0.0 to 1.0).
        """
        with self._lock:
            self.sub_step_progress = max(0.0, min(1.0, progress))
        self._update_display(status, animate=True)

    def update_status_static(self, status: str, progress: float = 0.5) -> None:
        """Update status (static/non-animated).

        Args:
            status: Status message.
            progress: Progress within step (0.0 to 1.0).
        """
        with self._lock:
            self.sub_step_progress = max(0.0, min(1.0, progress))
        self._update_display(status, animate=False)

    def complete_step(self, step_name: str = "") -> None:
        """Mark current step as complete.

        Args:
            step_name: Optional completion message.
        """
        with self._lock:
            self.sub_step_progress = 1.0
        message = step_name or f"{self.current_step_name} completed"
        self._update_display(message, animate=True)

    def _update_display(self, status: str, animate: bool = False) -> None:
        """Calculate and update progress display.

        Args:
            status: Status message.
            animate: Whether to animate spinner.
        """
        with self._lock:
            if self.total_steps > 0:
                base_progress = (self.current_step - 1) / self.total_steps
                step_contribution = self.sub_step_progress / self.total_steps
                progress = base_progress + step_contribution
            else:
                progress = 0.0

            progress = min(1.0, progress)

        self.startup_window.update_progress(progress, status, animate=animate)

    def finish(self) -> None:
        """Complete initialization and close window."""
        self.startup_window.update_progress(1.0, "Ready!", animate=False)
