"""
Production-Ready Startup Window with Thread-Safe Animation
"""

import logging
import queue
import threading
import time
import tkinter as tk
from typing import Optional, Union

import customtkinter as ctk

from vocalance.app.config.app_config import AssetPathsConfig
from vocalance.app.ui import ui_theme
from vocalance.app.ui.utils.logo_service import LogoService
from vocalance.app.ui.utils.ui_assets import AssetCache


class StartupWindow:
    """
    Thread-safe startup window with reliable spinner animation.

    Design:
    - All GUI operations run in the main Tkinter thread
    - Initialization runs in GUI event loop thread (not main thread!)
    - Cross-thread updates queued via thread-safe queue
    - Animation runs via Tkinter's after() mechanism
    - Monospace spinner font prevents character width jitter

    Thread Safety:
    - update_progress() can be called from any thread (uses queue for cross-thread calls)
    - Animation and UI updates always execute in main tkinter thread
    - _programmatic_close flag prevents accidental shutdown triggers
    """

    def __init__(
        self,
        logger: logging.Logger,
        main_root: Union[tk.Tk, ctk.CTk],
        asset_paths_config: AssetPathsConfig,
        shutdown_coordinator=None,
    ):
        self.logger = logger
        self.main_root = main_root
        self.shutdown_coordinator = shutdown_coordinator
        self.window: Optional[ctk.CTkToplevel] = None
        self.progress_bar: Optional[ctk.CTkProgressBar] = None
        self.text_label: Optional[ctk.CTkLabel] = None
        self.spinner_label: Optional[ctk.CTkLabel] = None
        self.is_closed = False
        self._lock = threading.Lock()
        self._programmatic_close = False  # Track if close was programmatic vs user-initiated

        # Animation state
        self.is_animating = False
        self.animation_base_text = ""
        self.animation_frame = 0
        self.animation_frames = ["|", "/", "-", "\\"]
        self.animation_after_id = None

        # Thread-safe update queue and checker
        self._update_queue = queue.Queue()
        self._check_queue_id = None

        # Track icon reinforcement callbacks for cancellation
        self._icon_after_ids = []

        # Asset services
        self.asset_cache = AssetCache(asset_paths_config=asset_paths_config)
        self.logo_service = LogoService(self.asset_cache)

    def show(self) -> None:
        """Display the startup window - must be called from main thread."""
        if self.window is not None:
            return

        try:
            self.main_root.update_idletasks()

            # Create window
            self.window = ctk.CTkToplevel(self.main_root)
            self.window.title("Vocalance")
            self.window.geometry(f"{ui_theme.theme.dimensions.startup_width}x{ui_theme.theme.dimensions.startup_height}")
            self.window.resizable(False, False)
            self.window.configure(fg_color=ui_theme.theme.shape_colors.darkest)

            # Window configuration
            self._center_window()
            self.window.protocol("WM_DELETE_WINDOW", self.close)
            self.window.attributes("-toolwindow", False)
            self.window.attributes("-disabled", False)

            # Build UI
            self._create_ui()

            # Set icon BEFORE displaying window (better inheritance from parent)
            self._set_icon_with_retry()

            # Display window
            self.window.update_idletasks()
            self.window.lift()

            # Reinforce icon after display - track callbacks for cancellation
            self._icon_after_ids.append(self.window.after(50, self._reinforce_icon))
            self._icon_after_ids.append(self.window.after(200, self._reinforce_icon))

            self._start_queue_checker()
            self.logger.info("Startup window displayed")

        except Exception as e:
            self.logger.error(f"Error creating startup window: {e}", exc_info=True)

    def _center_window(self) -> None:
        """Center window on screen."""
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
        """Build UI components."""
        self.window.grid_columnconfigure(0, weight=1)

        # Main container
        main_frame = ctk.CTkFrame(self.window, fg_color=ui_theme.theme.shape_colors.darkest, corner_radius=0)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=0)  # Logo
        main_frame.grid_rowconfigure(1, weight=0)  # Progress bar
        main_frame.grid_rowconfigure(2, weight=0)  # Status (no vertical expansion)

        # Logo
        self.logo_label = self.logo_service.create_logo_widget(
            main_frame,
            max_size=ui_theme.theme.dimensions.startup_logo_size,
            context="startup",
            text_fallback="IRIS",
            logo_type="full",
        )
        self.logo_label.grid(row=0, column=0, pady=(20, 20), sticky="ew")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            main_frame,
            width=ui_theme.theme.dimensions.progress_bar_width,
            height=ui_theme.theme.dimensions.progress_bar_height,
            progress_color=ui_theme.theme.shape_colors.lightest,
            fg_color=ui_theme.theme.shape_colors.medium,
        )
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, pady=(0, 5), sticky="ew")

        # Status container (centered text + spinner)
        status_frame = ctk.CTkFrame(main_frame, fg_color="transparent", corner_radius=0)
        status_frame.grid(row=2, column=0, pady=(2, 20), padx=10, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)  # Left expand
        status_frame.grid_columnconfigure(1, weight=0)  # Text (fixed)
        status_frame.grid_columnconfigure(2, weight=0)  # Spinner (fixed)
        status_frame.grid_columnconfigure(3, weight=1)  # Right expand

        # Text label (uses font_family)
        font_family = ui_theme.theme.font_family.get_primary_font("regular")
        self.text_label = ctk.CTkLabel(
            status_frame,
            text="Starting up",
            font=(font_family, ui_theme.theme.font_sizes.small),
            text_color=ui_theme.theme.shape_colors.light,
            justify="center",
            anchor="center",
        )
        self.text_label.grid(row=0, column=1, padx=(0, 10), sticky="e")

        # Spinner label (uses monospace font from theme)
        monospace_font = ui_theme.theme.font_family.get_monospace_font()
        self.spinner_label = ctk.CTkLabel(
            status_frame,
            text="\\",
            font=(monospace_font, ui_theme.theme.font_sizes.small),
            text_color=ui_theme.theme.shape_colors.light,
            justify="center",
            anchor="w",
            width=15,
        )
        self.spinner_label.grid(row=0, column=2, sticky="w")

    def _start_queue_checker(self) -> None:
        """Start queue polling from main thread."""
        if not self.window or self.is_closed:
            return
        self._check_update_queue()

    def _check_update_queue(self) -> None:
        """Poll queue for updates from background threads (runs in main thread)."""
        if not self.window or self.is_closed:
            return

        try:
            while not self._update_queue.empty():
                item = self._update_queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 3:
                    progress, status, animate = item
                    if progress == "CLOSE":
                        self._close_impl()
                        return
                    else:
                        self._update_progress_impl(progress, status, animate)

            # Schedule next check
            if not self.is_closed:
                self._check_queue_id = self.window.after(50, self._check_update_queue)

        except queue.Empty:
            if not self.is_closed:
                self._check_queue_id = self.window.after(50, self._check_update_queue)
        except Exception as e:
            self.logger.error(f"Error checking queue: {e}")

    def process_queue_now(self) -> None:
        """Manually process queue immediately - for use during initialization polling."""
        if not self.window or self.is_closed:
            return

        try:
            while not self._update_queue.empty():
                item = self._update_queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 3:
                    progress, status, animate = item
                    if progress == "CLOSE":
                        self._close_impl()
                        return
                    else:
                        self._update_progress_impl(progress, status, animate)
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing queue: {e}")

    def update_progress(self, progress: float, status: str, animate: bool = False) -> None:
        """
        Update progress - thread-safe.

        Can be called from any thread. Initialization runs in GUI event loop thread,
        but UI updates must happen in main tkinter thread, so we always use the queue
        for cross-thread safety.

        Note: In this architecture, main_thread() is tkinter thread, not GUI event loop thread.
        """
        if threading.current_thread() == threading.main_thread():
            self._update_progress_impl(progress, status, animate)
        else:
            self._update_queue.put((progress, status, animate))

    def _update_progress_impl(self, progress: float, status: str, animate: bool) -> None:
        """Update progress bar and status text."""
        with self._lock:
            if self.is_closed or not self.window:
                return

            try:
                progress = max(0.0, min(1.0, progress))
                if self.progress_bar:
                    self.progress_bar.set(progress)

                if self.text_label and self.spinner_label and status:
                    if animate:
                        self._start_animation_impl(status.rstrip("."))
                    else:
                        self._stop_animation_impl()
                        self.text_label.configure(text=status)
                        self.spinner_label.configure(text="")

            except Exception as e:
                self.logger.error(f"Error updating progress: {e}")

    def _start_animation_impl(self, base_text: str) -> None:
        """Begin spinner animation."""
        if self.is_closed or not self.window:
            return

        self._stop_animation_impl()
        self.is_animating = True
        self.animation_base_text = base_text
        self.animation_frame = 0

        if self.text_label:
            self.text_label.configure(text=base_text)

        self._update_animation_frame()

    def _update_animation_frame(self) -> None:
        """Cycle to next spinner frame."""
        if not self.is_animating or self.is_closed or not self.window or not self.spinner_label:
            return

        try:
            self.spinner_label.configure(text=self.animation_frames[self.animation_frame])
            self.animation_frame = (self.animation_frame + 1) % len(self.animation_frames)

            if self.window and self.is_animating:
                self.animation_after_id = self.window.after(100, self._update_animation_frame)

        except Exception as e:
            self.logger.error(f"Error updating animation: {e}")
            self.is_animating = False

    def _stop_animation_impl(self) -> None:
        """Stop spinner animation."""
        self.is_animating = False
        self.animation_base_text = ""

        if self.spinner_label:
            self.spinner_label.configure(text="")

        if self.animation_after_id and self.window:
            try:
                self.window.after_cancel(self.animation_after_id)
            except Exception:
                pass
            self.animation_after_id = None

    def _close_impl(self) -> None:
        """
        Close window (must run in main thread).

        Only triggers shutdown if:
        1. User manually closed the window (not programmatic close after initialization)
        2. Shutdown hasn't already been requested

        This prevents normal startup window closure from triggering shutdown.
        """
        with self._lock:
            if self.is_closed or not self.window:
                return

            try:
                self._stop_animation_impl()

                if self._check_queue_id and self.window:
                    try:
                        self.window.after_cancel(self._check_queue_id)
                    except Exception:
                        pass
                    self._check_queue_id = None

                self.is_closed = True

                for after_id in self._icon_after_ids:
                    if self.window:
                        try:
                            self.window.after_cancel(after_id)
                        except Exception:
                            pass
                self._icon_after_ids.clear()

                if self.window:
                    self.window.destroy()
                    self.window = None
                self.logger.info("Startup window closed")

                if (
                    not self._programmatic_close
                    and self.shutdown_coordinator
                    and not self.shutdown_coordinator.is_shutdown_requested()
                ):
                    self.shutdown_coordinator.request_shutdown(reason="User closed startup window", source="startup_window")

            except Exception as e:
                self.logger.error(f"Error closing window: {e}")

    def close(self) -> None:
        """Close window - thread-safe. Used when user manually closes window."""
        if threading.current_thread() == threading.main_thread():
            self._close_impl()
        else:
            self._update_queue.put(("CLOSE", None, None))

    def close_after_initialization(self) -> None:
        """Close window programmatically after successful initialization.

        This does NOT trigger shutdown - it's the normal close after initialization completes.
        """
        self._programmatic_close = True
        self.close()

    def is_visible(self) -> bool:
        """Check if window is visible."""
        with self._lock:
            return self.window is not None and not self.is_closed

    def _set_icon_with_retry(self) -> None:
        """Set icon with parent inheritance for best results."""
        with self._lock:
            if not self.window or self.is_closed:
                return
            window_ref = self.window

        try:
            from vocalance.app.ui.utils.ui_icon_utils import set_window_icon_robust

            set_window_icon_robust(window=window_ref)
            window_ref.update_idletasks()
        except Exception as e:
            self.logger.warning(f"Error setting startup window icon: {e}")

    def _reinforce_icon(self) -> None:
        """Reinforce the icon setting to prevent CustomTkinter override."""
        with self._lock:
            if not self.window or self.is_closed:
                return
            window_ref = self.window

        try:
            from vocalance.app.ui.utils.ui_icon_utils import set_window_icon_robust

            set_window_icon_robust(window=window_ref)
            window_ref.update_idletasks()
        except Exception as e:
            self.logger.debug(f"Error reinforcing icon: {e}")


class StartupProgressTracker:
    """Track and display progress during startup."""

    def __init__(self, startup_window: StartupWindow, total_steps: int):
        self.startup_window = startup_window
        self.total_steps = total_steps
        self.current_step = 0
        self.current_step_name = ""
        self.sub_step_progress = 0.0
        self._lock = threading.Lock()

    def start_step(self, step_name: str) -> None:
        """Start a new initialization step."""
        with self._lock:
            self.current_step += 1
            self.current_step_name = step_name
            self.sub_step_progress = 0.0
        self._update_display(step_name, animate=True)

    def update_sub_step(self, sub_step_name: str, progress: float = 0.5) -> None:
        """Update status within current step."""
        with self._lock:
            self.sub_step_progress = max(0.0, min(1.0, progress))
        self._update_display(sub_step_name, animate=True)

    def update_status_animated(self, status: str, progress: float = 0.5) -> None:
        """Update status (animated)."""
        with self._lock:
            self.sub_step_progress = max(0.0, min(1.0, progress))
        self._update_display(status, animate=True)

    def update_status_static(self, status: str, progress: float = 0.5) -> None:
        """Update status (static/non-animated)."""
        with self._lock:
            self.sub_step_progress = max(0.0, min(1.0, progress))
        self._update_display(status, animate=False)

    def complete_step(self, step_name: str = "") -> None:
        """Mark current step as complete."""
        with self._lock:
            self.sub_step_progress = 1.0
        self._update_display(step_name or f"{self.current_step_name} completed", animate=True)

    def _update_display(self, status: str, animate: bool = False) -> None:
        """Calculate and update progress display."""
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

        def delayed_close():
            time.sleep(0.8)
            self.startup_window.close()

        threading.Thread(target=delayed_close, daemon=True).start()
