import logging
import threading
import time
import tkinter as tk
from collections import deque
from typing import Optional

import customtkinter as ctk

from vocalance.app.ui import ui_theme
from vocalance.app.ui.controls.dictation_popup_control import DictationPopupController, DictationPopupMode
from vocalance.app.ui.utils.ui_icon_utils import set_window_icon_robust
from vocalance.app.ui.views.components.view_config import view_config

# Constants
SIMPLE_WINDOW_WIDTH = 200
SIMPLE_WINDOW_HEIGHT = 70
SMART_WINDOW_WIDTH = 800
SMART_WINDOW_HEIGHT = 550
WINDOW_MARGIN_X = 80
WINDOW_MARGIN_Y_BOTTOM = 80


class DictationPopupView:
    """
    Simplified dictation popup that never steals focus.

    Thread Safety:
    - UI operations run in main tkinter thread
    - Token buffering is thread-safe with lock protection
    - _pending_flush atomic check-and-set prevents duplicate flushes
    - Controller calls append_llm_token from GUI event loop thread
    """

    def __init__(self, parent_root: tk.Tk, controller: DictationPopupController):
        self.parent_root = parent_root
        self.controller = controller
        self.controller.set_view_callback(self)

        # Thread safety: Lock protects all shared state
        self._ui_lock = threading.RLock()

        # UI components (main thread only)
        self.popup_window = self._create_popup_window()
        self.simple_frame = self._create_simple_content()
        self.smart_frame = self._create_smart_content()

        self.popup_window.withdraw()
        self.is_visible = False
        self.current_mode = DictationPopupMode.HIDDEN

        # Token batching for smooth UI updates (protected by lock)
        self._token_buffer = deque()
        self._last_flush_time = 0
        self._flush_interval_ms = 16  # ~60 FPS
        self._pending_flush = False

        # Streaming token highlighting (UI state, accessed only from main thread)
        self._last_token_start_index = None
        self._last_token_end_index = None

        logging.info("Simplified DictationPopupView initialized")

    def _create_popup_window(self) -> ctk.CTkToplevel:
        """Create non-intrusive popup window"""
        popup = ctk.CTkToplevel(self.parent_root)
        popup.title("Dictation")

        # Set icon early - before making visible
        try:
            set_window_icon_robust(popup)
        except Exception:
            pass

        # Make window completely non-intrusive
        popup.wm_attributes("-topmost", True)
        popup.wm_attributes("-toolwindow", True)
        popup.overrideredirect(True)  # No title bar - prevents focus stealing
        popup.resizable(False, False)
        popup.configure(fg_color=ui_theme.theme.shape_colors.darkest)

        # Prevent window from taking focus
        popup.focus_set = lambda: None  # Disable focus_set
        popup.grab_set = lambda: None  # Disable grab_set

        return popup

    def _create_simple_content(self) -> ctk.CTkFrame:
        """Create simple listening indicator"""
        frame = ctk.CTkFrame(self.popup_window, fg_color=ui_theme.theme.shape_colors.darkest)

        self.simple_label = ctk.CTkLabel(
            frame,
            text="Listening...",
            font=(view_config.dictation_popup.font_family, 12),
            text_color=ui_theme.theme.text_colors.light,
        )
        self.simple_label.pack(pady=10)

        return frame

    def _create_smart_content(self) -> ctk.CTkFrame:
        """Create smart dictation content"""
        frame = ctk.CTkFrame(self.popup_window, fg_color="transparent")
        frame.grid_columnconfigure((0, 1), weight=1)
        frame.grid_rowconfigure(1, weight=1)

        # Dictation pane
        ctk.CTkLabel(frame, text="Dictation", font=(view_config.dictation_popup.font_family, 20, "bold")).grid(
            row=0, column=0, padx=(10, 10), pady=5, sticky="w"
        )
        self.dictation_box = ctk.CTkTextbox(
            frame,
            height=200,
            font=(view_config.dictation_popup.font_family, 13),
            fg_color=ui_theme.theme.shape_colors.dark,
            border_width=1,
            border_color=ui_theme.theme.shape_colors.medium,
        )
        self.dictation_box.grid(row=1, column=0, padx=(10, 10), pady=(0, 10), sticky="nsew")

        # LLM pane
        self.llm_label = ctk.CTkLabel(frame, text="AI Output", font=(view_config.dictation_popup.font_family, 20, "bold"))
        self.llm_label.grid(row=0, column=1, padx=(10, 10), pady=5, sticky="w")
        self.llm_box = ctk.CTkTextbox(
            frame,
            height=200,
            font=(view_config.dictation_popup.font_family, 13),
            fg_color=ui_theme.theme.shape_colors.dark,
            border_width=1,
            border_color=ui_theme.theme.shape_colors.medium,
        )
        self.llm_box.grid(row=1, column=1, padx=(10, 10), pady=(0, 10), sticky="nsew")

        return frame

    # Public API
    def show_simple_listening(self, mode: str, stop_command: Optional[str]) -> None:
        self._show_simple()

    def show_smart_dictation(self) -> None:
        self._show_smart()

    def show_llm_processing(self) -> None:
        if self.is_visible:
            self.llm_label.configure(text="Processing...")

    def hide_popup(self) -> None:
        self._hide()

    def append_dictation_text(self, text: str) -> None:
        """Append dictation text and force immediate UI update"""
        if self.dictation_box and self.dictation_box.winfo_exists():
            self.dictation_box.insert("end", text)
            self.dictation_box.see("end")
            self.dictation_box.update_idletasks()

    def remove_dictation_characters(self, count: int) -> None:
        """Remove characters from end of dictation text"""
        if self.dictation_box and self.dictation_box.winfo_exists():
            current_text = self.dictation_box.get("1.0", "end-1c")
            if len(current_text) >= count:
                self.dictation_box.delete(f"end-{count+1}c", "end-1c")
                self.dictation_box.see("end")
                self.dictation_box.update_idletasks()

    def append_llm_token(self, token: str) -> None:
        """Append LLM token with smart batching for smooth 60fps updates. Thread-safe."""
        logging.debug(f"VIEW: append_llm_token called with: '{token}'")
        if not self.llm_box or not self.llm_box.winfo_exists():
            logging.warning("VIEW: llm_box not available!")
            return

        with self._ui_lock:
            self._token_buffer.append(token)

            current_time = time.time() * 1000
            time_since_last_flush = current_time - self._last_flush_time

            should_flush = len(self._token_buffer) > 0 and (
                time_since_last_flush >= self._flush_interval_ms or len(self._token_buffer) >= 3
            )

            if should_flush and not self._pending_flush:
                self._pending_flush = True
                schedule_needed = True
            else:
                schedule_needed = False

        if schedule_needed:
            self.parent_root.after(1, self._flush_token_buffer)

    def _flush_token_buffer(self) -> None:
        """Flush buffered tokens to UI with streaming token highlighting. Thread-safe."""
        if not self.llm_box or not self.llm_box.winfo_exists():
            with self._ui_lock:
                self._token_buffer.clear()
                self._pending_flush = False
            return

        with self._ui_lock:
            if not self._token_buffer:
                self._pending_flush = False
                return

            batched = "".join(self._token_buffer)
            self._token_buffer.clear()
            self._last_flush_time = time.time() * 1000

        if self._last_token_start_index is not None and self._last_token_end_index is not None:
            try:
                self.llm_box.tag_remove("streaming", self._last_token_start_index, self._last_token_end_index)
            except Exception:
                pass

        start_index = self.llm_box.index("end-1c")

        self.llm_box.insert("end", batched)

        end_index = self.llm_box.index("end-1c")

        try:
            self.llm_box.tag_add("streaming", start_index, end_index)
            self.llm_box.tag_config("streaming", foreground=ui_theme.theme.text_colors.streaming_token)
        except Exception as e:
            logging.debug(f"Error applying streaming tag: {e}")

        self._last_token_start_index = start_index
        self._last_token_end_index = end_index

        self.llm_box.see("end")
        logging.debug(f"VIEW: Flushed batch: '{batched[:20]}...'")

        with self._ui_lock:
            self._pending_flush = False

    def update_llm_status(self, status: str) -> None:
        if self.llm_label and self.llm_label.winfo_exists():
            self.llm_label.configure(text=status)

    # Internal methods
    def _show_simple(self):
        with self._ui_lock:
            self._hide_frames()
            self.simple_frame.pack(fill="both", expand=True, padx=10, pady=10)
            self.popup_window.geometry(f"{SIMPLE_WINDOW_WIDTH}x{SIMPLE_WINDOW_HEIGHT}")
            self._position_window(SIMPLE_WINDOW_WIDTH, SIMPLE_WINDOW_HEIGHT)
            self._show_window()

    def _show_smart(self):
        with self._ui_lock:
            self._hide_frames()
            self.smart_frame.pack(fill="both", expand=True, padx=10, pady=10)
            self.popup_window.geometry(f"{SMART_WINDOW_WIDTH}x{SMART_WINDOW_HEIGHT}")
            self._position_window(SMART_WINDOW_WIDTH, SMART_WINDOW_HEIGHT)
            self._clear_smart_content()
            self._show_window()

    def _show_window(self):
        if not self.is_visible:
            self.popup_window.deiconify()
            self.is_visible = True
            # Explicitly prevent focus stealing
            self.popup_window.lift()  # Bring to front without focus

            # Reinforce icon after window is shown to prevent CustomTkinter override
            self.parent_root.after(50, self._reinforce_icon)
            self.parent_root.after(200, self._reinforce_icon)

    def _hide(self):
        with self._ui_lock:
            if self.is_visible:
                self.popup_window.withdraw()
                self.is_visible = False

    def _hide_frames(self):
        self.simple_frame.pack_forget()
        self.smart_frame.pack_forget()

    def _clear_smart_content(self):
        """Clear smart content. Thread-safe buffer clearing."""
        self.dictation_box.delete("1.0", "end")
        self.llm_box.delete("1.0", "end")
        self.llm_label.configure(text="AI Output")
        # Reset streaming token tracking
        self._last_token_start_index = None
        self._last_token_end_index = None
        with self._ui_lock:
            self._token_buffer.clear()

    def _position_window(self, width: int, height: int):
        self.parent_root.winfo_screenwidth()
        screen_height = self.parent_root.winfo_screenheight()

        x = WINDOW_MARGIN_X
        # Position higher up on screen (35% from top instead of 50% center)
        y = int(screen_height * 0.35 - height // 2) if height >= 200 else screen_height - height - WINDOW_MARGIN_Y_BOTTOM

        self.popup_window.geometry(f"+{x}+{y}")

    def _reinforce_icon(self) -> None:
        """Reinforce the icon setting to prevent CustomTkinter override."""
        if self.popup_window and self.popup_window.winfo_exists():
            try:
                set_window_icon_robust(self.popup_window)
                self.popup_window.update_idletasks()
            except Exception:
                pass
