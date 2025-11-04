import logging
import threading
import time
import tkinter as tk
from collections import deque
from typing import Optional

import customtkinter as ctk

from vocalance.app.ui import ui_theme
from vocalance.app.ui.controls.dictation_popup_control import DictationPopupController
from vocalance.app.ui.utils.ui_icon_utils import set_window_icon_robust
from vocalance.app.ui.views.components.view_config import view_config

# Constants
SIMPLE_WINDOW_WIDTH = 200
SIMPLE_WINDOW_HEIGHT = 70
SMART_WINDOW_WIDTH = 800
SMART_WINDOW_HEIGHT = 550
VISUAL_WINDOW_WIDTH = 400
VISUAL_WINDOW_HEIGHT = 550
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
        self.visual_frame = self._create_visual_content()

        self.popup_window.withdraw()
        self.is_visible = False
        self.current_mode = None  # Track which mode we're in for proper text box routing

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
        popup.attributes("-alpha", 0.9)  # Semi-transparent (90% opacity)
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

    def _create_visual_content(self) -> ctk.CTkFrame:
        """Create visual dictation content (single box, no LLM)"""
        frame = ctk.CTkFrame(self.popup_window, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        # Dictation pane only (no LLM pane)
        ctk.CTkLabel(frame, text="Dictation", font=(view_config.dictation_popup.font_family, 20, "bold")).grid(
            row=0, column=0, padx=(10, 10), pady=5, sticky="w"
        )
        self.visual_dictation_box = ctk.CTkTextbox(
            frame,
            height=200,
            width=380,
            font=(view_config.dictation_popup.font_family, 13),
            fg_color=ui_theme.theme.shape_colors.dark,
            border_width=1,
            border_color=ui_theme.theme.shape_colors.medium,
        )
        self.visual_dictation_box.grid(row=1, column=0, padx=(10, 10), pady=(0, 10), sticky="nsew")

        return frame

    # Public API
    def show_simple_listening(self, mode: str, stop_command: Optional[str]) -> None:
        self._show_simple()

    def show_smart_dictation(self) -> None:
        self._show_smart()

    def show_visual_dictation(self) -> None:
        self._show_visual()

    def show_llm_processing(self) -> None:
        if self.is_visible:
            self.llm_label.configure(text="Processing...")

    def hide_popup(self) -> None:
        self._hide()

    def append_dictation_text(self, text: str) -> None:
        """Append dictation text and force immediate UI update"""
        # Support both smart mode (dictation_box) and visual mode (visual_dictation_box)
        if self.current_mode == "smart" and self.dictation_box and self.dictation_box.winfo_exists():
            self.dictation_box.insert("end", text)
            self.dictation_box.see("end")
            self.dictation_box.update_idletasks()
        elif self.current_mode == "visual" and self.visual_dictation_box and self.visual_dictation_box.winfo_exists():
            self.visual_dictation_box.insert("end", text)
            self.visual_dictation_box.see("end")
            self.visual_dictation_box.update_idletasks()

    def remove_dictation_characters(self, count: int) -> None:
        """Remove characters from end of dictation text"""
        if self.dictation_box and self.dictation_box.winfo_exists():
            current_text = self.dictation_box.get("1.0", "end-1c")
            if len(current_text) >= count:
                self.dictation_box.delete(f"end-{count+1}c", "end-1c")
                self.dictation_box.see("end")
                self.dictation_box.update_idletasks()

    def display_partial_text(self, text: str, segment_id: str) -> None:
        """Display partial (unstable) text in gray for streaming dictation.

        Partial text is shown in gray to indicate it may still change.
        When the same segment becomes final, this text is replaced.

        Args:
            text: Partial transcription text.
            segment_id: Unique identifier for this text segment.
        """
        text_box = None
        if self.current_mode == "smart" and self.dictation_box and self.dictation_box.winfo_exists():
            text_box = self.dictation_box
        elif self.current_mode == "visual" and self.visual_dictation_box and self.visual_dictation_box.winfo_exists():
            text_box = self.visual_dictation_box

        if not text_box:
            return

        # Remove any existing partial text with same segment_id
        try:
            text_box.tag_delete(f"partial_{segment_id}")
        except Exception:
            pass

        # Remove any existing partial text (there should only be one at a time)
        try:
            ranges = text_box.tag_ranges("partial")
            if ranges:
                text_box.delete(ranges[0], ranges[1])
        except Exception:
            pass

        # Insert new partial text at end
        start_index = text_box.index("end-1c")
        text_box.insert("end", text)
        end_index = text_box.index("end-1c")

        # Tag as partial with gray color
        text_box.tag_add("partial", start_index, end_index)
        text_box.tag_add(f"partial_{segment_id}", start_index, end_index)
        text_box.tag_config("partial", foreground="#888888")  # Gray color

        text_box.see("end")
        text_box.update_idletasks()

    def display_final_text(self, text: str, segment_id: str) -> None:
        """Display final (stable) text in white for streaming dictation.

        Final text replaces any partial text with the same segment_id and
        is shown in white to indicate it will no longer change.

        Args:
            text: Final transcription text.
            segment_id: Unique identifier for this text segment.
        """
        text_box = None
        if self.current_mode == "smart" and self.dictation_box and self.dictation_box.winfo_exists():
            text_box = self.dictation_box
        elif self.current_mode == "visual" and self.visual_dictation_box and self.visual_dictation_box.winfo_exists():
            text_box = self.visual_dictation_box

        if not text_box:
            return

        # Remove partial text with same segment_id
        try:
            ranges = text_box.tag_ranges(f"partial_{segment_id}")
            if ranges:
                text_box.delete(ranges[0], ranges[1])
            text_box.tag_delete(f"partial_{segment_id}")
        except Exception:
            pass

        # Also remove any generic partial tag
        try:
            text_box.tag_delete("partial")
        except Exception:
            pass

        # Insert final text (white color is default, no special tag needed)
        if text:
            text_box.insert("end", text + " ")
            text_box.see("end")
            text_box.update_idletasks()

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
            self.current_mode = "smart"
            self._hide_frames()
            self.smart_frame.pack(fill="both", expand=True, padx=10, pady=10)
            self.popup_window.geometry(f"{SMART_WINDOW_WIDTH}x{SMART_WINDOW_HEIGHT}")
            self._position_window(SMART_WINDOW_WIDTH, SMART_WINDOW_HEIGHT)
            self._clear_smart_content()
            self._show_window()

    def _show_visual(self):
        with self._ui_lock:
            self.current_mode = "visual"
            self._hide_frames()
            self.visual_frame.pack(fill="both", expand=True, padx=10, pady=10)
            self.popup_window.geometry(f"{VISUAL_WINDOW_WIDTH}x{VISUAL_WINDOW_HEIGHT}")
            self._position_window(VISUAL_WINDOW_WIDTH, VISUAL_WINDOW_HEIGHT)
            self._clear_visual_content()
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
                self.current_mode = None

    def _hide_frames(self):
        self.simple_frame.pack_forget()
        self.smart_frame.pack_forget()
        self.visual_frame.pack_forget()

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

    def _clear_visual_content(self):
        """Clear visual content."""
        self.visual_dictation_box.delete("1.0", "end")

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
