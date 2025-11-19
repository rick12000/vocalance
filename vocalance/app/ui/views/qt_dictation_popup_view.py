"""Qt-based dictation popup view for streaming transcription display.

Provides real-time dictation text display with three modes:
- Simple: Listening indicator with spinner animation
- Smart: Dictation pane + AI output pane
- Visual: Single dictation pane for visual commands
"""

import logging
import threading
from collections import deque

from PySide6.QtCore import QMetaObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QTextCharFormat
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme_manager


class QtDictationPopupView(QMainWindow):
    """Dictation popup window for streaming transcription display.

    Features:
    - Three display modes: simple, smart, visual
    - Real-time text streaming
    - Non-intrusive (always-on-top, no focus stealing)
    - Thread-safe token buffering
    - Spinner animation in simple mode
    """

    # Signals for thread-safe text updates
    _signal_partial_text = Signal(str, str)  # text, segment_id
    _signal_final_text = Signal(str, str)  # text, segment_id
    _signal_llm_token = Signal(str)  # token

    # Window sizes
    SIMPLE_WIDTH = 200
    SIMPLE_HEIGHT = 70
    SMART_WIDTH = 800
    SMART_HEIGHT = 550
    VISUAL_WIDTH = 400
    VISUAL_HEIGHT = 550
    WINDOW_MARGIN_X = 80
    WINDOW_MARGIN_Y = 80

    def __init__(self):
        """Initialize dictation popup view."""
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        # Thread safety
        self._ui_lock = threading.RLock()

        # Token buffering for smooth updates
        self._token_buffer = deque()
        self._last_flush_time = 0
        self._flush_interval_ms = 16  # ~60 FPS
        self._pending_flush = False

        # Spinner animation
        self.is_animating = False
        self.animation_frame = 0
        self.animation_frames = ["|", "/", "-", "\\"]
        self.animation_timer = None

        # Current display mode
        self.current_mode = None

        # Setup window
        self._setup_window()
        self._create_ui()
        self._apply_styling()

        # Connect signals for thread-safe updates
        self._signal_partial_text.connect(self._do_display_partial_text)
        self._signal_final_text.connect(self._do_display_final_text)
        self._signal_llm_token.connect(self._do_append_llm_token)

        self.logger.info("QtDictationPopupView initialized")

    def _setup_window(self) -> None:
        """Configure window properties."""
        self.setWindowTitle("Dictation")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Semi-transparent window
        self.setWindowOpacity(0.95)

        # Set minimum size
        self.setMinimumSize(200, 70)

    def _create_ui(self) -> None:
        """Create UI elements."""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # Simple mode: Listening indicator
        self.simple_widget = QWidget()
        simple_layout = QVBoxLayout(self.simple_widget)
        simple_layout.setContentsMargins(0, 0, 0, 0)
        self.simple_label = QLabel("Listening")
        self.simple_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        simple_layout.addWidget(self.simple_label)
        self.simple_widget.setVisible(False)
        main_layout.addWidget(self.simple_widget)

        # Smart mode: Dictation + AI output (side by side like legacy)
        self.smart_widget = QWidget()
        smart_main_layout = QVBoxLayout(self.smart_widget)
        smart_main_layout.setContentsMargins(0, 0, 0, 0)
        smart_main_layout.setSpacing(10)

        # Container for side-by-side layout
        side_by_side_container = QWidget()
        side_by_side_layout = QHBoxLayout(side_by_side_container)
        side_by_side_layout.setContentsMargins(0, 0, 0, 0)
        side_by_side_layout.setSpacing(10)

        # Left column: Dictation
        dictation_container = QWidget()
        dictation_layout = QVBoxLayout(dictation_container)
        dictation_layout.setContentsMargins(0, 0, 0, 0)
        dictation_layout.setSpacing(5)

        dictation_label = QLabel("Dictation")
        dictation_label.setFont(theme_manager.get_font(size=theme_manager.font_sizes.xlarge, bold=True))
        dictation_layout.addWidget(dictation_label)

        self.dictation_box = QPlainTextEdit()
        self.dictation_box.setReadOnly(True)
        self.dictation_box.setMinimumWidth(350)
        dictation_layout.addWidget(self.dictation_box, 1)

        side_by_side_layout.addWidget(dictation_container, 1)

        # Right column: AI Output
        llm_container = QWidget()
        llm_layout = QVBoxLayout(llm_container)
        llm_layout.setContentsMargins(0, 0, 0, 0)
        llm_layout.setSpacing(5)

        self.llm_label = QLabel("AI Output")
        self.llm_label.setFont(theme_manager.get_font(size=theme_manager.font_sizes.xlarge, bold=True))
        llm_layout.addWidget(self.llm_label)

        self.llm_box = QPlainTextEdit()
        self.llm_box.setReadOnly(True)
        self.llm_box.setMinimumWidth(350)
        llm_layout.addWidget(self.llm_box, 1)

        side_by_side_layout.addWidget(llm_container, 1)

        smart_main_layout.addWidget(side_by_side_container, 1)

        self.smart_widget.setVisible(False)
        main_layout.addWidget(self.smart_widget, 1)

        # Visual mode: Dictation only
        self.visual_widget = QWidget()
        visual_layout = QVBoxLayout(self.visual_widget)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(10)

        visual_label = QLabel("Dictation")
        visual_label.setFont(theme_manager.get_font(size=theme_manager.font_sizes.xlarge, bold=True))
        visual_layout.addWidget(visual_label)

        self.visual_dictation_box = QPlainTextEdit()
        self.visual_dictation_box.setReadOnly(True)
        visual_layout.addWidget(self.visual_dictation_box)

        self.visual_widget.setVisible(False)
        main_layout.addWidget(self.visual_widget, 1)

    def _apply_styling(self) -> None:
        """Apply QSS styling."""
        stylesheet = f"""
        QMainWindow {{
            background-color: {theme_manager.shape_colors.darkest};
            color: {theme_manager.text_colors.lightest};
            border: 1px solid {theme_manager.shape_colors.medium};
            border-radius: 8px;
        }}

        QLabel {{
            color: {theme_manager.text_colors.light};
            font-size: {theme_manager.font_sizes.medium}px;
        }}

        QPlainTextEdit {{
            background-color: {theme_manager.shape_colors.dark};
            color: {theme_manager.text_colors.light};
            border: 1px solid {theme_manager.shape_colors.medium};
            border-radius: 4px;
            padding: 5px;
            font-size: {theme_manager.font_sizes.medium}px;
        }}

        QPlainTextEdit:focus {{
            border: 1px solid {theme_manager.shape_colors.accent};
        }}
        """
        self.setStyleSheet(stylesheet)

    # Public API

    @Slot()
    def show_simple_listening(self) -> None:
        """Show simple listening indicator - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_show_simple", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_show_simple(self) -> None:
        """Internal show simple - MUST run on main Qt thread."""
        with self._ui_lock:
            self._hide_all_modes()
            self.simple_widget.setVisible(True)
            self.current_mode = "simple"
            self._position_window(self.SIMPLE_WIDTH, self.SIMPLE_HEIGHT, "bottom_left")
            self._show_window()
            self._start_animation()

    @Slot()
    def show_smart_dictation(self) -> None:
        """Show smart dictation (dictation + LLM output) - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_show_smart", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_show_smart(self) -> None:
        """Internal show smart - MUST run on main Qt thread."""
        with self._ui_lock:
            self._hide_all_modes()
            self.current_mode = "smart"
            self.smart_widget.setVisible(True)
            self._clear_smart_content()
            self._position_window(self.SMART_WIDTH, self.SMART_HEIGHT, "center_left")
            self._show_window()
            self.logger.info(f"Smart dictation window shown, mode={self.current_mode}")

    @Slot()
    def show_visual_dictation(self) -> None:
        """Show visual dictation (single pane) - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_show_visual", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def show_llm_processing(self) -> None:
        """Show LLM processing mode (keep smart layout, just update label) - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_show_llm_processing", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_show_visual(self) -> None:
        """Internal show visual - MUST run on main Qt thread."""
        with self._ui_lock:
            self._hide_all_modes()
            self.current_mode = "visual"
            self.visual_widget.setVisible(True)
            self._clear_visual_content()
            self._position_window(self.VISUAL_WIDTH, self.VISUAL_HEIGHT, "center_left")
            self._show_window()
            self.logger.info(f"Visual dictation window shown, mode={self.current_mode}")

    @Slot()
    def _do_show_llm_processing(self) -> None:
        """Internal show LLM processing - MUST run on main Qt thread."""
        # Keep smart widget visible, just update the status
        # This is called after dictation stops and before LLM processing starts
        if self.current_mode == "smart":
            self.llm_label.setText("Processing...")
            self.logger.debug("Switched to LLM processing mode")

    @Slot()
    def hide_popup(self) -> None:
        """Hide the popup - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_hide_popup", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_hide_popup(self) -> None:
        """Internal hide popup - MUST run on main Qt thread."""
        with self._ui_lock:
            self._stop_animation()
            self.hide()
            self.current_mode = None

    def append_dictation_text(self, text: str) -> None:
        """Append text to dictation box - thread-safe."""
        self.logger.debug(f"append_dictation_text called with: '{text[:50]}...' (mode={self.current_mode})")
        # Use QTimer.singleShot for thread-safe GUI update
        QTimer.singleShot(0, lambda: self._do_append_dictation_text(text))

    def _do_append_dictation_text(self, text: str) -> None:
        """Internal append dictation text - MUST run on main Qt thread."""
        self.logger.debug(f"_do_append_dictation_text executing: mode={self.current_mode}, text='{text[:50]}...'")
        if self.current_mode == "smart":
            cursor = self.dictation_box.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(text)
            self.dictation_box.setTextCursor(cursor)
            self.logger.debug(f"Appended to smart dictation box: '{text[:30]}...'")
        elif self.current_mode == "visual":
            cursor = self.visual_dictation_box.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(text)
            self.visual_dictation_box.setTextCursor(cursor)
            self.logger.debug(f"Appended to visual dictation box: '{text[:30]}...'")
        else:
            self.logger.warning(f"append_dictation_text called but mode is '{self.current_mode}' - ignoring")

    def display_partial_text(self, text: str, segment_id: str) -> None:
        """Display partial (unstable) text in gray for streaming dictation.

        Partial text is shown in gray to indicate it may still change.
        When the same segment becomes final, this text is replaced.
        Thread-safe - uses Qt Signal for cross-thread communication.

        Args:
            text: Partial transcription text.
            segment_id: Unique identifier for this text segment.
        """
        self.logger.info(f"display_partial_text CALLED: text='{text}', segment_id={segment_id}, mode={self.current_mode}")
        # Emit signal - Qt will automatically marshal to main thread
        self._signal_partial_text.emit(text, segment_id)

    @Slot(str, str)
    def _do_display_partial_text(self, text: str, segment_id: str) -> None:
        """Internal display partial text - MUST run on main Qt thread.

        Connected to _signal_partial_text for automatic thread marshalling.
        """
        self.logger.info(f"_do_display_partial_text EXECUTING: mode={self.current_mode}, text='{text[:50]}'")

        # Determine which text box to use based on mode
        text_box = None
        if self.current_mode == "smart":
            text_box = self.dictation_box
            self.logger.info("Selected dictation_box for smart mode")
        elif self.current_mode == "visual":
            text_box = self.visual_dictation_box
            self.logger.info("Selected visual_dictation_box for visual mode")

        if not text_box:
            self.logger.error(f"NO TEXT BOX SELECTED! mode='{self.current_mode}'")
            return

        # CRITICAL: Remove ALL existing partial text before inserting new partial
        # Legacy behavior (lines 258-264): only ONE partial text exists at a time
        if not hasattr(self, "_partial_segments"):
            self._partial_segments = {}

        # Remove all existing partial text segments
        for old_segment_id in list(self._partial_segments.keys()):
            old_start, old_end = self._partial_segments[old_segment_id]
            cursor = text_box.textCursor()
            cursor.setPosition(old_start)
            cursor.setPosition(old_end, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            self.logger.debug(f"Removed old partial segment {old_segment_id} at pos {old_start}-{old_end}")

        # Clear the dictionary
        self._partial_segments.clear()

        # Now insert new partial text at end with GRAY formatting
        cursor = text_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)

        # Store position before insertion
        start_pos = cursor.position()

        # Insert text
        cursor.insertText(text)
        end_pos = cursor.position()

        # Apply GRAY color to the inserted text (partial = unstable)
        cursor.setPosition(start_pos)
        cursor.setPosition(end_pos, cursor.MoveMode.KeepAnchor)
        gray_format = QTextCharFormat()
        gray_format.setForeground(QColor("#888888"))  # Gray for partial
        cursor.setCharFormat(gray_format)

        # Store this segment's position for removal when final text arrives
        self._partial_segments[segment_id] = (start_pos, end_pos)

        text_box.setTextCursor(cursor)
        text_box.ensureCursorVisible()
        self.logger.debug(f"Displayed GRAY partial text at {start_pos}-{end_pos}: '{text[:30]}...'")

    def display_final_text(self, text: str, segment_id: str) -> None:
        """Display final (stable) text in white for streaming dictation.

        Final text replaces any partial text with the same segment_id and
        is shown in white to indicate it will no longer change.
        Thread-safe - uses Qt Signal for cross-thread communication.

        Args:
            text: Final transcription text.
            segment_id: Unique identifier for this text segment.
        """
        self.logger.info(f"display_final_text CALLED: text='{text}', segment_id={segment_id}, mode={self.current_mode}")
        # Emit signal - Qt will automatically marshal to main thread
        self._signal_final_text.emit(text, segment_id)

    @Slot(str, str)
    def _do_display_final_text(self, text: str, segment_id: str) -> None:
        """Internal display final text - MUST run on main Qt thread.

        Connected to _signal_final_text for automatic thread marshalling.
        """
        self.logger.info(f"_do_display_final_text EXECUTING: mode={self.current_mode}, text='{text[:50]}'")

        # Determine which text box to use based on mode
        text_box = None
        if self.current_mode == "smart":
            text_box = self.dictation_box
            self.logger.info("Selected dictation_box for smart mode")
        elif self.current_mode == "visual":
            text_box = self.visual_dictation_box
            self.logger.info("Selected visual_dictation_box for visual mode")

        if not text_box:
            self.logger.error(f"NO TEXT BOX SELECTED! mode='{self.current_mode}'")
            return

        # Remove partial text with same segment_id if it exists
        # Legacy behavior (lines 298-311): replace partial with final
        if hasattr(self, "_partial_segments") and segment_id in self._partial_segments:
            start_pos, end_pos = self._partial_segments[segment_id]
            cursor = text_box.textCursor()
            cursor.setPosition(start_pos)
            cursor.setPosition(end_pos, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            del self._partial_segments[segment_id]
            self.logger.debug(f"Removed partial text for segment {segment_id} at {start_pos}-{end_pos}")

        # Insert final text at end with WHITE formatting (stable/permanent)
        cursor = text_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)

        # Create character format for WHITE text (final = stable)
        white_format = QTextCharFormat()
        white_format.setForeground(QColor(theme_manager.text_colors.light))  # White for final
        cursor.setCharFormat(white_format)

        # Insert text with trailing space (matches legacy line 315)
        if text:
            cursor.insertText(text + " ")

        text_box.setTextCursor(cursor)
        text_box.ensureCursorVisible()
        self.logger.debug(f"Displayed WHITE final text: '{text[:30]}...'")

    def append_llm_token(self, token: str) -> None:
        """Append LLM output token with batching for smooth updates.

        Thread-safe - uses Qt Signal for cross-thread communication.
        Can be called from any thread.

        Args:
            token: LLM output token to append
        """
        self.logger.debug(f"append_llm_token CALLED: token='{token}', mode={self.current_mode}")
        # Emit signal - Qt will automatically marshal to main thread
        self._signal_llm_token.emit(token)

    @Slot(str)
    def _do_append_llm_token(self, token: str) -> None:
        """Internal append LLM token - MUST run on main Qt thread.

        Connected to _signal_llm_token for automatic thread marshalling.
        Buffers tokens and flushes when threshold reached for smooth 60fps updates.
        """
        self.logger.debug(f"_do_append_llm_token EXECUTING on main thread: token='{token}', mode={self.current_mode}")

        if self.current_mode != "smart":
            self.logger.warning(f"_do_append_llm_token called but mode is '{self.current_mode}'")
            return

        # Buffer the token
        with self._ui_lock:
            self._token_buffer.append(token)

            # Schedule flush if buffer has enough tokens and no flush pending
            if len(self._token_buffer) >= 3 and not self._pending_flush:
                self._pending_flush = True
                # Safe to use QTimer.singleShot here - we're on the main Qt thread
                QTimer.singleShot(1, self._flush_token_buffer)
                self.logger.debug(f"Scheduled token buffer flush ({len(self._token_buffer)} tokens buffered)")

    def _flush_token_buffer(self) -> None:
        """Flush buffered tokens to LLM output box.

        Must be called from main Qt thread only (scheduled via QTimer.singleShot).
        """
        self.logger.debug(f"_flush_token_buffer CALLED: mode={self.current_mode}, buffer_size={len(self._token_buffer)}")

        if self.current_mode != "smart":
            with self._ui_lock:
                self._token_buffer.clear()
                self._pending_flush = False
            self.logger.warning(f"_flush_token_buffer aborted - wrong mode: {self.current_mode}")
            return

        # Get batched tokens
        with self._ui_lock:
            if not self._token_buffer:
                self._pending_flush = False
                self.logger.debug("_flush_token_buffer: no tokens to flush")
                return

            batched = "".join(self._token_buffer)
            token_count = len(self._token_buffer)
            self._token_buffer.clear()

        # Insert into LLM box
        if not self.llm_box:
            self.logger.error("_flush_token_buffer: llm_box is None!")
            with self._ui_lock:
                self._pending_flush = False
            return

        cursor = self.llm_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(batched)
        self.llm_box.setTextCursor(cursor)
        self.llm_box.ensureCursorVisible()

        self.logger.debug(f"_flush_token_buffer: flushed {token_count} tokens ('{batched[:50]}...')")

        with self._ui_lock:
            self._pending_flush = False

    def update_llm_status(self, status: str) -> None:
        """Update LLM output label status."""
        if self.current_mode == "smart":
            self.llm_label.setText(status)

    # Internal methods

    def _hide_all_modes(self) -> None:
        """Hide all mode widgets."""
        self.simple_widget.setVisible(False)
        self.smart_widget.setVisible(False)
        self.visual_widget.setVisible(False)

    def _clear_smart_content(self) -> None:
        """Clear smart mode content."""
        self.dictation_box.clear()
        self.llm_box.clear()
        self.llm_label.setText("AI Output")
        with self._ui_lock:
            self._token_buffer.clear()

    def _clear_visual_content(self) -> None:
        """Clear visual mode content."""
        self.visual_dictation_box.clear()

    def _show_window(self) -> None:
        """Show the window at top level (matches legacy behavior)."""
        if not self.isVisible():
            self.show()
            self.raise_()
            self.activateWindow()  # Ensure window is at top level
            # Don't call setFocus() to prevent stealing focus from user's current task
            self.logger.debug(f"Dictation popup shown in {self.current_mode} mode")

    def _position_window(self, width: int, height: int, position_type: str = "center_left") -> None:
        """Position window on screen matching legacy positioning behavior."""
        # Get primary screen for positioning
        from PySide6.QtWidgets import QApplication

        primary_screen = QApplication.primaryScreen()

        if not primary_screen:
            # Fallback positioning
            self.setGeometry(100, 100, width, height)
            self.logger.warning("No screen available for positioning, using fallback")
            return

        # Use available geometry (excludes taskbar) for positioning calculations
        screen_geom = primary_screen.availableGeometry()

        if position_type == "bottom_left":
            # Position at bottom-left of screen (matches legacy)
            x = self.WINDOW_MARGIN_X
            y = screen_geom.height() - height - self.WINDOW_MARGIN_Y
        elif position_type == "center_left":
            # Position at center-left of screen (matches legacy)
            x = self.WINDOW_MARGIN_X
            y = (screen_geom.height() - height) // 2
        else:  # center
            x = (screen_geom.width() - width) // 2
            y = (screen_geom.height() - height) // 2

        self.setGeometry(x, y, width, height)
        self.logger.debug(f"Positioned window at ({x}, {y}) with size ({width}, {height}), type={position_type}")

    def _start_animation(self) -> None:
        """Start spinner animation for simple mode."""
        if self.is_animating or self.current_mode != "simple":
            return

        self.is_animating = True
        self.animation_frame = 0

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation_frame)
        self.animation_timer.start(100)

    def _update_animation_frame(self) -> None:
        """Update spinner animation frame."""
        if not self.is_animating or self.current_mode != "simple":
            self._stop_animation()
            return

        frame_char = self.animation_frames[self.animation_frame]
        self.simple_label.setText(f"Listening {frame_char}")
        self.animation_frame = (self.animation_frame + 1) % len(self.animation_frames)

    def _stop_animation(self) -> None:
        """Stop spinner animation."""
        self.is_animating = False

        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None

        self.simple_label.setText("Listening")

    def keyPressEvent(self, event) -> None:
        """Handle key press events - allow Escape to close."""
        if event.key() == Qt.Key.Key_Escape:
            self.hide_popup()
        else:
            super().keyPressEvent(event)
