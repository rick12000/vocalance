"""Qt-based dictation popup view for streaming transcription display.

Provides real-time dictation text display with three modes:
- Simple: Listening indicator with spinner animation
- Smart: Dictation pane + AI output pane
- Visual: Single dictation pane for visual commands
"""

import logging
import threading
from collections import deque

from PySide6.QtCore import QEasingCurve, QMetaObject, QPropertyAnimation, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QTextCharFormat
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QVBoxLayout, QWidget

from vocalance.app.ui.components.sound_wave_widget import SoundWaveWidget
from vocalance.app.ui.qt_theme import theme


class QtDictationPopupView(QMainWindow):
    """Dictation popup window for streaming transcription display.

    Features:
    - Three display modes: simple, smart, visual
    - Real-time text streaming
    - Non-intrusive (always-on-top, no focus stealing)
    - Thread-safe token buffering
    - Sound wave animation in simple mode
    """

    # Signals for thread-safe text updates
    _signal_partial_text = Signal(str, str)  # text, segment_id
    _signal_final_text = Signal(str, str)  # text, segment_id
    _signal_llm_token = Signal(str)  # token
    _signal_audio_level = Signal(float)  # audio level

    # Window sizes (determined by the sound wave widget)
    SIMPLE_WIDTH = 60  # Exact fit for sound wave widget
    SIMPLE_HEIGHT = 30  # Exact fit for sound wave widget
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

        # Current display mode
        self.current_mode = None

        # Animation properties
        self._animation_in = None
        self._animation_out = None
        self._opacity_animation_in = None
        self._opacity_animation_out = None
        self._final_position = None
        self._target_geometry = None
        self._animation_duration_ms = 400  # Animation duration in milliseconds

        # Setup window
        self._setup_window()
        self._create_ui()
        self._apply_styling()

        # Connect signals for thread-safe updates
        self._signal_partial_text.connect(self._do_display_partial_text)
        self._signal_final_text.connect(self._do_display_final_text)
        self._signal_llm_token.connect(self._do_append_llm_token)
        self._signal_audio_level.connect(self._do_update_audio_level)

        self.logger.info("QtDictationPopupView initialized")

    def _setup_window(self) -> None:
        """Configure window properties."""
        self.setWindowTitle("Dictation")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def paintEvent(self, event) -> None:
        """Draw rounded background for the window."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.setBrush(QColor(theme.config.shapes.darkest))
        painter.setPen(Qt.PenStyle.NoPen)

        # Rounded rect filling the entire window
        rect = self.rect()
        painter.drawRoundedRect(rect, 16, 16)

        # Optional: Draw subtle border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor(theme.config.shapes.medium))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 16, 16)

    def _create_ui(self) -> None:
        """Create UI elements."""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # Simple mode: Sound wave animation
        self.simple_widget = QWidget()
        simple_layout = QVBoxLayout(self.simple_widget)
        simple_layout.setContentsMargins(0, 0, 0, 0)
        simple_layout.setSpacing(0)

        self.sound_wave_widget = SoundWaveWidget()
        simple_layout.addWidget(self.sound_wave_widget, alignment=Qt.AlignmentFlag.AlignCenter)

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
        dictation_label.setFont(theme.get_font(size=theme.config.fonts.xlarge, weight="semibold"))
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
        self.llm_label.setFont(theme.get_font(size=theme.config.fonts.xlarge, weight="semibold"))
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
        visual_label.setFont(theme.get_font(size=theme.config.fonts.xlarge, weight="semibold"))
        visual_layout.addWidget(visual_label)

        self.visual_dictation_box = QPlainTextEdit()
        self.visual_dictation_box.setReadOnly(True)
        visual_layout.addWidget(self.visual_dictation_box)

        self.visual_widget.setVisible(False)
        main_layout.addWidget(self.visual_widget, 1)

    def _apply_styling(self) -> None:
        """Apply QSS styling.

        Uses class-specific selectors to avoid affecting other QMainWindow,
        QLabel, or QPlainTextEdit instances in the application.
        """
        stylesheet = f"""
        QtDictationPopupView {{
            background-color: {theme.config.shapes.darkest};
            color: {theme.config.text.lightest};
            border: 2px solid {theme.config.blue.blue_2};
            border-radius: 8px;
        }}

        QtDictationPopupView QLabel {{
            color: {theme.config.text.light};
            font-size: {theme.config.fonts.medium}px;
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
            self._show_window_with_animation()
            # Animation runs automatically in widget

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
            self._show_window_with_animation()
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
            self._show_window_with_animation()
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
            self._hide_window_with_animation()
            self.current_mode = None

    def update_audio_level(self, level: float) -> None:
        """Update audio level for visualization - thread-safe."""
        self._signal_audio_level.emit(level)

    @Slot(float)
    def _do_update_audio_level(self, level: float) -> None:
        """Internal update audio level - MUST run on main Qt thread."""
        if self.current_mode == "simple" and hasattr(self, "sound_wave_widget"):
            self.sound_wave_widget.update_level(level)

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
        white_format.setForeground(QColor(theme.config.text.light))  # White for final
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
            # Don't call activateWindow() or setFocus() to prevent stealing focus from user's current task
            self.logger.debug(f"Dictation popup shown in {self.current_mode} mode")

    def _show_window_with_animation(self) -> None:
        """Show window with slide-up and fade-in animation."""
        # Cancel any existing animations
        if self._animation_in and self._animation_in.state() == QPropertyAnimation.State.Running:
            self._animation_in.stop()
        if self._animation_out and self._animation_out.state() == QPropertyAnimation.State.Running:
            self._animation_out.stop()

        # Use stored target geometry from _position_window
        if not hasattr(self, "_target_geometry") or self._target_geometry is None:
            # Fallback: just show without animation
            self.logger.warning("No target geometry stored, showing without animation")
            self._show_window()
            return

        target_geom = self._target_geometry
        self._final_position = (target_geom.x(), target_geom.y())

        # Calculate starting position: below the bottom of the screen
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QApplication

        primary_screen = QApplication.primaryScreen()
        if primary_screen:
            screen_geom = primary_screen.availableGeometry()
            # Start position: window completely below visible screen
            start_y = screen_geom.height() + 20  # 20px below screen for smooth entry
            start_geom = QRect(target_geom.x(), start_y, target_geom.width(), target_geom.height())
        else:
            # Fallback
            start_geom = target_geom

        # Set starting position BEFORE showing
        self.setGeometry(start_geom)

        # Show window with 0 opacity
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        # Don't call activateWindow() to prevent stealing focus from user's current task

        self.logger.info(f"Animation: starting y={start_geom.y()}, target y={target_geom.y()}, mode={self.current_mode}")

        # Create position animation (Y coordinate)
        self._animation_in = QPropertyAnimation(self, b"geometry")
        self._animation_in.setDuration(self._animation_duration_ms)
        self._animation_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation_in.setStartValue(start_geom)
        self._animation_in.setEndValue(target_geom)

        # Create opacity animation (fade in) - store as instance variable to prevent garbage collection
        self._opacity_animation_in = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_animation_in.setDuration(self._animation_duration_ms)
        self._opacity_animation_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._opacity_animation_in.setStartValue(0.0)
        self._opacity_animation_in.setEndValue(1.0)

        # Start both animations
        self._animation_in.start()
        self._opacity_animation_in.start()

        self.logger.info(f"Slide-up and fade-in animation started for {self.current_mode} mode")

    def _hide_window_with_animation(self) -> None:
        """Hide window with slide-down and fade-out animation."""
        # Cancel any existing animations
        if self._animation_in and self._animation_in.state() == QPropertyAnimation.State.Running:
            self._animation_in.stop()
        if self._animation_out and self._animation_out.state() == QPropertyAnimation.State.Running:
            self._animation_out.stop()
        if self._opacity_animation_in and self._opacity_animation_in.state() == QPropertyAnimation.State.Running:
            self._opacity_animation_in.stop()
        if self._opacity_animation_out and self._opacity_animation_out.state() == QPropertyAnimation.State.Running:
            self._opacity_animation_out.stop()

        if not self.isVisible():
            self.logger.debug("Window not visible, skipping hide animation")
            return

        # Get current geometry
        current_geom = self.geometry()

        # Calculate end position: slide down below bottom of screen
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QApplication

        primary_screen = QApplication.primaryScreen()
        if primary_screen:
            screen_geom = primary_screen.availableGeometry()
            end_y = screen_geom.height() + 20  # 20px below screen for smooth exit
            end_geom = QRect(current_geom.x(), end_y, current_geom.width(), current_geom.height())
        else:
            end_geom = current_geom

        self.logger.info(f"Window hiding from y={current_geom.y()} to y={end_geom.y()}, opacity={self.windowOpacity()}")

        # Create position animation (slide down)
        self._animation_out = QPropertyAnimation(self, b"geometry")
        self._animation_out.setDuration(self._animation_duration_ms)
        self._animation_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self._animation_out.setStartValue(current_geom)
        self._animation_out.setEndValue(end_geom)

        # Create opacity animation (fade out) - store as instance variable
        self._opacity_animation_out = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_animation_out.setDuration(self._animation_duration_ms)
        self._opacity_animation_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self._opacity_animation_out.setStartValue(self.windowOpacity())  # Use current opacity, not assuming 1.0
        self._opacity_animation_out.setEndValue(0.0)

        # Connect finish signal to actually hide the window (disconnect first to avoid duplicates)
        try:
            self._animation_out.finished.disconnect()
        except RuntimeError:
            pass  # No connections to disconnect
        self._animation_out.finished.connect(self._on_animation_finished)

        # Start both animations
        self._animation_out.start()
        self._opacity_animation_out.start()

        self.logger.info("Slide-down and fade-out animation started")

    def _on_animation_finished(self) -> None:
        """Called when hide animation finishes."""
        self.hide()
        self.setWindowOpacity(1.0)  # Reset opacity for next show

        # Disconnect to prevent duplicate calls
        try:
            if self._animation_out:
                self._animation_out.finished.disconnect(self._on_animation_finished)
        except RuntimeError:
            pass  # Already disconnected or no connection

        self.logger.info("Hide animation finished, window hidden")

    def _position_window(self, width: int, height: int, position_type: str = "center_left") -> None:
        """Calculate and store target window position for animation."""
        # Get primary screen for positioning
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QApplication

        primary_screen = QApplication.primaryScreen()

        if not primary_screen:
            # Fallback positioning
            self._target_geometry = QRect(100, 100, width, height)
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

        # Store target geometry for animation system (don't set yet - let animation handle it)
        self._target_geometry = QRect(x, y, width, height)
        self.logger.debug(f"Target position calculated: ({x}, {y}) with size ({width}, {height}), type={position_type}")

    def keyPressEvent(self, event) -> None:
        """Handle key press events - allow Escape to close."""
        if event.key() == Qt.Key.Key_Escape:
            self.hide_popup()
        else:
            super().keyPressEvent(event)
