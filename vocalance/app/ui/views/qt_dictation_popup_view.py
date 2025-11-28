"""Qt-based dictation popup view for streaming transcription display.

Provides real-time dictation text display with three modes:
- Simple: Listening indicator with spinner animation
- Smart: Dictation pane + AI output pane
- Visual: Single dictation pane for visual commands
"""

import logging
import threading
from collections import deque

from PySide6.QtCore import QEasingCurve, QMetaObject, QPointF, QPropertyAnimation, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QTextCharFormat
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from vocalance.app.ui.components.labels import BoxTitleLabel
from vocalance.app.ui.components.sound_wave_widget import SoundWaveWidget
from vocalance.app.ui.components.spinner_widget import SpinnerWidget
from vocalance.app.ui.components.text_display import TextDisplayContainer
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
    _signal_show_llm_processing = Signal()  # Signal to show LLM processing on main thread

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

        # Border color state (for stop word indication)
        self._border_is_orange = False

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
        self._signal_show_llm_processing.connect(self._do_show_llm_processing, Qt.ConnectionType.QueuedConnection)

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
        """Draw rounded background with 3px gradient or orange border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        border_width = 3

        # Use orange border if stop word detected, otherwise use gradient
        if self._border_is_orange:
            # Solid orange border
            painter.setBrush(QColor(theme.config.shapes.orange))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 16, 16)
        else:
            # Create gradient for border
            gradient_colors = theme.config.text.gradient_colors
            gradient = QLinearGradient(QPointF(0, 0), QPointF(rect.width(), rect.height()))
            gradient.setColorAt(0, QColor(gradient_colors[0]))
            gradient.setColorAt(1, QColor(gradient_colors[1]))

            # Draw outer rounded rect with gradient (this is the border)
            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 16, 16)

        # Draw inner rounded rect with background color (creates border effect)
        inner_rect = rect.adjusted(border_width, border_width, -border_width, -border_width)
        painter.setBrush(QColor(theme.config.shapes.darkest))
        painter.drawRoundedRect(inner_rect, 12, 12)

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

        # Smart mode: Dictation + AI output (side by side)
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

        # Create title label with xlarge font and gradient
        dictation_label = BoxTitleLabel("Dictation")
        # Ensure enough horizontal space for gradient calculation
        dictation_label.setMinimumWidth(200)
        dictation_layout.addWidget(dictation_label)

        # Text box for dictation - inside a dark rounded container
        dictation_container_widget = TextDisplayContainer()
        dictation_container_widget.setMinimumWidth(350)
        self.dictation_box = dictation_container_widget.text_edit
        dictation_layout.addWidget(dictation_container_widget, 1)

        side_by_side_layout.addWidget(dictation_container, 1)

        # Right column: AI Output
        llm_container = QWidget()
        llm_layout = QVBoxLayout(llm_container)
        llm_layout.setContentsMargins(0, 0, 0, 0)
        llm_layout.setSpacing(5)

        # Create title row with label on left, spinner on right
        llm_title_container = QWidget()
        llm_title_layout = QHBoxLayout(llm_title_container)
        # Right margin matches the padding of the LLM text box (spacing.small = 8px)
        llm_title_layout.setContentsMargins(0, 0, 8, 0)
        llm_title_layout.setSpacing(8)

        # Create title label with xlarge font and gradient (left side)
        self.llm_label = BoxTitleLabel("AI Output")
        self.llm_label.setMinimumWidth(200)
        llm_title_layout.addWidget(self.llm_label)

        # Add stretch to push spinner to the right
        llm_title_layout.addStretch()

        # Create spinner widget on the right (starts hidden)
        self.llm_spinner = SpinnerWidget(parent=llm_title_container, size=24)
        self.llm_spinner.setVisible(False)
        llm_title_layout.addWidget(self.llm_spinner, alignment=Qt.AlignmentFlag.AlignVCenter)

        llm_layout.addWidget(llm_title_container)

        # Text box for LLM output - inside a dark rounded container
        llm_container_widget = TextDisplayContainer()
        llm_container_widget.setMinimumWidth(350)
        self.llm_box = llm_container_widget.text_edit
        llm_layout.addWidget(llm_container_widget, 1)

        side_by_side_layout.addWidget(llm_container, 1)

        smart_main_layout.addWidget(side_by_side_container, 1)

        self.smart_widget.setVisible(False)
        main_layout.addWidget(self.smart_widget, 1)

        # Visual mode: Dictation only
        self.visual_widget = QWidget()
        visual_layout = QVBoxLayout(self.visual_widget)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(10)

        # Create title label with xlarge font and gradient
        visual_label = BoxTitleLabel("Dictation")
        # Ensure enough horizontal space for gradient calculation
        visual_label.setMinimumWidth(200)
        visual_layout.addWidget(visual_label)

        # Text box for visual dictation - inside a dark rounded container
        visual_container_widget = TextDisplayContainer()
        self.visual_dictation_box = visual_container_widget.text_edit
        visual_layout.addWidget(visual_container_widget, 1)

        self.visual_widget.setVisible(False)
        main_layout.addWidget(self.visual_widget, 1)

    def _apply_styling(self) -> None:
        """Apply QSS styling.

        Note: Window background/border is handled in paintEvent, not QSS.
        Only setting inherited properties here.
        Component-specific stylesheets (like for labels) handle their own styling.
        Avoid window-level color property as it can interfere with custom paintEvent
        gradient rendering in child widgets.
        """
        # No stylesheet needed - individual components handle their own styling

    # Public API

    @Slot()
    def set_border_orange(self) -> None:
        """Set border to orange (stop word detected) - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_set_border_orange", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_set_border_orange(self) -> None:
        """Internal set border orange - MUST run on main Qt thread."""
        with self._ui_lock:
            self._border_is_orange = True
            self.update()  # Trigger repaint

    @Slot()
    def reset_border_color(self) -> None:
        """Reset border to gradient color - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_reset_border_color", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_reset_border_color(self) -> None:
        """Internal reset border color - MUST run on main Qt thread."""
        with self._ui_lock:
            self._border_is_orange = False
            self.update()  # Trigger repaint

    @Slot()
    def show_simple_listening(self) -> None:
        """Show simple listening indicator - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_show_simple", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_show_simple(self) -> None:
        """Internal show simple - MUST run on main Qt thread."""
        with self._ui_lock:
            self._hide_all_modes()
            self._border_is_orange = False  # Reset border color for new session
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
            self._border_is_orange = False  # Reset border color for new session
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
        # Use QMetaObject.invokeMethod with QueuedConnection to marshal to main Qt thread
        QMetaObject.invokeMethod(self, "_do_show_llm_processing", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_show_visual(self) -> None:
        """Internal show visual - MUST run on main Qt thread."""
        with self._ui_lock:
            self._hide_all_modes()
            self._border_is_orange = False  # Reset border color for new session
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
            # Start spinner when LLM processing begins
            if hasattr(self, "llm_spinner"):
                self.llm_spinner.start()
            self.logger.debug("Switched to LLM processing mode with spinner")

    @Slot()
    def hide_popup(self) -> None:
        """Hide the popup - thread-safe."""
        QMetaObject.invokeMethod(self, "_do_hide_popup", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _do_hide_popup(self) -> None:
        """Internal hide popup - MUST run on main Qt thread."""
        with self._ui_lock:
            # Stop spinner when hiding popup
            if hasattr(self, "llm_spinner"):
                self.llm_spinner.stop()
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

        # Now insert new partial text at end with MEDIUM color formatting
        cursor = text_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)

        # Create format for partial text (medium color = visible/current input)
        partial_format = QTextCharFormat()
        partial_format.setForeground(QColor(theme.config.text.medium))
        partial_format.setBackground(QBrush(Qt.BrushStyle.NoBrush))

        # CRITICAL: Reset cursor format to partial format (not inheriting from previous text)
        # This is at source - we explicitly set what format should be used
        cursor.setCharFormat(partial_format)

        # Store position before insertion
        start_pos = cursor.position()

        # Insert text with the format explicitly set on cursor
        cursor.insertText(text)
        end_pos = cursor.position()

        # Store this segment's position for removal when final text arrives
        self._partial_segments[segment_id] = (start_pos, end_pos)

        text_box.setTextCursor(cursor)
        text_box.ensureCursorVisible()
        self.logger.debug(f"Displayed medium-colored partial text at {start_pos}-{end_pos}: '{text[:30]}...'")

    def display_final_text(self, text: str, segment_id: str) -> None:
        """Display final (stable) text in lightest color for streaming dictation.

        Final text replaces any partial text with the same segment_id and
        is shown in lightest color to indicate it will no longer change.
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

        # Insert final text at end with light color formatting (stable/permanent)
        cursor = text_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)

        # Create character format for light color text (final = stable)
        final_format = QTextCharFormat()
        final_format.setForeground(QColor(theme.config.text.light))  # Light color for final
        final_format.setBackground(QBrush(Qt.BrushStyle.NoBrush))  # No background (transparent)
        cursor.setCharFormat(final_format)

        # Insert text with trailing space (matches legacy line 315)
        if text:
            cursor.insertText(text + " ")

        # Ensure format is applied
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.clearSelection()

        text_box.setTextCursor(cursor)
        text_box.ensureCursorVisible()
        self.logger.debug(f"Displayed lightest-colored final text: '{text[:30]}...'")

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
        """Flush buffered tokens to LLM output box with color formatting.

        Must be called from main Qt thread only (scheduled via QTimer.singleShot).
        Only the last token is shown in medium color (fading effect).
        All other historical tokens are shown in lightest color.
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

        # Get the full text with new tokens appended
        cursor = self.llm_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(batched)

        # Now reformat ALL text: lightest for all, but medium for the LAST token
        full_text = self.llm_box.toPlainText()

        if full_text:
            # Select all text and set to light color first
            cursor.select(cursor.SelectionType.Document)
            light_format = QTextCharFormat()
            light_format.setForeground(QColor(theme.config.text.light))
            light_format.setBackground(QBrush(Qt.BrushStyle.NoBrush))
            cursor.setCharFormat(light_format)

            # Clear selection before formatting last tokens
            cursor.clearSelection()

            # Now format just the last token in medium color
            if batched:  # Only format if we added new tokens
                last_token_start = len(full_text) - len(batched)
                cursor.setPosition(last_token_start)
                cursor.setPosition(len(full_text), cursor.MoveMode.KeepAnchor)

                medium_format = QTextCharFormat()
                medium_format.setForeground(QColor(theme.config.text.medium))
                medium_format.setBackground(QBrush(Qt.BrushStyle.NoBrush))
                cursor.setCharFormat(medium_format)

        # Move cursor to end for proper positioning
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.clearSelection()
        self.llm_box.setTextCursor(cursor)
        self.llm_box.ensureCursorVisible()

        self.logger.debug(f"_flush_token_buffer: flushed {token_count} tokens ('{batched[:50]}...')")

        with self._ui_lock:
            self._pending_flush = False

    def update_llm_status(self, status: str) -> None:
        """Update LLM output label status and manage spinner.

        Args:
            status: Status text to display. If "Complete!" or similar, stops spinner.
        """
        if self.current_mode == "smart":
            self.llm_label.setText(status)
            # Stop spinner when LLM completes
            if hasattr(self, "llm_spinner"):
                if status in ("Complete!", "AI Output", "Error"):
                    self.llm_spinner.stop()
                elif status == "Processing...":
                    self.llm_spinner.start()

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
        # Ensure spinner is hidden when clearing
        if hasattr(self, "llm_spinner"):
            self.llm_spinner.stop()
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
