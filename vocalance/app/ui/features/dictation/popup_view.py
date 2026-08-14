from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable, Optional, cast

from PySide6.QtCore import QEasingCurve, QMetaObject, QPointF, QPropertyAnimation, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QKeyEvent, QLinearGradient, QPainter, QPaintEvent, QPalette, QTextCharFormat
from PySide6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from vocalance.app.ui.components.labels import BoxTitleLabel
from vocalance.app.ui.components.sound_wave_widget import SoundWaveWidget
from vocalance.app.ui.components.spinner_widget import SpinnerWidget
from vocalance.app.ui.components.text_display import TextDisplayContainer
from vocalance.app.ui.qt_theme import theme


class QtDictationPopupView(QMainWindow):
    """Dictation popup for streaming transcription and dual-pane LLM output.

    Modes include simple listening, smart dictation + LLM, amend (prompt + LLM),
    and visual single-pane dictation. Updates are marshalled to the Qt main thread.
    """

    # Signals for thread-safe text updates
    partial_text_signal = Signal(str, str)  # text, segment_id
    final_text_signal = Signal(str, str)  # text, segment_id
    llm_token_signal = Signal(str)  # token
    audio_level_signal = Signal(float)  # audio level
    show_llm_processing_signal = Signal()  # Signal to show LLM processing on main thread
    modifier_banner_signal = Signal(str, bool)  # display_label, active

    # Window sizes (determined by the sound wave widget)
    SIMPLE_WIDTH = 60  # Exact fit for sound wave widget
    SIMPLE_HEIGHT = 30  # Exact fit for sound wave widget
    SMART_WIDTH = 800
    SMART_HEIGHT = 550
    VISUAL_WIDTH = 400
    VISUAL_HEIGHT = 550
    WINDOW_MARGIN_X = 80
    WINDOW_MARGIN_Y = 80
    DUAL_PANE_MODES: frozenset[str] = frozenset({"smart", "amend"})

    def __init__(self) -> None:
        """Initialize dictation popup view."""
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        # Thread safety
        self.ui_lock = threading.RLock()

        # Token buffering for smooth updates
        self.token_buffer = deque()
        self.last_flush_time = 0
        self.flush_interval_ms = 16  # ~60 FPS
        self.pending_flush = False

        self.current_mode: Optional[str] = None

        # Border color state (for stop word indication)
        self.border_is_orange = False

        # Animation properties
        self.animation_in = None
        self.animation_out = None
        self.opacity_animation_in = None
        self.opacity_animation_out = None
        self.final_position = None
        self.target_geometry = None
        self.animation_duration_ms = 400  # Animation duration in milliseconds
        self.partial_segments: dict[str, tuple[int, int]] = {}

        # Setup window
        self.setup_window()
        self.create_ui()
        self.apply_styling()

        # Connect signals for thread-safe updates
        self.partial_text_signal.connect(self.do_display_partial_text)
        self.final_text_signal.connect(self.do_display_final_text)
        self.llm_token_signal.connect(self.do_append_llm_token)
        self.audio_level_signal.connect(self.do_update_audio_level)
        self.show_llm_processing_signal.connect(self.do_show_llm_processing, Qt.ConnectionType.QueuedConnection)
        self.modifier_banner_signal.connect(self.do_set_modifier_banner, Qt.ConnectionType.QueuedConnection)

        self.modifier_fade_anim: Optional[QPropertyAnimation] = None

        self.logger.info("QtDictationPopupView initialized")

    def setup_window(self) -> None:
        """Configure window properties."""
        self.setWindowTitle("Dictation")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw rounded background with 3px gradient or orange border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        border_width = 3

        # Use orange border if stop word detected, otherwise use gradient
        if self.border_is_orange:
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

    def modifier_status_accent_color(self) -> str:
        gc = theme.config.text.gradient_colors
        return gc[1] if len(gc) > 1 else gc[0]

    def configure_reserved_modifier_status_label(self, label: QLabel) -> None:
        """Fixed-width slot to the right of dictation titles; opacity fades text without layout shift."""
        label.setFont(theme.get_font(size="medium", weight="semibold", display=False))
        pal = label.palette()
        pal.setColor(QPalette.ColorRole.WindowText, QColor(self.modifier_status_accent_color()))
        label.setPalette(pal)
        label.setMinimumWidth(200)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setText("")
        label.setVisible(True)
        eff = QGraphicsOpacityEffect(label)
        eff.setOpacity(0.0)
        label.setGraphicsEffect(eff)

    def create_ui(self) -> None:
        """Create UI elements."""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # Simple mode: Sound wave animation
        self.simple_widget = QWidget()
        self.simple_widget.setAutoFillBackground(False)
        self.simple_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        simple_layout = QVBoxLayout(self.simple_widget)
        simple_layout.setContentsMargins(0, 0, 0, 0)
        simple_layout.setSpacing(0)

        self.sound_wave_widget = SoundWaveWidget()
        simple_layout.addWidget(self.sound_wave_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        self.simple_widget.setVisible(False)
        main_layout.addWidget(self.simple_widget)

        # Smart mode: Dictation + AI output (side by side)
        self.smart_widget = QWidget()
        self.smart_widget.setAutoFillBackground(False)
        self.smart_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        smart_main_layout = QVBoxLayout(self.smart_widget)
        smart_main_layout.setContentsMargins(0, 0, 0, 0)
        smart_main_layout.setSpacing(10)

        # Container for side-by-side layout
        side_by_side_container = QWidget()
        side_by_side_container.setAutoFillBackground(False)
        side_by_side_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        side_by_side_layout = QHBoxLayout(side_by_side_container)
        side_by_side_layout.setContentsMargins(0, 0, 0, 0)
        side_by_side_layout.setSpacing(10)

        # Left column: Dictation
        dictation_container = QWidget()
        dictation_container.setAutoFillBackground(False)
        dictation_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        dictation_layout = QVBoxLayout(dictation_container)
        dictation_layout.setContentsMargins(0, 0, 0, 0)
        dictation_layout.setSpacing(5)

        self.dictation_title_row = QWidget()
        dtr_layout = QHBoxLayout(self.dictation_title_row)
        dtr_layout.setContentsMargins(0, 0, 14, 0)
        dtr_layout.setSpacing(theme.config.spacing.small)
        self.dictation_column_label = BoxTitleLabel("Dictation")
        self.dictation_column_label.setMinimumWidth(120)
        dtr_layout.addWidget(self.dictation_column_label, 0, Qt.AlignmentFlag.AlignLeft)
        self.dictation_modifier_status = QLabel("")
        self.configure_reserved_modifier_status_label(self.dictation_modifier_status)
        dtr_layout.addWidget(self.dictation_modifier_status, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dictation_layout.addWidget(self.dictation_title_row)

        # Text box for dictation - inside a dark rounded container
        dictation_container_widget = TextDisplayContainer()
        dictation_container_widget.setMinimumWidth(350)
        self.dictation_box = dictation_container_widget.text_edit
        dictation_layout.addWidget(dictation_container_widget, 1)

        side_by_side_layout.addWidget(dictation_container, 1)

        # Right column: AI Output
        llm_container = QWidget()
        llm_container.setAutoFillBackground(False)
        llm_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        llm_layout = QVBoxLayout(llm_container)
        llm_layout.setContentsMargins(0, 0, 0, 0)
        llm_layout.setSpacing(5)

        # Create title row with label on left, spinner on right
        llm_title_container = QWidget()
        llm_title_container.setAutoFillBackground(False)
        llm_title_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
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
        self.visual_widget.setAutoFillBackground(False)
        self.visual_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        visual_layout = QVBoxLayout(self.visual_widget)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(10)

        self.visual_title_row = QWidget()
        vtr_layout = QHBoxLayout(self.visual_title_row)
        vtr_layout.setContentsMargins(0, 0, 14, 0)
        vtr_layout.setSpacing(theme.config.spacing.small)
        self.visual_column_label = BoxTitleLabel("Dictation")
        self.visual_column_label.setMinimumWidth(120)
        vtr_layout.addWidget(self.visual_column_label, 0, Qt.AlignmentFlag.AlignLeft)
        self.visual_modifier_status = QLabel("")
        self.configure_reserved_modifier_status_label(self.visual_modifier_status)
        vtr_layout.addWidget(self.visual_modifier_status, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        visual_layout.addWidget(self.visual_title_row)

        # Text box for visual dictation - inside a dark rounded container
        visual_container_widget = TextDisplayContainer()
        self.visual_dictation_box = visual_container_widget.text_edit
        visual_layout.addWidget(visual_container_widget, 1)

        self.visual_widget.setVisible(False)
        main_layout.addWidget(self.visual_widget, 1)

    def apply_styling(self) -> None:
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
        QMetaObject.invokeMethod(self, "do_set_border_orange", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def do_set_border_orange(self) -> None:
        """Internal set border orange - MUST run on main Qt thread."""
        with self.ui_lock:
            self.border_is_orange = True
            self.update()  # Trigger repaint

    @Slot()
    def reset_border_color(self) -> None:
        """Reset border to gradient color - thread-safe."""
        QMetaObject.invokeMethod(self, "do_reset_border_color", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def do_reset_border_color(self) -> None:
        """Internal reset border color - MUST run on main Qt thread."""
        with self.ui_lock:
            self.border_is_orange = False
            self.update()  # Trigger repaint

    @Slot()
    def show_simple_listening(self) -> None:
        """Show simple listening indicator - thread-safe."""
        QMetaObject.invokeMethod(self, "do_show_simple", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def do_show_simple(self) -> None:
        """Internal show simple - MUST run on main Qt thread."""
        with self.ui_lock:
            self.hide_all_modes()
            self.border_is_orange = False  # Reset border color for new session
            self.simple_widget.setVisible(True)
            self.current_mode = "simple"
            self.position_window(self.SIMPLE_WIDTH, self.SIMPLE_HEIGHT, "bottom_left")
            self.show_window_with_animation()
            # Animation runs automatically in widget

    @Slot()
    def show_smart_dictation(self) -> None:
        """Show smart dictation (dictation + LLM output) - thread-safe."""
        QMetaObject.invokeMethod(self, "do_show_smart", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def show_amend_dictation(self) -> None:
        """Show amend mode: left = spoken instructions, right = LLM output (same layout as smart)."""
        QMetaObject.invokeMethod(self, "do_show_amend", Qt.ConnectionType.QueuedConnection)

    def apply_dual_pane_layout(self, mode: str, left_column_title: str) -> None:
        """Apply the dual-pane window layout on the Qt main thread (via calling slots)."""
        with self.ui_lock:
            self.hide_all_modes()
            self.border_is_orange = False
            self.current_mode = mode
            self.dictation_column_label.setText(left_column_title)
            self.smart_widget.setVisible(True)
            self.clear_smart_content()
            self.position_window(self.SMART_WIDTH, self.SMART_HEIGHT, "center_left")
            self.show_window_with_animation()
            self.logger.info(f"Dual-pane dictation shown, mode={self.current_mode}")

    @Slot()
    def do_show_smart(self) -> None:
        """Internal show smart - MUST run on main Qt thread."""
        self.apply_dual_pane_layout("smart", "Dictation")

    @Slot()
    def do_show_amend(self) -> None:
        """Internal show amend mode; must run on the Qt main thread."""
        self.apply_dual_pane_layout("amend", "Prompt")

    @Slot()
    def show_visual_dictation(self) -> None:
        """Show visual dictation (single pane) - thread-safe."""
        QMetaObject.invokeMethod(self, "do_show_visual", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def show_llm_processing(self) -> None:
        """Show LLM processing mode (keep smart layout, just update label) - thread-safe."""
        # Use QMetaObject.invokeMethod with QueuedConnection to marshal to main Qt thread
        QMetaObject.invokeMethod(self, "do_show_llm_processing", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def do_show_visual(self) -> None:
        """Internal show visual - MUST run on main Qt thread."""
        with self.ui_lock:
            self.hide_all_modes()
            self.border_is_orange = False  # Reset border color for new session
            self.current_mode = "visual"
            self.visual_widget.setVisible(True)
            self.clear_visual_content()
            self.position_window(self.VISUAL_WIDTH, self.VISUAL_HEIGHT, "center_left")
            self.show_window_with_animation()
            self.logger.info(f"Visual dictation window shown, mode={self.current_mode}")

    @Slot()
    def do_show_llm_processing(self) -> None:
        """Internal show LLM processing - MUST run on main Qt thread."""
        # Keep smart widget visible, just update the status
        # This is called after dictation stops and before LLM processing starts
        if self.current_mode in self.DUAL_PANE_MODES:
            self.llm_label.setText("Processing...")
            # Start spinner when LLM processing begins
            self.llm_spinner.start()
            self.logger.debug("Switched to LLM processing mode with spinner")

    @Slot()
    def hide_popup(self) -> None:
        """Hide the popup - thread-safe."""
        QMetaObject.invokeMethod(self, "do_hide_popup", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def do_hide_popup(self) -> None:
        """Internal hide popup - MUST run on main Qt thread."""
        with self.ui_lock:
            # Stop spinner when hiding popup
            self.llm_spinner.stop()
            self.reset_reserved_modifier_slot(self.dictation_modifier_status)
            self.reset_reserved_modifier_slot(self.visual_modifier_status)
            self.hide_window_with_animation()
            self.current_mode = None

    def update_audio_level(self, level: float) -> None:
        """Update audio level for visualization - thread-safe."""
        self.audio_level_signal.emit(level)

    @Slot(float)
    def do_update_audio_level(self, level: float) -> None:
        """Internal update audio level - MUST run on main Qt thread."""
        if self.current_mode == "simple":
            self.sound_wave_widget.update_level(level)

    def append_dictation_text(self, text: str) -> None:
        """Append text to dictation box - thread-safe."""
        self.logger.debug(f"append_dictation_text called with: '{text[:50]}...' (mode={self.current_mode})")
        # Use QTimer.singleShot for thread-safe GUI update
        QTimer.singleShot(0, lambda: self.do_append_dictation_text(text))

    def do_append_dictation_text(self, text: str) -> None:
        """Internal append dictation text - MUST run on main Qt thread."""
        self.logger.debug(f"_do_append_dictation_text executing: mode={self.current_mode}, text='{text[:50]}...'")
        if self.current_mode in self.DUAL_PANE_MODES:
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
        self.partial_text_signal.emit(text, segment_id)

    @Slot(str, str)
    def do_display_partial_text(self, text: str, segment_id: str) -> None:
        """Internal display partial text - MUST run on main Qt thread.

        Connected to _signal_partial_text for automatic thread marshalling.
        """
        self.logger.info(f"_do_display_partial_text EXECUTING: mode={self.current_mode}, text='{text[:50]}'")

        # Determine which text box to use based on mode
        text_box = None
        if self.current_mode in self.DUAL_PANE_MODES:
            text_box = self.dictation_box
            self.logger.info("Selected dictation_box for dual-pane mode")
        elif self.current_mode == "visual":
            text_box = self.visual_dictation_box
            self.logger.info("Selected visual_dictation_box for visual mode")

        if not text_box:
            self.logger.error("No text box for current mode; partial text ignored (mode=%s)", self.current_mode)
            return

        # Get document length for bounds checking
        doc_length = text_box.document().characterCount() - 1  # -1 for trailing newline

        # Remove all existing partial text segments
        for old_segment_id in list(self.partial_segments.keys()):
            old_start, old_end = self.partial_segments[old_segment_id]

            # Validate positions are within document bounds
            if old_start < 0 or old_end > doc_length or old_start >= old_end:
                self.logger.warning(
                    f"Skipping invalid partial segment {old_segment_id}: pos {old_start}-{old_end} (doc_length={doc_length})"
                )
                continue

            cursor = text_box.textCursor()
            cursor.setPosition(min(old_start, doc_length))
            cursor.setPosition(min(old_end, doc_length), cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            self.logger.debug(f"Removed old partial segment {old_segment_id} at pos {old_start}-{old_end}")

        # Clear the dictionary
        self.partial_segments.clear()

        # Now insert new partial text at end with MEDIUM color formatting
        cursor = text_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)

        # Create format for partial text (medium color = visible/current input)
        partial_format = QTextCharFormat()
        partial_format.setForeground(QColor(theme.config.text.medium))
        partial_format.setBackground(QBrush(Qt.BrushStyle.NoBrush))

        cursor.setCharFormat(partial_format)

        # Store position before insertion
        start_pos = cursor.position()

        # Insert text with the format explicitly set on cursor
        cursor.insertText(text)
        end_pos = cursor.position()

        # Store this segment's position for removal when final text arrives
        self.partial_segments[segment_id] = (start_pos, end_pos)

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
        self.final_text_signal.emit(text, segment_id)

    @Slot(str, str)
    def do_display_final_text(self, text: str, segment_id: str) -> None:
        """Internal display final text - MUST run on main Qt thread.

        Connected to _signal_final_text for automatic thread marshalling.
        """
        self.logger.info(f"_do_display_final_text EXECUTING: mode={self.current_mode}, text='{text[:50]}'")

        # Determine which text box to use based on mode
        text_box = None
        if self.current_mode in self.DUAL_PANE_MODES:
            text_box = self.dictation_box
            self.logger.info("Selected dictation_box for dual-pane mode")
        elif self.current_mode == "visual":
            text_box = self.visual_dictation_box
            self.logger.info("Selected visual_dictation_box for visual mode")

        if not text_box:
            self.logger.error("No text box for current mode; final text ignored (mode=%s)", self.current_mode)
            return

        # Remove ALL partial segments before inserting final text. The partial event always
        # uses segment_id="" while the final event uses a UUID, so they never match by id.
        # Clearing all partials ensures there is no gray text visible when white is appended.
        doc_length = text_box.document().characterCount() - 1  # -1 for trailing newline
        for old_id in list(self.partial_segments.keys()):
            old_start, old_end = self.partial_segments[old_id]
            if old_start >= 0 and old_end <= doc_length and old_start < old_end:
                cursor = text_box.textCursor()
                cursor.setPosition(min(old_start, doc_length))
                cursor.setPosition(min(old_end, doc_length), cursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
                self.logger.debug(f"Removed partial segment {old_id} at {old_start}-{old_end} before inserting final")
            else:
                self.logger.warning(
                    f"Skipping invalid partial segment {old_id}: pos {old_start}-{old_end} (doc_length={doc_length})"
                )
        self.partial_segments.clear()

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
        self.llm_token_signal.emit(token)

    @Slot(str)
    def do_append_llm_token(self, token: str) -> None:
        """Internal append LLM token - MUST run on main Qt thread.

        Connected to _signal_llm_token for automatic thread marshalling.
        Buffers tokens and flushes when threshold reached for smooth 60fps updates.
        """
        self.logger.debug(f"_do_append_llm_token EXECUTING on main thread: token='{token}', mode={self.current_mode}")

        if self.current_mode not in self.DUAL_PANE_MODES:
            self.logger.warning(f"_do_append_llm_token called but mode is '{self.current_mode}'")
            return

        # Buffer the token
        with self.ui_lock:
            self.token_buffer.append(token)

            # Schedule flush if buffer has enough tokens and no flush pending
            if len(self.token_buffer) >= 3 and not self.pending_flush:
                self.pending_flush = True
                # Safe to use QTimer.singleShot here - we're on the main Qt thread
                QTimer.singleShot(1, self.flush_token_buffer)
                self.logger.debug(f"Scheduled token buffer flush ({len(self.token_buffer)} tokens buffered)")

    def flush_token_buffer(self) -> None:
        """Flush buffered tokens to LLM output box with color formatting.

        Must be called from main Qt thread only (scheduled via QTimer.singleShot).
        Only the last token is shown in medium color (fading effect).
        All other historical tokens are shown in lightest color.
        """
        self.logger.debug(f"_flush_token_buffer CALLED: mode={self.current_mode}, buffer_size={len(self.token_buffer)}")

        if self.current_mode not in self.DUAL_PANE_MODES:
            with self.ui_lock:
                self.token_buffer.clear()
                self.pending_flush = False
            self.logger.warning(f"_flush_token_buffer aborted - wrong mode: {self.current_mode}")
            return

        # Get batched tokens
        with self.ui_lock:
            if not self.token_buffer:
                self.pending_flush = False
                self.logger.debug("_flush_token_buffer: no tokens to flush")
                return

            batched = "".join(self.token_buffer)
            token_count = len(self.token_buffer)
            self.token_buffer.clear()

        # Insert into LLM box
        if not self.llm_box:
            self.logger.error("_flush_token_buffer: llm_box is None!")
            with self.ui_lock:
                self.pending_flush = False
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

        with self.ui_lock:
            self.pending_flush = False

    def update_llm_status(self, status: str) -> None:
        """Update LLM output label status and manage spinner.

        Args:
            status: Status text to display. If "Complete!" or similar, stops spinner.
        """
        if self.current_mode in self.DUAL_PANE_MODES:
            self.llm_label.setText(status)
            if status in ("Complete!", "AI Output", "Error"):
                self.llm_spinner.stop()
            elif status == "Processing...":
                self.llm_spinner.start()

    # Internal methods

    def set_modifier_banner(self, display_label: str, active: bool) -> None:
        """Show or hide modifier status with fade (thread-safe)."""
        self.modifier_banner_signal.emit(display_label, active)

    def fade_modifier_label_opacity(
        self,
        label: QLabel,
        end: float,
        on_finished: Optional[Callable[[], None]] = None,
    ) -> None:
        eff = label.graphicsEffect()
        if not isinstance(eff, QGraphicsOpacityEffect):
            return
        anim = QPropertyAnimation(eff, b"opacity", label)
        anim.setDuration(self.animation_duration_ms if end > 0.5 else max(150, self.animation_duration_ms // 2))
        anim.setStartValue(eff.opacity())
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic if end > eff.opacity() else QEasingCurve.Type.InCubic)
        if on_finished is not None:
            anim.finished.connect(on_finished)
        anim.start()
        self.modifier_fade_anim = anim

    def reset_reserved_modifier_slot(self, label: QLabel) -> None:
        label.setText("")
        eff = label.graphicsEffect()
        if isinstance(eff, QGraphicsOpacityEffect):
            eff.setOpacity(0.0)

    @Slot(str, bool)
    def do_set_modifier_banner(self, display_label: str, active: bool) -> None:
        """Show or clear the modifier label on smart/amend/visual layouts; no-op for wave-only modes."""
        if self.modifier_fade_anim and self.modifier_fade_anim.state() == QPropertyAnimation.State.Running:
            self.modifier_fade_anim.stop()

        reserved_slots = (self.dictation_modifier_status, self.visual_modifier_status)

        if not active:
            for lbl in reserved_slots:
                self.reset_reserved_modifier_slot(lbl)
            return

        for lbl in reserved_slots:
            self.reset_reserved_modifier_slot(lbl)

        if not display_label.strip():
            return

        mode = self.current_mode
        if mode in self.DUAL_PANE_MODES:
            chip = self.dictation_modifier_status
            chip.setText(display_label)
        elif mode == "visual":
            chip = self.visual_modifier_status
            chip.setText(display_label)
        else:
            return

        eff = cast(QGraphicsOpacityEffect, chip.graphicsEffect())
        eff.setOpacity(0.0)
        self.fade_modifier_label_opacity(chip, 1.0)

    def hide_all_modes(self) -> None:
        """Hide all mode widgets."""
        self.simple_widget.setVisible(False)
        self.smart_widget.setVisible(False)
        self.visual_widget.setVisible(False)
        self.reset_reserved_modifier_slot(self.dictation_modifier_status)
        self.reset_reserved_modifier_slot(self.visual_modifier_status)

    def clear_smart_content(self) -> None:
        """Clear smart mode content."""
        self.dictation_box.clear()
        self.llm_box.clear()
        self.llm_label.setText("AI Output")
        self.llm_spinner.stop()
        with self.ui_lock:
            self.token_buffer.clear()

    def clear_visual_content(self) -> None:
        """Clear visual mode content."""
        self.visual_dictation_box.clear()

    def show_window(self) -> None:
        """Show the window at top level (matches legacy behavior)."""
        if not self.isVisible():
            self.show()
            self.raise_()
            # Don't call activateWindow() or setFocus() to prevent stealing focus from user's current task
            self.logger.debug(f"Dictation popup shown in {self.current_mode} mode")

    def show_window_with_animation(self) -> None:
        """Show window with slide-up and fade-in animation."""
        # Cancel any existing animations
        if self.animation_in and self.animation_in.state() == QPropertyAnimation.State.Running:
            self.animation_in.stop()
        if self.animation_out and self.animation_out.state() == QPropertyAnimation.State.Running:
            self.animation_out.stop()

        if self.target_geometry is None:
            self.logger.warning("No target geometry stored, showing without animation")
            self.show_window()
            return

        target_geom = self.target_geometry
        self.final_position = (target_geom.x(), target_geom.y())

        # Calculate starting position: below the bottom of the screen
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QApplication

        primary_screen = QApplication.primaryScreen()
        if primary_screen:
            screen_geom = primary_screen.availableGeometry()
            start_y = screen_geom.y() + screen_geom.height() + 20
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
        self.animation_in = QPropertyAnimation(self, b"geometry")
        self.animation_in.setDuration(self.animation_duration_ms)
        self.animation_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation_in.setStartValue(start_geom)
        self.animation_in.setEndValue(target_geom)

        # Create opacity animation (fade in) - store as instance variable to prevent garbage collection
        self.opacity_animation_in = QPropertyAnimation(self, b"windowOpacity")
        self.opacity_animation_in.setDuration(self.animation_duration_ms)
        self.opacity_animation_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.opacity_animation_in.setStartValue(0.0)
        self.opacity_animation_in.setEndValue(1.0)

        # Start both animations
        self.animation_in.start()
        self.opacity_animation_in.start()

        self.logger.info(f"Slide-up and fade-in animation started for {self.current_mode} mode")

    def hide_window_with_animation(self) -> None:
        """Hide window with slide-down and fade-out animation."""
        # Cancel any existing animations
        if self.animation_in and self.animation_in.state() == QPropertyAnimation.State.Running:
            self.animation_in.stop()
        if self.animation_out and self.animation_out.state() == QPropertyAnimation.State.Running:
            self.animation_out.stop()
        if self.opacity_animation_in and self.opacity_animation_in.state() == QPropertyAnimation.State.Running:
            self.opacity_animation_in.stop()
        if self.opacity_animation_out and self.opacity_animation_out.state() == QPropertyAnimation.State.Running:
            self.opacity_animation_out.stop()

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
            end_y = screen_geom.y() + screen_geom.height() + 20
            end_geom = QRect(current_geom.x(), end_y, current_geom.width(), current_geom.height())
        else:
            end_geom = current_geom

        self.logger.info(f"Window hiding from y={current_geom.y()} to y={end_geom.y()}, opacity={self.windowOpacity()}")

        # Create position animation (slide down)
        self.animation_out = QPropertyAnimation(self, b"geometry")
        self.animation_out.setDuration(self.animation_duration_ms)
        self.animation_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self.animation_out.setStartValue(current_geom)
        self.animation_out.setEndValue(end_geom)

        # Create opacity animation (fade out) - store as instance variable
        self.opacity_animation_out = QPropertyAnimation(self, b"windowOpacity")
        self.opacity_animation_out.setDuration(self.animation_duration_ms)
        self.opacity_animation_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self.opacity_animation_out.setStartValue(self.windowOpacity())  # Use current opacity, not assuming 1.0
        self.opacity_animation_out.setEndValue(0.0)

        # Connect finish signal to actually hide the window
        self.animation_out.finished.connect(self.on_animation_finished)

        # Start both animations
        self.animation_out.start()
        self.opacity_animation_out.start()

        self.logger.info("Slide-down and fade-out animation started")

    def on_animation_finished(self) -> None:
        """Called when hide animation finishes."""
        self.hide()
        self.setWindowOpacity(1.0)  # Reset opacity for next show

        # Disconnect to prevent duplicate calls
        try:
            if self.animation_out:
                self.animation_out.finished.disconnect(self.on_animation_finished)
        except RuntimeError:
            pass  # Already disconnected or no connection

        self.logger.info("Hide animation finished, window hidden")

    def position_window(self, width: int, height: int, position_type: str = "center_left") -> None:
        """Calculate and store target window position for animation."""
        # Get primary screen for positioning
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QApplication

        primary_screen = QApplication.primaryScreen()

        if not primary_screen:
            # Fallback positioning
            self.target_geometry = QRect(100, 100, width, height)
            self.logger.warning("No screen available for positioning, using fallback")
            return

        # Use available geometry (excludes taskbar) for positioning calculations
        screen_geom = primary_screen.availableGeometry()

        sx = screen_geom.x()
        sy = screen_geom.y()
        sw = screen_geom.width()
        sh = screen_geom.height()

        if position_type == "bottom_left":
            x = sx + self.WINDOW_MARGIN_X
            y = sy + sh - height - self.WINDOW_MARGIN_Y
        elif position_type == "center_left":
            x = sx + self.WINDOW_MARGIN_X
            y = sy + (sh - height) // 2
        else:
            x = sx + (sw - width) // 2
            y = sy + (sh - height) // 2

        # Store target geometry for animation system (don't set yet - let animation handle it)
        self.target_geometry = QRect(x, y, width, height)
        self.logger.debug(f"Target position calculated: ({x}, {y}) with size ({width}, {height}), type={position_type}")

    def keyPressEvent(self, key_event: QKeyEvent) -> None:
        """Handle key press events - allow Escape to close."""
        if key_event.key() == Qt.Key.Key_Escape:
            self.hide_popup()
        else:
            super().keyPressEvent(key_event)
