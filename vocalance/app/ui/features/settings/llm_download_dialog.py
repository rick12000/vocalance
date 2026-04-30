from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QPalette, QShowEvent
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from vocalance.app.ui.components.buttons import DangerButton
from vocalance.app.ui.qt_theme import theme


class LlmDownloadProgressDialog(QDialog):
    """Indeterminate progress, status line, spinner, and cancel (same visual language as startup)."""

    cancel_clicked = Signal()

    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, parent: Optional[QWidget] = None, model_label: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Download language model")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.spinner_frame_index = 0
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self.tick_spinner)
        self.final_outcome: Optional[Tuple[bool, str]] = None
        self.setup_ui(model_label)

    def setup_ui(self, model_label: str) -> None:
        self.setMinimumWidth(440)
        dlg_palette = self.palette()
        dlg_palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.dark))
        dlg_palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.light))
        self.setPalette(dlg_palette)
        self.setAutoFillBackground(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.config.spacing.medium)
        layout.setContentsMargins(
            theme.config.spacing.large,
            theme.config.spacing.large,
            theme.config.spacing.large,
            theme.config.spacing.large,
        )

        title = QLabel(model_label.strip() or "Downloading model")
        title.setFont(theme.get_font(size=theme.config.fonts.medium, weight="semibold"))
        title.setWordWrap(True)
        tp = title.palette()
        tp.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.light))
        title.setPalette(tp)
        layout.addWidget(title)

        self.hint_label = QLabel("You can cancel anytime. Partial files for this model will be removed.")
        self.hint_label.setFont(theme.get_font(size=theme.config.fonts.small))
        self.hint_label.setWordWrap(True)
        hp = self.hint_label.palette()
        hp.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.medium))
        self.hint_label.setPalette(hp)
        layout.addWidget(self.hint_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(3)
        layout.addWidget(self.progress_bar)

        status_row = QHBoxLayout()
        status_row.setSpacing(theme.config.spacing.small)
        self.status_label = QLabel("Preparing download…")
        self.status_label.setFont(theme.get_font(size=theme.config.fonts.small))
        sp = self.status_label.palette()
        sp.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.light))
        self.status_label.setPalette(sp)
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, stretch=1)

        self.spinner_label = QLabel(self.SPINNER_FRAMES[0])
        self.spinner_label.setFont(theme.get_font(size=theme.config.fonts.large))
        ssp = self.spinner_label.palette()
        ssp.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.blue.blue_2))
        self.spinner_label.setPalette(ssp)
        self.spinner_label.setFixedWidth(28)
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_row.addWidget(self.spinner_label)
        layout.addLayout(status_row)

        cancel_btn = DangerButton(text="Cancel", command=self.on_cancel_clicked)
        layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignLeft)

    def on_cancel_clicked(self) -> None:
        self.cancel_clicked.emit()

    def tick_spinner(self) -> None:
        self.spinner_frame_index = (self.spinner_frame_index + 1) % len(self.SPINNER_FRAMES)
        self.spinner_label.setText(self.SPINNER_FRAMES[self.spinner_frame_index])

    def showEvent(self, show_event: QShowEvent) -> None:
        self.spinner_timer.start(80)
        super().showEvent(show_event)

    def closeEvent(self, close_event: QCloseEvent) -> None:
        self.spinner_timer.stop()
        if self.final_outcome is None:
            self.cancel_clicked.emit()
            close_event.ignore()
            return
        super().closeEvent(close_event)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def apply_outcome(self, ok: bool, message: str) -> None:
        self.spinner_timer.stop()
        self.final_outcome = (ok, message)
        self.accept()

    @property
    def outcome(self) -> Optional[Tuple[bool, str]]:
        return self.final_outcome
