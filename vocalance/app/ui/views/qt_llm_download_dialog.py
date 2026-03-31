"""Modal progress UI for cancellable Hugging Face GGUF downloads."""

from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout

from vocalance.app.ui.components.buttons import DangerButton
from vocalance.app.ui.qt_theme import theme


class LlmDownloadProgressDialog(QDialog):
    """Indeterminate progress, status line, spinner, and cancel (same visual language as startup)."""

    cancel_clicked = Signal()

    _SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, parent=None, model_label: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Download language model")
        self.setModal(True)
        self._frame_i = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_spinner)
        self._outcome: Optional[Tuple[bool, str]] = None
        self._setup_ui(model_label)

    def _setup_ui(self, model_label: str) -> None:
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

        self._hint = QLabel("You can cancel anytime. Partial files for this model will be removed.")
        self._hint.setFont(theme.get_font(size=theme.config.fonts.small))
        self._hint.setWordWrap(True)
        hp = self._hint.palette()
        hp.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.medium))
        self._hint.setPalette(hp)
        layout.addWidget(self._hint)

        self._bar = QProgressBar(self)
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(3)
        self._bar.setStyleSheet(
            f"""
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
        )
        layout.addWidget(self._bar)

        status_row = QHBoxLayout()
        status_row.setSpacing(theme.config.spacing.small)
        self._status = QLabel("Preparing download…")
        self._status.setFont(theme.get_font(size=theme.config.fonts.small))
        sp = self._status.palette()
        sp.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.light))
        self._status.setPalette(sp)
        self._status.setWordWrap(True)
        status_row.addWidget(self._status, stretch=1)

        self._spinner = QLabel(self._SPINNER_FRAMES[0])
        self._spinner.setFont(theme.get_font(size=theme.config.fonts.large))
        ssp = self._spinner.palette()
        ssp.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.blue.blue_2))
        self._spinner.setPalette(ssp)
        self._spinner.setFixedWidth(28)
        self._spinner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_row.addWidget(self._spinner)
        layout.addLayout(status_row)

        cancel_btn = DangerButton(text="Cancel", command=self._on_cancel_clicked)
        layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignLeft)

    def _on_cancel_clicked(self) -> None:
        self.cancel_clicked.emit()

    def _tick_spinner(self) -> None:
        self._frame_i = (self._frame_i + 1) % len(self._SPINNER_FRAMES)
        self._spinner.setText(self._SPINNER_FRAMES[self._frame_i])

    def showEvent(self, event) -> None:
        self._timer.start(80)
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self._outcome is None:
            self.cancel_clicked.emit()
            event.ignore()
            return
        super().closeEvent(event)

    def set_status(self, message: str) -> None:
        self._status.setText(message)

    def apply_outcome(self, ok: bool, message: str) -> None:
        self._timer.stop()
        self._outcome = (ok, message)
        self.accept()

    @property
    def outcome(self) -> Optional[Tuple[bool, str]]:
        return self._outcome
