import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox, QWidget

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDeviceErrorEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtSystemController(QtBaseController[QWidget]):
    """Surfaces global audio-device failures as a modal on ``main_window``."""

    audio_device_error = Signal(str)

    def __init__(self, event_bus: EventBus, main_window: QWidget) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger(self.__class__.__name__))
        self.main_window = main_window
        self.subscribe(AudioDeviceErrorEvent, self._on_audio_device_error)
        self.audio_device_error.connect(self._show_microphone_error_dialog)

    def _on_audio_device_error(self, device_error: AudioDeviceErrorEvent) -> None:
        self.logger.warning("Audio device error: %s", device_error.error_message)
        self.audio_device_error.emit(device_error.error_message)

    def _show_microphone_error_dialog(self, error_message: str) -> None:
        msg = QMessageBox(self.main_window)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Microphone unavailable")
        msg.setText(error_message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def shutdown(self) -> None:
        self.main_window = None
        super().shutdown()
