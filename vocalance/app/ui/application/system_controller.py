import logging
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDeviceErrorEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtSystemController(QtBaseController):
    """System-level Qt bridge: audio device errors and critical dialogs."""

    audio_device_error = Signal(str)

    def __init__(self, event_bus: EventBus, main_window: Any) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger(self.__class__.__name__))

        self.main_window = main_window
        self._setup_subscriptions()
        self._connect_signals()

        self.logger.debug("QtSystemController initialized")

    def _setup_subscriptions(self) -> None:
        try:
            self.event_bus.subscribe(event_type=AudioDeviceErrorEvent, handler=self._handle_audio_device_error)
            self.logger.debug("System controller event subscriptions configured")
        except Exception as e:
            self.logger.error("Error setting up event subscriptions: %s", e, exc_info=True)

    def _connect_signals(self) -> None:
        self.audio_device_error.connect(self._show_audio_device_error_dialog)

    def _handle_audio_device_error(self, device_error: AudioDeviceErrorEvent) -> None:
        try:
            self.logger.warning("Audio device error: %s", device_error.error_message)
            self.audio_device_error.emit(device_error.error_message)
        except Exception as e:
            self.logger.error("Error handling audio device error: %s", e, exc_info=True)

    def _show_audio_device_error_dialog(self, error_message: str) -> None:
        try:
            msg = QMessageBox(self.main_window)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Microphone unavailable")
            msg.setText(error_message)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            self.logger.info("Audio device error dialog shown to user")
        except Exception as e:
            self.logger.error("Error showing audio device error dialog: %s", e, exc_info=True)

    def cleanup(self) -> None:
        try:
            self.event_bus.unsubscribe(AudioDeviceErrorEvent, self._handle_audio_device_error)
        except Exception as e:
            self.logger.debug("System controller event unsubscribe: %s", e)
        self.main_window = None
        super().cleanup()
