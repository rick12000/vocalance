import asyncio
import logging
from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox, QWidget

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDeviceErrorEvent, StorageCleanupRequestEvent, StorageCorruptionWarningEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtSystemController(QtBaseController[QWidget]):
    """Surfaces global system failures as modals on ``main_window``."""

    audio_device_error = Signal(str)
    storage_corruption_warning = Signal(list)

    def __init__(self, event_bus: EventBus, main_window: QWidget) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger(self.__class__.__name__))
        self.main_window = main_window
        self.subscribe(AudioDeviceErrorEvent, self._on_audio_device_error)
        self.subscribe(StorageCorruptionWarningEvent, self._on_storage_corruption)
        self.audio_device_error.connect(self._show_microphone_error_dialog)
        self.storage_corruption_warning.connect(self._show_storage_corruption_dialog)

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

    def _on_storage_corruption(self, event: StorageCorruptionWarningEvent) -> None:
        self.logger.error("Storage corruption detected in files: %s", event.corrupt_files)
        self.storage_corruption_warning.emit(event.corrupt_files)

    def _show_storage_corruption_dialog(self, corrupt_files: List[str]) -> None:
        file_list = "\n".join(f"  • {f}" for f in corrupt_files)
        msg = QMessageBox(self.main_window)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Corrupted data files detected")
        msg.setText(
            "One or more Vocalance data files could not be read and have been reset to defaults for this session.\n\n"
            f"Affected files:\n{file_list}\n\n"
            "Delete the corrupt files now so Vocalance starts cleanly next time?\n"
            "(If you choose No, this warning will appear on every launch until the files are removed.)"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)
        choice = msg.exec()

        if choice == QMessageBox.StandardButton.Yes:
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: asyncio.ensure_future(self.event_bus.publish(StorageCleanupRequestEvent(files_to_delete=corrupt_files)))
            )
            self.logger.info("User confirmed deletion of corrupt storage files: %s", corrupt_files)

    def shutdown(self) -> None:
        self.main_window = None
        super().shutdown()
