import logging
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox

from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDeviceErrorEvent
from vocalance.app.ui.controls.qt_base_controller import QtBaseController


class QtSystemController(QtBaseController):
    """Controller for system-level events and error handling.

    Handles:
    - Audio device errors
    - System notifications
    - Critical error dialogs
    """

    # Signals for system events
    audio_device_error = Signal(str)

    def __init__(self, event_bus: EventBus, main_window: Any) -> None:
        """Initialize system controller.

        Args:
            event_bus: Event bus for pub/sub.
            main_window: Reference to main window used as parent for dialogs.
        """
        super().__init__(event_bus=event_bus, logger=logging.getLogger(self.__class__.__name__))

        self.main_window = main_window
        self._setup_subscriptions()
        self._connect_signals()

        self.logger.debug("QtSystemController initialized")

    def _setup_subscriptions(self) -> None:
        """Subscribe to system-level events."""
        try:
            self.event_bus.subscribe(event_type=AudioDeviceErrorEvent, handler=self._handle_audio_device_error)
            self.logger.debug("System controller event subscriptions configured")
        except Exception as e:
            self.logger.error(f"Error setting up event subscriptions: {e}", exc_info=True)

    def _connect_signals(self) -> None:
        """Connect Qt signals to slots."""
        self.audio_device_error.connect(self._show_audio_device_error_dialog)

    def _handle_audio_device_error(self, device_error: AudioDeviceErrorEvent) -> None:
        """Handle audio device error event from service layer.

        Args:
            device_error: Audio device error event.
        """
        try:
            self.logger.warning(f"Audio device error: {device_error.error_message}")

            # Emit Qt signal (thread-safe - will invoke on main thread)
            self.audio_device_error.emit(device_error.error_message)

        except Exception as e:
            self.logger.error(f"Error handling audio device error: {e}", exc_info=True)

    def _show_audio_device_error_dialog(self, error_message: str) -> None:
        """Show audio device error dialog to user (runs on Qt main thread).

        Args:
            error_message: Full message from AudioDeviceErrorEvent.
        """
        try:
            msg = QMessageBox(self.main_window)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Microphone unavailable")
            msg.setText(error_message)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

            self.logger.info("Audio device error dialog shown to user")

        except Exception as e:
            self.logger.error(f"Error showing audio device error dialog: {e}", exc_info=True)

    def cleanup(self) -> None:
        """Clean up controller resources."""
        self.main_window = None
        super().cleanup()
