"""Qt-based settings view.

Displays application settings with management capabilities.
Uses new component subclasses from components module.
"""

import logging
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.ui.components.buttons import DangerButton
from vocalance.app.ui.components.checkboxes import Checkbox
from vocalance.app.ui.components.complex_components import FormGroup
from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BodyLabel, BoxTitleLabel, SubtitleLabel
from vocalance.app.ui.components.layouts import Box, ScrollableContainer
from vocalance.app.ui.qt_theme import theme


class QtSettingsView(QWidget):
    """Qt-based settings view.

    Features:
    - Display all settings
    - Update individual settings
    - Reset to defaults
    - Real-time updates from controller
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize settings view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
        self.settings: Dict[str, Any] = {}
        self.setting_widgets = {}

        self._setup_ui()
        self.logger.debug("QtSettingsView initialized")

    def set_controller(self, controller) -> None:
        """Set the controller and connect signals."""
        self.controller = controller

        # Connect controller signals
        self.controller.settings_loaded.connect(self._on_settings_loaded)
        self.controller.setting_changed.connect(self._on_setting_changed)
        self.controller.all_settings_changed.connect(self._on_all_settings_changed)
        self.controller.settings_reset.connect(self._on_settings_reset)
        self.controller.operation_error.connect(self._on_error)

        # Load initial settings
        self.logger.info("Loading settings from controller")
        self.controller.load_settings()

    def _setup_ui(self) -> None:
        """Build two-box layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large
        )
        main_layout.setSpacing(theme.config.spacing.large)

        # Create two-box layout
        boxes_layout = QHBoxLayout()
        boxes_layout.setSpacing(theme.config.spacing.large)

        # LEFT BOX - Actions
        self.left_box = Box(layout="vertical")

        # Title
        self.left_box.add(BoxTitleLabel(text="Settings Actions"))

        # Reset to defaults button
        self.reset_btn = DangerButton(text="Reset to Defaults", command=self._on_reset_clicked)
        self.left_box.add(self.reset_btn)

        # Info label
        self.info_label = BodyLabel("Settings loaded", color=theme.config.text.medium)
        self.left_box.add(self.info_label)

        self.left_box.add_stretch()

        # RIGHT BOX - Settings list
        self.right_box = Box(layout="vertical")

        # Title
        self.right_box.add(BoxTitleLabel(text="Settings"))

        # Scrollable settings area
        self.scroll_container = ScrollableContainer()
        self.right_box.add(self.scroll_container, stretch=1)

        # Add boxes to main layout
        boxes_layout.addWidget(self.left_box, 0)
        boxes_layout.addWidget(self.right_box, 1)

        main_layout.addLayout(boxes_layout)

    def _on_settings_loaded(self, settings: Dict[str, Any]) -> None:
        """Handle settings loaded from controller."""
        try:
            self.settings = settings
            self._refresh_settings_display()
            self.logger.info(f"Settings loaded: {len(settings)} items")
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}", exc_info=True)
            self._show_error(f"Error loading settings: {e}")

    def _on_setting_changed(self, key: str, value: Any) -> None:
        """Handle individual setting changed event."""
        self.settings[key] = value
        self.logger.debug(f"Setting changed: {key} = {value}")

    def _on_all_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Handle all settings changed event."""
        self.settings = settings
        self._refresh_settings_display()
        self.logger.info("All settings updated")

    def _on_settings_reset(self) -> None:
        """Handle settings reset event."""
        if self.controller:
            self.controller.load_settings()
        self.logger.info("Settings reset to defaults")
        QMessageBox.information(self, "Success", "Settings reset to defaults!")

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._show_error(error_msg)

    def _refresh_settings_display(self) -> None:
        """Refresh the settings display with organized sections."""
        # Clear existing widgets
        while self.scroll_container.content_layout.count():
            item = self.scroll_container.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.setting_widgets.clear()

        # Define which settings to display
        visible_settings = {
            "LLM Model Settings": [
                ("llm", "context_length", "Max Context Tokens"),
                ("llm", "max_tokens", "Max Output Tokens"),
            ],
            "Grid Settings": [
                ("grid", "default_rect_count", "Default Cell Count"),
            ],
            "Markov Chain Settings": [
                ("markov_predictor", "enabled", "Enable"),
                ("markov_predictor", "confidence_threshold", "Prediction Confidence"),
            ],
            "Sound Recognizer Settings": [
                ("sound_recognizer", "confidence_threshold", "Confidence Threshold"),
                ("sound_recognizer", "vote_threshold", "Vote Threshold"),
            ],
            "Voice Settings": [
                ("vad", "dictation_silent_chunks_for_end", "Max Silent Dictation Chunks"),
                ("vad", "command_silent_chunks_for_end", "Max Silent Command Chunks"),
            ],
        }

        total_count = 0

        # Display each section with filtered settings
        for section_name, field_specs in visible_settings.items():
            # Section title
            self.scroll_container.add(SubtitleLabel(section_name, color=theme.config.text.light))

            # Section items
            for category, key, label_text in field_specs:
                # Get value from settings
                value = None
                if category in self.settings and isinstance(self.settings[category], dict):
                    value = self.settings[category].get(key)

                if value is None:
                    continue

                total_count += 1
                setting_key = f"{category}.{key}"

                # Create widgets based on type
                if isinstance(value, bool):
                    checkbox = Checkbox(
                        text=label_text,
                        checked=value,
                        command=lambda state, k=setting_key: self._on_setting_value_changed(k, state == 2),
                    )
                    self.scroll_container.add(checkbox)
                    self.setting_widgets[setting_key] = checkbox

                elif isinstance(value, (int, float, str)):
                    inp = TextInput(str(value))
                    inp.editingFinished.connect(lambda k=setting_key, w=inp: self._on_setting_value_changed(k, w.text()))

                    group = FormGroup(label_text, inp)
                    self.scroll_container.add(group)
                    self.setting_widgets[setting_key] = inp

        # Add stretch at end
        self.scroll_container.add_stretch()

        # Update info
        self.info_label.setText(f"{total_count} settings")

    def _on_setting_value_changed(self, key: str, value: Any) -> None:
        """Handle setting value changed."""
        try:
            if self.controller:
                self.controller.update_setting(key, value)
                self.logger.debug(f"Setting updated: {key} = {value}")
        except Exception as e:
            self.logger.error(f"Error updating setting: {e}", exc_info=True)
            self._show_error(f"Error updating setting: {e}")

    def _on_reset_clicked(self) -> None:
        """Handle reset button clicked."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.controller:
                self.controller.reset_to_defaults()

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
