"""Qt-based settings view - FULLY INTEGRATED.

Displays application settings with management capabilities.
"""

import logging
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QScrollArea, QSpinBox, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme_manager
from vocalance.app.ui.views.components.qt_themed_components import DangerButton, ThemedFrame, TitleLabel


class QtSettingsView(QWidget):
    """Qt-based settings view - FULLY INTEGRATED.

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
            theme_manager.two_box_layout.outer_padding_left,
            theme_manager.two_box_layout.outer_padding_top,
            theme_manager.two_box_layout.outer_padding_right,
            theme_manager.two_box_layout.outer_padding_bottom,
        )
        main_layout.setSpacing(theme_manager.two_box_layout.base_spacing)

        # Create two-box layout
        boxes_layout = QHBoxLayout()
        boxes_layout.setSpacing(theme_manager.two_box_layout.base_spacing)

        # LEFT BOX - Actions
        left_box = ThemedFrame(frame_type="two_box")
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(
            theme_manager.two_box_layout.inner_content_padx,
            20,
            theme_manager.two_box_layout.inner_content_padx,
            20,
        )
        left_layout.setSpacing(15)

        # Title
        left_layout.addWidget(TitleLabel(text="Settings Actions"))

        # Reset to defaults button
        self.reset_btn = DangerButton(text="Reset to Defaults")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        left_layout.addWidget(self.reset_btn)

        # Info label
        self.info_label = QLabel("Settings loaded")
        left_layout.addWidget(self.info_label)

        left_layout.addStretch()

        # RIGHT BOX - Settings list
        right_box = ThemedFrame(frame_type="two_box")
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(
            theme_manager.two_box_layout.inner_content_padx,
            20,
            theme_manager.two_box_layout.inner_content_padx,
            20,
        )
        right_layout.setSpacing(10)

        # Title
        right_layout.addWidget(TitleLabel(text="Settings"))

        # Scrollable settings area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        self.settings_widget = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_widget)
        self.settings_layout.setSpacing(10)

        scroll_area.setWidget(self.settings_widget)
        right_layout.addWidget(scroll_area)

        # Add boxes to main layout
        boxes_layout.addWidget(left_box, 0)
        boxes_layout.addWidget(right_box, 1)

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
        try:
            self.settings[key] = value
            self.logger.debug(f"Setting changed: {key} = {value}")
        except Exception as e:
            self.logger.error(f"Error handling setting changed: {e}", exc_info=True)

    def _on_all_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Handle all settings changed event."""
        try:
            self.settings = settings
            self._refresh_settings_display()
            self.logger.info("All settings updated")
        except Exception as e:
            self.logger.error(f"Error handling all settings changed: {e}", exc_info=True)

    def _on_settings_reset(self) -> None:
        """Handle settings reset event."""
        try:
            # Reload settings to get defaults
            if self.controller:
                self.controller.load_settings()
            self.logger.info("Settings reset to defaults")
            QMessageBox.information(self, "Success", "Settings reset to defaults!")
        except Exception as e:
            self.logger.error(f"Error handling settings reset: {e}", exc_info=True)

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._show_error(error_msg)

    def _refresh_settings_display(self) -> None:
        """Refresh the settings display with organized sections - matching legacy UI."""
        # Clear existing widgets
        while self.settings_layout.count():
            item = self.settings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.setting_widgets.clear()

        # Define which settings to display (matching legacy implementation)
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
            section_label = QLabel(section_name)
            section_font = theme_manager.get_font(size=theme_manager.font_sizes.large, bold=True)
            section_label.setFont(section_font)
            section_label.setStyleSheet(f"color: {theme_manager.text_colors.light}; margin-top: 15px; margin-bottom: 10px;")
            self.settings_layout.addWidget(section_label)

            # Section items
            for category, key, label_text in field_specs:
                # Get value from settings
                value = None
                if category in self.settings and isinstance(self.settings[category], dict):
                    value = self.settings[category].get(key)

                if value is None:
                    continue  # Skip if setting not found

                total_count += 1

                # Create row for each setting
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(10, 5, 10, 5)

                # Label
                label = QLabel(label_text)
                label.setMinimumWidth(200)
                row_layout.addWidget(label, 0)

                # Input widget based on type
                setting_key = f"{category}.{key}"

                if isinstance(value, bool):
                    widget = QCheckBox()
                    widget.setChecked(value)
                    widget.stateChanged.connect(
                        lambda state, k=setting_key, w=widget: self._on_setting_value_changed(k, w.isChecked())
                    )
                elif isinstance(value, (int, float)):
                    widget = QSpinBox()
                    widget.setValue(int(value))
                    widget.setMinimum(0)
                    widget.setMaximum(10000)
                    widget.valueChanged.connect(lambda v, k=setting_key, w=widget: self._on_setting_value_changed(k, w.value()))
                else:
                    widget = QLineEdit()
                    widget.setText(str(value))
                    widget.editingFinished.connect(lambda k=setting_key, w=widget: self._on_setting_value_changed(k, w.text()))

                row_layout.addWidget(widget, 1)
                self.settings_layout.addWidget(row_widget)

                self.setting_widgets[setting_key] = widget

        # Add stretch at end
        self.settings_layout.addStretch()

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
        reply = QMessageBox.question(self, "Reset Settings", "Reset all settings to defaults?", QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.controller:
                self.controller.reset_to_defaults()

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
