"""Qt-based settings view.

Displays application settings with management capabilities.
Uses new component subclasses from components module.
"""

import logging
from functools import partial
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.ui.components.buttons import DangerButton, PrimaryButton
from vocalance.app.ui.components.checkboxes import Checkbox
from vocalance.app.ui.components.complex_components import FormGroup
from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BoxTitleLabel, SectionTitle
from vocalance.app.ui.components.layouts import Box, ScrollableContainer
from vocalance.app.ui.qt_theme import theme


class QtSettingsView(QWidget):
    """Qt-based settings view.

    Features:
    - Display all settings in sections
    - Save settings per section
    - Reset to defaults per section
    - Real-time updates from controller
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize settings view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
        self.settings: Dict[str, Any] = {}
        self.setting_widgets = {}
        self.section_widgets = {}  # Track widgets per section

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
        """Build single-box layout with per-section controls."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large
        )
        main_layout.setSpacing(theme.config.spacing.large)

        # Single box for settings
        self.settings_box = Box(layout="vertical")

        # Title
        self.settings_box.add(BoxTitleLabel(text="Settings"))

        # Scrollable settings area
        self.scroll_container = ScrollableContainer()
        self.settings_box.add(self.scroll_container, stretch=1)

        main_layout.addWidget(self.settings_box)

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
        """Handle settings reset event from controller."""
        if self.controller:
            self.controller.load_settings()
        self.logger.info("Settings reset event received")
        # Note: specific section messages are shown by the reset button handlers

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._show_error(error_msg)

    def _refresh_settings_display(self) -> None:
        """Refresh the settings display with organized sections and per-section controls."""
        # Clear existing widgets
        while self.scroll_container.content_layout.count():
            item = self.scroll_container.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.setting_widgets.clear()
        self.section_widgets.clear()

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

        # Display each section with filtered settings
        for section_name, field_specs in visible_settings.items():
            section_widgets_dict = {}

            # Section title
            self.scroll_container.add(SectionTitle(section_name))

            # Section items
            for category, key, label_text in field_specs:
                # Get value from settings
                value = None
                if category in self.settings and isinstance(self.settings[category], dict):
                    value = self.settings[category].get(key)

                if value is None:
                    continue

                setting_key = f"{category}.{key}"

                # Create widgets based on type
                # CRITICAL: Check bool BEFORE int/float because isinstance(True, int) == True in Python!
                if isinstance(value, bool):
                    checkbox = Checkbox(
                        text=label_text,
                        checked=value,
                        command=lambda state, k=setting_key: None,  # Don't save on change
                    )
                    self.scroll_container.add(checkbox)
                    self.setting_widgets[setting_key] = checkbox
                    section_widgets_dict[setting_key] = checkbox

                elif isinstance(value, (int, float, str)):
                    inp = TextInput(str(value))
                    # Don't connect to auto-save on editing finished

                    group = FormGroup(label_text, inp)
                    self.scroll_container.add(group)
                    self.setting_widgets[setting_key] = inp
                    section_widgets_dict[setting_key] = inp

            # Store section widgets mapping
            self.section_widgets[section_name] = {"fields": field_specs, "widgets": section_widgets_dict}

            # Add button row for this section
            button_layout = QHBoxLayout()
            button_layout.setSpacing(theme.config.spacing.medium)

            # Save button - use partial to avoid lambda closure issues
            save_btn = PrimaryButton(text="Save", command=partial(self._on_save_section_clicked, section_name))
            button_layout.addWidget(save_btn)

            # Reset to defaults button - use partial to avoid lambda closure issues
            reset_btn = DangerButton(text="Reset to Defaults", command=partial(self._on_reset_section_clicked, section_name))
            button_layout.addWidget(reset_btn)

            button_layout.addStretch()

            # Add button row to scroll container
            self.scroll_container.content_layout.addLayout(button_layout)

            # Add spacing between sections
            self.scroll_container.content_layout.addSpacing(theme.config.spacing.large)

        # Add stretch at end
        self.scroll_container.add_stretch()

    def _on_save_section_clicked(self, section_name: str) -> None:
        """Handle save button clicked for a specific section."""
        try:
            if not self.controller:
                return

            # Get widgets for this section
            section_info = self.section_widgets.get(section_name)
            if not section_info:
                self.logger.warning(f"Section not found: {section_name}")
                return

            # Define expected types for each setting
            setting_types = self._get_setting_types()

            # Collect settings to save from this section
            settings_to_save = {}
            for setting_key, widget in section_info["widgets"].items():
                # Get current value from widget
                if isinstance(widget, Checkbox):
                    value = widget.isChecked()
                elif isinstance(widget, TextInput):
                    text_value = widget.text().strip()

                    # Get the expected type for this setting
                    expected_type = setting_types.get(setting_key)
                    if expected_type is None:
                        self._show_error(f"Unknown setting type for {setting_key}")
                        return

                    # Convert text input to the expected type
                    try:
                        if expected_type == int:
                            value = int(text_value)
                        elif expected_type == float:
                            value = float(text_value)
                        elif expected_type == bool:
                            # Handle bool strings
                            if text_value.lower() in ("true", "1", "yes", "on"):
                                value = True
                            elif text_value.lower() in ("false", "0", "no", "off"):
                                value = False
                            else:
                                self._show_error(f"Invalid boolean value for {setting_key}. Use: true/false, yes/no, on/off, 1/0")
                                return
                        else:  # str
                            value = text_value
                    except ValueError:
                        self._show_error(f"Invalid value for {setting_key}. Expected {expected_type.__name__}, got: {text_value}")
                        return
                else:
                    continue

                settings_to_save[setting_key] = value

            # Save all settings in this section
            if settings_to_save:
                self.controller.update_settings(settings_to_save)
                self.logger.info(f"Saved {len(settings_to_save)} settings from section: {section_name}")
                QMessageBox.information(self, "Success", f"{section_name} saved successfully!")

        except Exception as e:
            self.logger.error(f"Error saving section {section_name}: {e}", exc_info=True)
            self._show_error(f"Error saving settings: {e}")

    def _on_reset_section_clicked(self, section_name: str) -> None:
        """Handle reset button clicked for a specific section."""
        reply = QMessageBox.question(
            self,
            "Reset Section",
            f"Reset {section_name} to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if not self.controller:
                    return

                # Get settings keys for this section
                section_info = self.section_widgets.get(section_name)
                if not section_info:
                    return

                setting_keys = list(section_info["widgets"].keys())

                # Reset settings for this section
                self.controller.reset_section_settings(setting_keys)
                self.logger.info(f"Reset section to defaults: {section_name}")

                # Show success message
                QMessageBox.information(self, "Success", f"{section_name} reset to defaults!")

            except Exception as e:
                self.logger.error(f"Error resetting section: {e}", exc_info=True)
                self._show_error(f"Error resetting section: {e}")

    def _get_setting_types(self) -> Dict[str, type]:
        """Define the expected type for each setting."""
        return {
            # LLM Model Settings
            "llm.context_length": int,
            "llm.max_tokens": int,
            # Grid Settings
            "grid.default_rect_count": int,
            # Markov Chain Settings
            "markov_predictor.enabled": bool,
            "markov_predictor.confidence_threshold": float,
            # Sound Recognizer Settings
            "sound_recognizer.confidence_threshold": float,
            "sound_recognizer.vote_threshold": float,
            # Voice Settings
            "vad.dictation_silent_chunks_for_end": int,
            "vad.command_silent_chunks_for_end": int,
        }

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
