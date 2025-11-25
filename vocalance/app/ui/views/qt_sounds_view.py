"""Qt-based sounds training view.

Displays trained sounds and allows training new sounds with command mapping.
Uses new component subclasses from components module.
"""

from typing import List, Optional

from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QVBoxLayout, QWidget

from vocalance.app.ui.components.buttons import DangerButton, PrimaryButton
from vocalance.app.ui.components.dialogs import BaseDialog
from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BodyLabel, SmallLabel
from vocalance.app.ui.components.layouts import BaseContainer, ScrollableContainer, TransparentWidget, TwoColumnLayout
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.views.qt_base_view import QtBaseView


class SoundMappingDialog(BaseDialog):
    """Dialog for mapping sounds to commands, marks, or grid triggers."""

    def __init__(self, sound_name: str, controller, parent: Optional[QWidget] = None):
        super().__init__(
            parent=parent,
            title=f"Map Sound: {sound_name}",
            min_width=theme.config.components.sound_mapping_dialog_width,
        )

        self.sound_name = sound_name
        self.controller = controller
        self.selected_command = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        # Current mapping section
        current_frame = BaseContainer()
        current_layout = current_frame.layout()
        current_layout.setContentsMargins(
            theme.config.spacing.medium, theme.config.spacing.medium, theme.config.spacing.medium, theme.config.spacing.medium
        )
        current_layout.setSpacing(theme.config.spacing.small)

        current_title = SmallLabel("Current Mapping:", color=theme.config.text.light)
        current_title.setFont(theme.get_font(size=theme.config.fonts.medium, weight="semibold"))
        current_layout.addWidget(current_title)

        # Get current mapping
        try:
            mapped_command = self.controller.get_sound_command_mapping(self.sound_name)
            mapping_text = mapped_command if mapped_command else "Unmapped"
        except Exception:
            mapping_text = "Unmapped"

        current_label = SmallLabel(mapping_text, color=theme.config.text.medium)
        current_layout.addWidget(current_label)

        self._main_layout.addWidget(current_frame)

        # Mapping selection section
        mapping_frame = BaseContainer()
        mapping_layout = mapping_frame.layout()
        mapping_layout.setContentsMargins(
            theme.config.spacing.medium, theme.config.spacing.medium, theme.config.spacing.medium, theme.config.spacing.medium
        )
        mapping_layout.setSpacing(theme.config.spacing.medium)

        # Command Type dropdown
        type_label = SmallLabel("Command Type:", color=theme.config.text.light)
        type_label.setFont(theme.get_font(size=theme.config.fonts.medium, weight="semibold"))
        mapping_layout.addWidget(type_label)

        self.type_combo = QComboBox()
        command_types = self.controller.get_mapping_command_types()
        self.type_combo.addItems(command_types)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        mapping_layout.addWidget(self.type_combo)

        # Command Value dropdown
        value_label = SmallLabel("Command Value:", color=theme.config.text.light)
        value_label.setFont(theme.get_font(size=theme.config.fonts.medium, weight="semibold"))
        mapping_layout.addWidget(value_label)

        self.value_combo = QComboBox()
        mapping_layout.addWidget(self.value_combo)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(theme.config.spacing.medium)

        confirm_btn = PrimaryButton(text="Confirm")
        confirm_btn.clicked.connect(self._on_confirm)
        button_layout.addWidget(confirm_btn)

        cancel_btn = DangerButton(text="Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        mapping_layout.addLayout(button_layout)

        self._main_layout.addWidget(mapping_frame)

        # Initialize value dropdown
        self._on_type_changed(self.type_combo.currentText())

    def _on_type_changed(self, selected_type: str) -> None:
        """Handle command type dropdown change."""
        self.value_combo.clear()

        if selected_type == "Commands":
            values = self.controller.get_available_exact_match_commands()
        elif selected_type == "Marks":
            values = self.controller.get_available_mark_names()
        elif selected_type == "Grid":
            values = self.controller.get_grid_trigger_words()
        else:
            values = []

        if values:
            self.value_combo.addItems(values)
        else:
            self.value_combo.addItem("No options available")

    def _on_confirm(self) -> None:
        """Handle confirm button click."""
        command_value = self.value_combo.currentText().strip()

        if command_value and command_value != "No options available":
            self.selected_command = command_value
            self.accept()


class QtSoundsView(QtBaseView):
    """Qt-based sounds training view.

    Features:
    - Training form with name input and samples spinner
    - Progress bar with status updates
    - Trained sounds list
    - Map button for each sound (opens mapping dialog)
    - Delete button for each sound
    - Delete all sounds button
    - Real-time updates from controller
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize sounds view."""
        super().__init__(parent)

        self.sounds_list: List[str] = []
        self.current_training_sound = None

        self._setup_ui()
        self.logger.debug("QtSoundsView initialized")

    def set_controller(self, controller) -> None:
        """Set the controller and connect signals."""
        self.controller = controller

        # Connect controller signals
        self.controller.sounds_loaded.connect(self._on_sounds_loaded)
        self.controller.training_started.connect(self._on_training_started)
        self.controller.training_progress.connect(self._on_training_progress)
        self.controller.training_completed.connect(self._on_training_completed)
        self.controller.training_error.connect(self._on_training_error)
        self.controller.sound_deleted.connect(self._on_sound_deleted)
        self.controller.operation_error.connect(self._on_error)

        # Load initial sounds
        self.logger.info("Loading sounds from controller")
        self.controller.refresh_sound_list()
        self.controller.on_view_ready()

    def _setup_ui(self) -> None:
        """Build UI with TwoColumnLayout."""
        self.layout = TwoColumnLayout("Train Sound", "Trained Sounds", self)
        self.main_layout.addWidget(self.layout, stretch=1)

        self._setup_training_form()
        self._setup_sounds_list_panel()

    def _setup_training_form(self) -> None:
        """Setup training form in left content area."""
        content = self.layout.left_content

        # Sound name input
        sound_name_label = BodyLabel("Sound Name:")
        content.add(sound_name_label)
        self.sound_name_input = TextInput(placeholder="e.g., doorbell")
        self.sound_name_input.setMaxLength(50)
        content.add(self.sound_name_input)

        # Number of samples
        samples_label = BodyLabel("Samples:")
        content.add(samples_label)
        self.samples_spinbox = TextInput(placeholder="e.g., 5")
        default_samples = self.controller.get_default_training_samples() if self.controller else 5
        self.samples_spinbox.setText(str(default_samples))
        content.add(self.samples_spinbox)

        # Start training button
        self.start_training_btn = PrimaryButton(text="Record")
        self.start_training_btn.clicked.connect(self._on_start_training_clicked)
        content.add(self.start_training_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        content.add(self.progress_bar)

        # Status label
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        content.add(self.status_label)

        content.add_stretch()

    def _setup_sounds_list_panel(self) -> None:
        """Setup sounds list panel in right content area."""
        content = self.layout.right_content

        # Sounds list widget
        self.sounds_list_widget = TransparentWidget()
        self.sounds_list_layout = QVBoxLayout(self.sounds_list_widget)
        self.sounds_list_layout.setSpacing(theme.config.container.list_item_spacing)
        self.sounds_list_layout.setContentsMargins(0, 0, theme.config.spacing.small, 0)

        # Scroll area for sounds
        scroll_area = ScrollableContainer()
        scroll_area.content_layout.addWidget(self.sounds_list_widget)
        content.add(scroll_area, stretch=1)

        # Delete all button
        self.delete_all_btn = DangerButton(text="Reset")
        self.delete_all_btn.clicked.connect(self._on_delete_all_clicked)
        content.add(self.delete_all_btn)

    def _on_sounds_loaded(self, sounds: List[str]) -> None:
        """Handle sounds loaded from controller."""
        try:
            self.sounds_list = sounds
            self._refresh_sounds_list()
            self.logger.info(f"Sounds loaded: {len(sounds)} total")
        except Exception as e:
            self.logger.error(f"Error loading sounds: {e}", exc_info=True)
            self._show_error(f"Error loading sounds: {e}")

    def _on_training_started(self, sound_name: str, total_samples: int) -> None:
        """Handle training started event."""
        self.current_training_sound = sound_name
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(total_samples)
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"Recording sample 1 of {total_samples}")
        self.status_label.setVisible(True)
        self.start_training_btn.setEnabled(False)
        self.sound_name_input.setEnabled(False)
        self.samples_spinbox.setEnabled(False)
        self.logger.info(f"Training started: {sound_name}")

    def _on_training_progress(self, sound_name: str, current: int, total: int) -> None:
        """Handle training progress event."""
        if sound_name == self.current_training_sound:
            self.progress_bar.setValue(current)
            if current < total:
                self.status_label.setText(f"Recording sample {current + 1} of {total}")
            else:
                self.progress_bar.setVisible(False)
                self.status_label.setText("Training...")

    def _on_training_completed(self, sound_name: str) -> None:
        """Handle training completed event."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Training complete for '{sound_name}'!")
        self.status_label.setVisible(True)
        self.start_training_btn.setEnabled(True)
        self.sound_name_input.setEnabled(True)
        self.samples_spinbox.setEnabled(True)
        self.sound_name_input.clear()
        self.current_training_sound = None
        self.logger.info(f"Training completed: {sound_name}")

    def _on_training_error(self, sound_name: str, error_msg: str) -> None:
        """Handle training error event."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Training failed: {error_msg}")
        self.status_label.setVisible(True)
        self.start_training_btn.setEnabled(True)
        self.sound_name_input.setEnabled(True)
        self.samples_spinbox.setEnabled(True)
        self.current_training_sound = None
        self.logger.error(f"Training error for {sound_name}: {error_msg}")

    def _on_sound_deleted(self, sound_name: str) -> None:
        """Handle sound deleted event."""
        if self.controller:
            self.controller.refresh_sound_list()
        self.logger.info(f"Sound deleted: {sound_name}")

    def _on_error(self, error_message: str) -> None:
        """Handle error from controller."""
        self.logger.error(f"Controller error: {error_message}")
        self._show_error(error_message)

    def _refresh_sounds_list(self) -> None:
        """Refresh the display of trained sounds."""
        # Clear all existing widgets
        while self.sounds_list_layout.count() > 0:
            item = self.sounds_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.sounds_list:
            empty_label = BodyLabel(
                "No available sounds.\nUse the left panel to record a sound.",
                align="center",
                color=theme.config.text.medium,
            )
            self.sounds_list_layout.addWidget(empty_label)
        else:
            for sound_name in sorted(self.sounds_list):
                item_widget = TransparentWidget()
                item_widget.setProperty("itemType", "list_item")
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(
                    0,
                    theme.config.container.list_item_padding_vertical,
                    0,
                    theme.config.container.list_item_padding_vertical,
                )
                item_layout.setSpacing(theme.config.spacing.small)

                # Sound name label
                name_label = SmallLabel(sound_name, color=theme.config.text.medium)
                item_layout.addWidget(name_label, 1)

                # Map button
                map_btn = PrimaryButton(text="Map")
                map_btn.setFixedWidth(theme.config.components.button_action_width)
                map_btn.clicked.connect(lambda checked, s=sound_name: self._on_map_sound(s))
                item_layout.addWidget(map_btn)

                # Delete button
                delete_btn = DangerButton(text="Delete")
                delete_btn.setFixedWidth(theme.config.components.button_action_width)
                delete_btn.clicked.connect(lambda checked, s=sound_name: self._on_delete_sound(s))
                item_layout.addWidget(delete_btn)

                self.sounds_list_layout.addWidget(item_widget)

        self.sounds_list_layout.addStretch()

    def _on_map_sound(self, sound_name: str) -> None:
        """Handle map button clicked - show mapping dialog."""
        if not self.controller:
            self._show_error("Controller not initialized.")
            return

        dialog = SoundMappingDialog(sound_name, self.controller, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.selected_command:
                self.controller.map_sound_to_command(sound_name, dialog.selected_command)
                self.logger.info(f"Mapped sound '{sound_name}' to '{dialog.selected_command}'")

    def _on_delete_sound(self, sound_name: str) -> None:
        """Handle delete button clicked."""
        reply = QMessageBox.question(
            self,
            "Delete Sound",
            f"Delete trained sound '{sound_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.controller:
                self.controller.delete_individual_sound(sound_name)

    def _on_start_training_clicked(self) -> None:
        """Handle start training button click."""
        sound_name = self.sound_name_input.text().strip()
        if not sound_name:
            self._show_error("Please enter a sound name.")
            return

        try:
            num_samples = int(self.samples_spinbox.text().strip())
            if num_samples < 1 or num_samples > 100:
                self._show_error("Please enter a number between 1 and 100 for samples.")
                return
        except ValueError:
            self._show_error("Please enter a valid number for samples.")
            return

        if self.controller:
            self.controller.start_training(sound_name, num_samples)

    def _on_delete_all_clicked(self) -> None:
        """Handle delete all button click."""
        if not self.sounds_list:
            return

        reply = QMessageBox.question(
            self,
            "Reset",
            "Delete all trained sounds? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.controller:
                self.controller.delete_all_sounds()

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        from vocalance.app.ui.components.dialogs import showerror

        showerror(message, parent=self)
