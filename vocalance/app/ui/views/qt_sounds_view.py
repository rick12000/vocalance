"""Qt-based sounds training view - FULLY INTEGRATED WITH MAPPING DIALOG.

Displays trained sounds and allows training new sounds with command mapping.
"""

import logging
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vocalance.app.ui.qt_theme import theme_manager
from vocalance.app.ui.views.components.qt_themed_components import DangerButton, PrimaryButton, ThemedFrame, TitleLabel


class SoundMappingDialog(QDialog):
    """Dialog for mapping sounds to commands, marks, or grid triggers."""

    def __init__(self, sound_name: str, controller, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.sound_name = sound_name
        self.controller = controller
        self.selected_command = None

        self.setWindowTitle(f"Map Sound: {sound_name}")
        self.setModal(True)
        self.setMinimumWidth(500)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Current mapping frame
        current_frame = ThemedFrame()
        current_layout = QVBoxLayout(current_frame)
        current_layout.setContentsMargins(15, 15, 15, 15)
        current_layout.setSpacing(5)

        current_title = QLabel("Current Mapping:")
        current_title_font = theme_manager.get_font(size=theme_manager.font_sizes.medium, weight="bold")
        current_title.setFont(current_title_font)
        current_title.setStyleSheet(f"color: {theme_manager.text_colors.light};")
        current_layout.addWidget(current_title)

        # Get current mapping
        try:
            mapped_command = self.controller.get_sound_command_mapping(self.sound_name)
            mapping_text = mapped_command if mapped_command else "Unmapped"
        except Exception:
            mapping_text = "Unmapped"

        current_label = QLabel(mapping_text)
        current_font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        current_label.setFont(current_font)
        current_label.setStyleSheet(f"color: {theme_manager.text_colors.medium};")
        current_layout.addWidget(current_label)

        layout.addWidget(current_frame)

        # Main mapping frame
        mapping_frame = ThemedFrame()
        mapping_layout = QVBoxLayout(mapping_frame)
        mapping_layout.setContentsMargins(15, 15, 15, 15)
        mapping_layout.setSpacing(10)

        # Command Type dropdown
        type_label = QLabel("Command Type:")
        type_label_font = theme_manager.get_font(size=theme_manager.font_sizes.medium, weight="bold")
        type_label.setFont(type_label_font)
        type_label.setStyleSheet(f"color: {theme_manager.text_colors.light};")
        mapping_layout.addWidget(type_label)

        self.type_combo = QComboBox()
        command_types = self.controller.get_mapping_command_types()
        self.type_combo.addItems(command_types)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        mapping_layout.addWidget(self.type_combo)

        # Command Value dropdown
        value_label = QLabel("Command Value:")
        value_label.setFont(type_label_font)
        value_label.setStyleSheet(f"color: {theme_manager.text_colors.light};")
        mapping_layout.addWidget(value_label)

        self.value_combo = QComboBox()
        mapping_layout.addWidget(self.value_combo)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        confirm_btn = PrimaryButton(text="Confirm")
        confirm_btn.clicked.connect(self._on_confirm)
        button_layout.addWidget(confirm_btn)

        cancel_btn = DangerButton(text="Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        mapping_layout.addLayout(button_layout)

        layout.addWidget(mapping_frame)

        # Initialize value dropdown
        self._on_type_changed(self.type_combo.currentText())

    def _on_type_changed(self, selected_type: str) -> None:
        """Handle command type dropdown change."""
        self.value_combo.clear()

        # Get appropriate values based on selected type
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


class QtSoundsView(QWidget):
    """Qt-based sounds training view - FULLY INTEGRATED WITH MAPPING.

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

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
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

        # LEFT BOX - Training controls
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
        left_layout.addWidget(TitleLabel(text="Train Sound"))

        # Sound name input
        left_layout.addWidget(QLabel("Sound Name:"))
        self.sound_name_input = QLineEdit()
        self.sound_name_input.setPlaceholderText("e.g., doorbell")
        self.sound_name_input.setMaxLength(50)
        left_layout.addWidget(self.sound_name_input)

        # Number of samples
        left_layout.addWidget(QLabel("Samples:"))
        self.samples_spinbox = QSpinBox()
        self.samples_spinbox.setMinimum(1)
        self.samples_spinbox.setMaximum(100)
        self.samples_spinbox.setValue(self.controller.get_default_training_samples() if self.controller else 5)
        left_layout.addWidget(self.samples_spinbox)

        # Start training button
        self.start_training_btn = PrimaryButton(text="Record")
        self.start_training_btn.clicked.connect(self._on_start_training_clicked)
        left_layout.addWidget(self.start_training_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()

        # RIGHT BOX - Sounds list
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
        right_layout.addWidget(TitleLabel(text="Trained Sounds"))

        # Sounds list widget (will show custom items with Map/Delete buttons)
        self.sounds_list_widget = QWidget()
        self.sounds_list_layout = QVBoxLayout(self.sounds_list_widget)
        self.sounds_list_layout.setSpacing(5)
        self.sounds_list_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for sounds
        from PySide6.QtWidgets import QScrollArea

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.sounds_list_widget)
        right_layout.addWidget(scroll_area)

        # Delete all button
        self.delete_all_btn = DangerButton(text="Delete All Sounds")
        self.delete_all_btn.clicked.connect(self._on_delete_all_clicked)
        right_layout.addWidget(self.delete_all_btn)

        # Add boxes to main layout
        boxes_layout.addWidget(left_box, 0)
        boxes_layout.addWidget(right_box, 1)

        main_layout.addLayout(boxes_layout)

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
        try:
            self.current_training_sound = sound_name
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(total_samples)
            self.progress_bar.setVisible(True)
            self.status_label.setText(f"Recording sample 1 of {total_samples}")
            self.start_training_btn.setEnabled(False)
            self.sound_name_input.setEnabled(False)
            self.samples_spinbox.setEnabled(False)
            self.logger.info(f"Training started: {sound_name}")
        except Exception as e:
            self.logger.error(f"Error handling training started: {e}", exc_info=True)

    def _on_training_progress(self, sound_name: str, current: int, total: int) -> None:
        """Handle training progress event."""
        try:
            if sound_name == self.current_training_sound:
                self.progress_bar.setValue(current)
                if current < total:
                    self.status_label.setText(f"Recording sample {current + 1} of {total}")
                else:
                    self.progress_bar.setVisible(False)
                    self.status_label.setText("Training...")
        except Exception as e:
            self.logger.error(f"Error handling training progress: {e}", exc_info=True)

    def _on_training_completed(self, sound_name: str) -> None:
        """Handle training completed event."""
        try:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"Training complete for '{sound_name}'!")
            self.start_training_btn.setEnabled(True)
            self.sound_name_input.setEnabled(True)
            self.samples_spinbox.setEnabled(True)
            self.sound_name_input.clear()
            self.current_training_sound = None

            self.logger.info(f"Training completed: {sound_name}")
        except Exception as e:
            self.logger.error(f"Error handling training completed: {e}", exc_info=True)

    def _on_training_error(self, sound_name: str, error_msg: str) -> None:
        """Handle training error event."""
        try:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"Error: {error_msg}")
            self.start_training_btn.setEnabled(True)
            self.sound_name_input.setEnabled(True)
            self.samples_spinbox.setEnabled(True)
            self.current_training_sound = None

            self.logger.error(f"Training error for {sound_name}: {error_msg}")
            self._show_error(f"Training failed: {error_msg}")
        except Exception as e:
            self.logger.error(f"Error handling training error: {e}", exc_info=True)

    def _on_sound_deleted(self, sound_name: str) -> None:
        """Handle sound deleted event."""
        try:
            if sound_name in self.sounds_list:
                self.sounds_list.remove(sound_name)
            self._refresh_sounds_list()
            self.logger.info(f"Sound deleted: {sound_name}")
        except Exception as e:
            self.logger.error(f"Error handling sound deleted: {e}", exc_info=True)

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._show_error(error_msg)

    def _refresh_sounds_list(self) -> None:
        """Refresh the sounds list display with Map and Delete buttons."""
        # Clear existing items
        while self.sounds_list_layout.count():
            item = self.sounds_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.sounds_list:
            # Show empty message
            empty_label = QLabel("No available sounds.\nUse the left panel to record a sound.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
            empty_label.setFont(empty_font)
            empty_label.setStyleSheet(f"color: {theme_manager.text_colors.medium};")
            self.sounds_list_layout.addWidget(empty_label)
        else:
            for sound_name in sorted(self.sounds_list):
                # Create item widget
                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(5, 5, 5, 5)
                item_layout.setSpacing(10)

                # Sound name label
                name_label = QLabel(sound_name)
                name_font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
                name_label.setFont(name_font)
                name_label.setStyleSheet(f"color: {theme_manager.text_colors.light};")
                item_layout.addWidget(name_label, 1)

                # Map button
                map_btn = PrimaryButton(text="Map")
                map_btn.clicked.connect(lambda checked, s=sound_name: self._on_map_sound(s))
                item_layout.addWidget(map_btn)

                # Delete button
                delete_btn = DangerButton(text="Delete")
                delete_btn.clicked.connect(lambda checked, s=sound_name: self._on_delete_sound(s))
                item_layout.addWidget(delete_btn)

                # Style the item
                item_widget.setStyleSheet(
                    f"""
                    QWidget {{
                        background-color: {theme_manager.shape_colors.dark};
                        border: 1px solid {theme_manager.shape_colors.medium};
                        border-radius: {theme_manager.border_radius.small}px;
                    }}
                """
                )

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
        """Handle start training button clicked."""
        sound_name = self.sound_name_input.text().strip()

        if not sound_name:
            QMessageBox.warning(self, "Invalid Input", "Please enter a sound name.")
            return

        if not self.controller:
            QMessageBox.critical(self, "Error", "Controller not initialized.")
            return

        num_samples = self.samples_spinbox.value()
        self.controller.train_sound(sound_name, num_samples)

    def _on_delete_all_clicked(self) -> None:
        """Handle delete all button clicked."""
        if not self.sounds_list:
            QMessageBox.information(self, "No Sounds", "There are no sounds to delete.")
            return

        if self.controller:
            reply = QMessageBox.question(
                self,
                "Delete All Sounds",
                f"Delete all {len(self.sounds_list)} trained sounds?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.controller.delete_all_sounds()

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
