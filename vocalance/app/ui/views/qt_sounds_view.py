"""Qt-based sounds training view.

Displays trained sounds and allows training new sounds with command mapping using TwoColumnTabLayout.
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
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from vocalance.app.ui.components.atoms import Button
from vocalance.app.ui.components.complex import TwoColumnLayout
from vocalance.app.ui.components.containers import BaseContainer
from vocalance.app.ui.qt_theme import theme


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
        current_frame = BaseContainer()
        # Get the layout created by BaseContainer
        current_layout = current_frame.layout()
        if current_layout is None:
            current_layout = QVBoxLayout(current_frame)
        current_layout.setContentsMargins(15, 15, 15, 15)
        current_layout.setSpacing(5)

        current_title = QLabel("Current Mapping:")
        current_title_font = theme.get_font(size=theme.config.fonts.medium, weight="semibold")
        current_title.setFont(current_title_font)
        current_title.setStyleSheet(f"color: {theme.config.text.light};")
        current_layout.addWidget(current_title)

        # Get current mapping
        try:
            mapped_command = self.controller.get_sound_command_mapping(self.sound_name)
            mapping_text = mapped_command if mapped_command else "Unmapped"
        except Exception:
            mapping_text = "Unmapped"

        current_label = QLabel(mapping_text)
        current_font = theme.get_font(size=theme.config.fonts.medium)
        current_label.setFont(current_font)
        current_label.setStyleSheet(f"color: {theme.config.text.medium};")
        current_layout.addWidget(current_label)

        layout.addWidget(current_frame)

        # Main mapping frame
        mapping_frame = BaseContainer()
        # Get the layout created by BaseContainer
        mapping_layout = mapping_frame.layout()
        if mapping_layout is None:
            mapping_layout = QVBoxLayout(mapping_frame)
        mapping_layout.setContentsMargins(15, 15, 15, 15)
        mapping_layout.setSpacing(10)

        # Command Type dropdown
        type_label = QLabel("Command Type:")
        type_label_font = theme.get_font(size=theme.config.fonts.medium, weight="semibold")
        type_label.setFont(type_label_font)
        type_label.setStyleSheet(f"color: {theme.config.text.light};")
        mapping_layout.addWidget(type_label)

        self.type_combo = QComboBox()
        command_types = self.controller.get_mapping_command_types()
        self.type_combo.addItems(command_types)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        mapping_layout.addWidget(self.type_combo)

        # Command Value dropdown
        value_label = QLabel("Command Value:")
        value_label.setFont(type_label_font)
        value_label.setStyleSheet(f"color: {theme.config.text.light};")
        mapping_layout.addWidget(value_label)

        self.value_combo = QComboBox()
        mapping_layout.addWidget(self.value_combo)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        confirm_btn = Button(text="Confirm", variant="primary")
        confirm_btn.clicked.connect(self._on_confirm)
        button_layout.addWidget(confirm_btn)

        cancel_btn = Button(text="Cancel", variant="danger")
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
    """Qt-based sounds training view using TwoColumnTabLayout.

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
        """Build UI with TwoColumnTabLayout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create two-column layout with titles
        self.layout = TwoColumnLayout("Train Sound", "Trained Sounds", self)
        main_layout.addWidget(self.layout)

        # Setup panels
        self._setup_training_form()
        self._setup_sounds_list_panel()

    def _setup_training_form(self) -> None:
        """Setup training form in left content area."""
        container = self.layout.left_content

        # Create layout if it doesn't exist
        container_layout = container.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(container)

        container_layout.setContentsMargins(
            theme.config.dims.inner_content_padx,
            0,
            theme.config.dims.inner_content_padx,
            0,
        )
        container_layout.setSpacing(theme.config.spacing.medium)

        # Sound name input
        sound_name_label = QLabel("Sound Name:")
        sound_name_label.setStyleSheet("border: none; background: transparent;")
        container_layout.addWidget(sound_name_label)
        self.sound_name_input = QLineEdit()
        self.sound_name_input.setPlaceholderText("e.g., doorbell")
        self.sound_name_input.setMaxLength(50)
        container_layout.addWidget(self.sound_name_input)

        # Number of samples
        samples_label = QLabel("Samples:")
        samples_label.setStyleSheet("border: none; background: transparent;")
        container_layout.addWidget(samples_label)
        self.samples_spinbox = QLineEdit()
        default_samples = self.controller.get_default_training_samples() if self.controller else 5
        self.samples_spinbox.setText(str(default_samples))
        self.samples_spinbox.setPlaceholderText("e.g., 5")
        container_layout.addWidget(self.samples_spinbox)

        # Start training button
        self.start_training_btn = Button(text="Record", variant="primary")
        self.start_training_btn.clicked.connect(self._on_start_training_clicked)
        container_layout.addWidget(self.start_training_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        container_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        container_layout.addWidget(self.status_label)

        container_layout.addStretch()

    def _setup_sounds_list_panel(self) -> None:
        """Setup sounds list panel in right content area."""
        container = self.layout.right_content

        # Create layout if it doesn't exist
        container_layout = container.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(container)

        container_layout.setContentsMargins(
            theme.config.dims.inner_content_padx,
            0,
            theme.config.dims.inner_content_padx,
            0,
        )
        container_layout.setSpacing(theme.config.spacing.small)

        # Sounds list widget
        self.sounds_list_widget = QWidget()
        self.sounds_list_widget.setStyleSheet("background: transparent;")
        self.sounds_list_layout = QVBoxLayout(self.sounds_list_widget)
        self.sounds_list_layout.setSpacing(theme.config.spacing.tiny)
        self.sounds_list_layout.setContentsMargins(0, 0, 0, 0)
        self.sounds_list_layout.addStretch()

        # Scroll area for sounds
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent; border: none;")
        scroll_area.setWidget(self.sounds_list_widget)
        container_layout.addWidget(scroll_area)

        # Delete all button
        self.delete_all_btn = Button(text="Reset", variant="danger")
        self.delete_all_btn.clicked.connect(self._on_delete_all_clicked)
        container_layout.addWidget(self.delete_all_btn)
        # Add bottom padding after button
        container_layout.addSpacing(theme.config.spacing.large)

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
            self.status_label.setVisible(True)
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
            self.status_label.setVisible(True)
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
            self.status_label.setText(f"Training failed: {error_msg}")
            self.status_label.setVisible(True)
            self.start_training_btn.setEnabled(True)
            self.sound_name_input.setEnabled(True)
            self.samples_spinbox.setEnabled(True)
            self.current_training_sound = None

            self.logger.error(f"Training error for {sound_name}: {error_msg}")
        except Exception as e:
            self.logger.error(f"Error handling training error: {e}", exc_info=True)

    def _on_sound_deleted(self, sound_name: str) -> None:
        """Handle sound deleted event."""
        try:
            if self.controller:
                self.controller.refresh_sound_list()
            self.logger.info(f"Sound deleted: {sound_name}")
        except Exception as e:
            self.logger.error(f"Error handling sound deleted: {e}", exc_info=True)

    def _on_error(self, error_message: str) -> None:
        """Handle error from controller."""
        self.logger.error(f"Controller error: {error_message}")
        self._show_error(error_message)

    def _refresh_sounds_list(self) -> None:
        """Refresh the display of trained sounds."""
        # Clear existing widgets (keep the stretch at the end)
        while self.sounds_list_layout.count() > 1:  # Keep the stretch
            item = self.sounds_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add sound widgets
        if not self.sounds_list:
            # Show empty message
            empty_label = QLabel("No available sounds.\nUse the left panel to record a sound.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_font = theme.get_font(size=theme.config.fonts.medium)
            empty_label.setFont(empty_font)
            empty_label.setStyleSheet(f"color: {theme.config.text.medium}; background: transparent; border: none;")
            self.sounds_list_layout.insertWidget(0, empty_label)
        else:
            for sound_name in sorted(self.sounds_list):
                # Create item widget
                item_widget = QWidget()
                item_widget.setProperty("itemType", "list_item")
                item_widget.setStyleSheet("background: transparent; border: none;")
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(0, 0, 0, 0)
                item_layout.setSpacing(theme.config.spacing.small)

                # Sound name label
                name_label = QLabel(sound_name)
                name_font = theme.get_font(size=theme.config.fonts.medium)
                name_label.setFont(name_font)
                name_label.setStyleSheet(f"color: {theme.config.text.medium}; background: transparent; border: none;")
                item_layout.addWidget(name_label, 1)

                # Map button (pill-shaped)
                map_btn = Button(text="Map", variant="primary")
                map_btn.setFixedWidth(80)
                map_btn.clicked.connect(lambda checked, s=sound_name: self._on_map_sound(s))
                item_layout.addWidget(map_btn)

                # Delete button (pill-shaped)
                delete_btn = Button(text="Delete", variant="danger")
                delete_btn.setFixedWidth(80)
                delete_btn.clicked.connect(lambda checked, s=sound_name: self._on_delete_sound(s))
                item_layout.addWidget(delete_btn)

                self.sounds_list_layout.insertWidget(self.sounds_list_layout.count() - 1, item_widget)

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
