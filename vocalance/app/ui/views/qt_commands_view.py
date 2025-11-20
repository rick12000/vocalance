"""Qt-based commands management view - FULLY INTEGRATED WITH COMMAND EDIT DIALOG.

Displays commands grouped by category with edit/delete capabilities.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QScrollArea, QVBoxLayout, QWidget

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.ui.qt_theme import theme_manager
from vocalance.app.ui.views.components.qt_themed_components import DangerButton, PrimaryButton, ThemedFrame, TwoColumnTabLayout


class CommandEditDialog(QDialog):
    """Dialog for editing command phrases matching legacy design."""

    def __init__(self, command: AutomationCommand, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.command = command
        self.result_action = None
        self.new_phrase_value = None

        self.setWindowTitle(f"Edit Command: {command.command_key}")
        self.setModal(True)
        self.setMinimumWidth(500)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Description tile
        desc_frame = ThemedFrame()
        desc_layout = QVBoxLayout(desc_frame)
        desc_layout.setContentsMargins(15, 15, 15, 15)
        desc_layout.setSpacing(5)

        desc_title = QLabel("Description")
        desc_title_font = theme_manager.get_font(size=theme_manager.font_sizes.medium, weight="bold")
        desc_title.setFont(desc_title_font)
        desc_title.setStyleSheet(f"color: {theme_manager.text_colors.light};")
        desc_layout.addWidget(desc_title)

        desc_text = self._get_command_description()
        desc_label = QLabel(desc_text)
        desc_font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        desc_label.setFont(desc_font)
        desc_label.setStyleSheet(f"color: {theme_manager.text_colors.light};")
        desc_label.setWordWrap(True)
        desc_layout.addWidget(desc_label)

        layout.addWidget(desc_frame)

        # Edit tile
        edit_frame = ThemedFrame()
        edit_layout = QVBoxLayout(edit_frame)
        edit_layout.setContentsMargins(15, 15, 15, 15)
        edit_layout.setSpacing(10)

        edit_title = QLabel("Edit Command Phrase")
        edit_title.setFont(desc_title_font)
        edit_title.setStyleSheet(f"color: {theme_manager.text_colors.light};")
        edit_layout.addWidget(edit_title)

        self.entry = QLineEdit()
        self.entry.setText(self.command.command_key)
        self.entry.selectAll()
        edit_layout.addWidget(self.entry)

        save_btn = PrimaryButton(text="Save Changes")
        save_btn.clicked.connect(self._on_save)
        edit_layout.addWidget(save_btn)

        layout.addWidget(edit_frame)

        # Delete tile
        delete_frame = ThemedFrame()
        delete_layout = QVBoxLayout(delete_frame)
        delete_layout.setContentsMargins(15, 15, 15, 15)
        delete_layout.setSpacing(10)

        delete_title = QLabel("Delete Command")
        delete_title.setFont(desc_title_font)
        delete_title.setStyleSheet(f"color: {theme_manager.text_colors.light};")
        delete_layout.addWidget(delete_title)

        if self.command.is_custom:
            delete_desc = QLabel("This is a custom command and can be safely deleted.")
            delete_desc.setFont(desc_font)
            delete_desc.setStyleSheet(f"color: {theme_manager.text_colors.light};")
            delete_desc.setWordWrap(True)
            delete_layout.addWidget(delete_desc)

            delete_btn = DangerButton(text="Delete Command")
            delete_btn.clicked.connect(self._on_delete)
            delete_layout.addWidget(delete_btn)
        else:
            delete_desc = QLabel("This is a built-in command and cannot be deleted.")
            delete_desc.setFont(desc_font)
            delete_desc.setStyleSheet(f"color: {theme_manager.text_colors.light};")
            delete_desc.setWordWrap(True)
            delete_layout.addWidget(delete_desc)

        layout.addWidget(delete_frame)

        # Focus entry field
        self.entry.setFocus()

    def _get_command_description(self) -> str:
        """Get a detailed description of what the command does."""
        if self.command.long_description:
            return self.command.long_description
        else:
            # Fallback to generating description based on action type
            if self.command.action_type == "hotkey":
                return f"Triggers hotkey: {self.command.action_value or 'Not set'}"
            elif self.command.action_type == "key":
                return f"Simulates pressing the key: {self.command.action_value or 'Not set'}"
            elif self.command.action_type == "key_sequence":
                return f"Executes key sequence: {self.command.action_value or 'Not set'}"
            elif self.command.action_type == "click":
                return f"Performs a mouse click action: {self.command.action_value or 'Left click'}"
            elif self.command.action_type == "scroll":
                return f"Performs a scroll action: {self.command.action_value or 'Scroll'}"
            elif self.command.action_type == "type":
                return f"Types the text: {self.command.action_value or 'No text set'}"
            else:
                return f"Custom action: {self.command.action_value or 'No action defined'}"

    def _on_save(self) -> None:
        """Handle save button click."""
        new_phrase = self.entry.text().strip()
        if new_phrase and new_phrase != self.command.command_key:
            self.result_action = "save"
            self.new_phrase_value = new_phrase
            self.accept()
        else:
            self.reject()

    def _on_delete(self) -> None:
        """Handle delete button click."""
        if not self.command.is_custom:
            return

        self.result_action = "delete"
        self.new_phrase_value = None
        self.accept()

    def get_result(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the dialog result."""
        return self.result_action, self.new_phrase_value


class QtCommandsView(QWidget):
    """Qt-based commands management view - FULLY INTEGRATED.

    Features:
    - Add command form (command phrase + hotkey)
    - Commands list grouped by functional_group
    - Change button for each command (opens edit dialog)
    - Edit dialog with description, phrase edit, and delete
    - Reset to defaults button
    - Real-time updates from controller
    """

    GROUP_ORDER = ["Basic", "Window Navigation", "Editing", "General IDE", "Cursor IDE", "VSCode IDE", "Other", "Custom"]

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize commands view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
        self.commands_list: List[AutomationCommand] = []

        self._setup_ui()
        self.logger.debug("QtCommandsView initialized")

    def set_controller(self, controller) -> None:
        """Set the controller and connect signals."""
        self.controller = controller

        # Connect controller signals
        self.controller.commands_loaded.connect(self._on_commands_loaded)
        self.controller.command_created.connect(self._on_command_created)
        self.controller.command_updated.connect(self._on_command_updated)
        self.controller.command_deleted.connect(self._on_command_deleted)
        self.controller.validation_error.connect(self._on_validation_error)
        self.controller.operation_error.connect(self._on_error)

        # Load initial commands
        self.logger.info("Loading commands from controller")
        self.controller.on_view_ready()

    def _setup_ui(self) -> None:
        """Build UI with TwoColumnTabLayout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create two-column layout with titles
        self.layout = TwoColumnTabLayout(self, "Add Command", "Manage Commands")
        main_layout.addWidget(self.layout)

        # Setup panels
        self._setup_add_command_form()
        self._setup_commands_list_panel()

    def _setup_add_command_form(self) -> None:
        """Setup add command form in left content area."""
        container = self.layout.left_content

        # Get existing layout
        container_layout = container.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(container)

        container_layout.setContentsMargins(
            theme_manager.two_box_layout.inner_content_padx,
            0,
            theme_manager.two_box_layout.inner_content_padx,
            0,
        )
        container_layout.setSpacing(theme_manager.spacing.medium)

        # Command phrase input
        command_phrase_label = QLabel("Command Phrase:")
        command_phrase_label.setStyleSheet("border: none; background: transparent;")
        container_layout.addWidget(command_phrase_label)
        self.command_phrase_entry = QLineEdit()
        self.command_phrase_entry.setPlaceholderText("Enter command phrase...")
        container_layout.addWidget(self.command_phrase_entry)

        # Hotkey input
        hotkey_label = QLabel("Hotkey:")
        hotkey_label.setStyleSheet("border: none; background: transparent;")
        container_layout.addWidget(hotkey_label)
        self.hotkey_entry = QLineEdit()
        self.hotkey_entry.setPlaceholderText("e.g. ctrl+alt+7")
        container_layout.addWidget(self.hotkey_entry)

        # Add button
        self.add_btn = PrimaryButton(text="Add Command")
        self.add_btn.clicked.connect(self._on_add_command_clicked)
        container_layout.addWidget(self.add_btn)

        container_layout.addStretch()

    def _setup_commands_list_panel(self) -> None:
        """Setup commands list panel in right content area."""
        container = self.layout.right_content

        # Get existing layout
        container_layout = container.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(container)

        container_layout.setContentsMargins(
            theme_manager.two_box_layout.inner_content_padx,
            0,
            theme_manager.two_box_layout.inner_content_padx,
            0,
        )
        container_layout.setSpacing(theme_manager.spacing.small)

        # Commands list widget (will show grouped commands)
        self.commands_list_widget = QWidget()
        self.commands_list_widget.setStyleSheet("background: transparent;")
        self.commands_list_layout = QVBoxLayout(self.commands_list_widget)
        self.commands_list_layout.setSpacing(theme_manager.spacing.tiny)
        self.commands_list_layout.setContentsMargins(0, 0, 0, 0)
        self.commands_list_layout.addStretch()

        # Scroll area for commands
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent; border: none;")
        scroll_area.setWidget(self.commands_list_widget)
        container_layout.addWidget(scroll_area)

        # Reset to defaults button
        self.reset_btn = DangerButton(text="Reset to Defaults")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        container_layout.addWidget(self.reset_btn)

    def _on_commands_loaded(self, commands: List[AutomationCommand]) -> None:
        """Handle commands loaded from controller."""
        try:
            self.commands_list = commands
            self._display_commands(commands)
            self.logger.info(f"Commands loaded: {len(commands)} total")
        except Exception as e:
            self.logger.error(f"Error loading commands: {e}", exc_info=True)
            self._show_error(f"Error loading commands: {e}")

    def _on_command_created(self, command_phrase: str) -> None:
        """Handle command created event."""
        try:
            # Clear form
            self.command_phrase_entry.clear()
            self.hotkey_entry.clear()
            self.logger.info(f"Command created: {command_phrase}")
        except Exception as e:
            self.logger.error(f"Error handling command created: {e}", exc_info=True)

    def _on_command_updated(self, old_phrase: str, new_phrase: str) -> None:
        """Handle command updated event."""
        try:
            self.logger.info(f"Command updated: {old_phrase} -> {new_phrase}")
        except Exception as e:
            self.logger.error(f"Error handling command updated: {e}", exc_info=True)

    def _on_command_deleted(self, command_phrase: str) -> None:
        """Handle command deleted event."""
        try:
            self.logger.info(f"Command deleted: {command_phrase}")
        except Exception as e:
            self.logger.error(f"Error handling command deleted: {e}", exc_info=True)

    def _on_validation_error(self, error_msg: str, command_phrase: str) -> None:
        """Handle validation error from controller."""
        self._show_error(f"Validation error: {error_msg}")

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._show_error(error_msg)

    def _display_commands(self, commands: List[AutomationCommand]) -> None:
        """Display commands in a grouped list."""
        # Clear existing items
        while self.commands_list_layout.count():
            item = self.commands_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not commands:
            # Show empty message
            empty_label = QLabel("No commands available.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
            empty_label.setFont(empty_font)
            empty_label.setStyleSheet(f"color: {theme_manager.text_colors.medium};")
            self.commands_list_layout.addWidget(empty_label)
        else:
            grouped_commands = self._group_commands(commands)
            sorted_groups = self._sort_groups(grouped_commands)

            for group_name, group_commands in sorted_groups:
                # Group header
                group_label = QLabel(group_name)
                group_font = theme_manager.get_font(size=theme_manager.font_sizes.medium, weight="bold")
                group_label.setFont(group_font)
                group_label.setStyleSheet(f"color: {theme_manager.text_colors.light}; padding: 5px;")
                self.commands_list_layout.addWidget(group_label)

                # Divider
                divider = QWidget()
                divider.setFixedHeight(1)
                divider.setStyleSheet(f"background-color: {theme_manager.shape_colors.medium};")
                self.commands_list_layout.addWidget(divider)

                # Commands in this group
                sorted_commands = sorted(group_commands, key=lambda cmd: cmd.command_key.lower())
                for command in sorted_commands:
                    self._create_command_item(command)

        self.commands_list_layout.addStretch()

    def _group_commands(self, commands: List[AutomationCommand]) -> Dict[str, List[AutomationCommand]]:
        """Group commands by their functional_group attribute."""
        grouped = defaultdict(list)
        for command in commands:
            group = getattr(command, "functional_group", "Other")
            grouped[group].append(command)
        return dict(grouped)

    def _sort_groups(self, grouped_commands: Dict[str, List[AutomationCommand]]) -> List[tuple]:
        """Sort groups according to GROUP_ORDER."""
        sorted_groups = []

        for group_name in self.GROUP_ORDER:
            if group_name in grouped_commands:
                sorted_groups.append((group_name, grouped_commands[group_name]))

        for group_name in sorted(grouped_commands.keys()):
            if group_name not in self.GROUP_ORDER:
                sorted_groups.append((group_name, grouped_commands[group_name]))

        return sorted_groups

    def _create_command_item(self, command: AutomationCommand) -> None:
        """Create a command list item with phrase and change button."""
        item_widget = QWidget()
        item_widget.setProperty("itemType", "list_item")
        item_widget.setStyleSheet("background: transparent; border: none;")
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(theme_manager.spacing.small)

        # Command phrase label
        phrase_label = QLabel(command.command_key)
        phrase_font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        phrase_label.setFont(phrase_font)
        phrase_label.setStyleSheet(f"color: {theme_manager.text_colors.light}; background: transparent; border: none;")
        item_layout.addWidget(phrase_label, 1)

        # Change button (pill-shaped)
        change_btn = PrimaryButton(text="Change")
        change_btn.setFixedWidth(90)
        change_btn.clicked.connect(lambda checked, c=command: self._on_change_command(c))
        item_layout.addWidget(change_btn)

        self.commands_list_layout.insertWidget(self.commands_list_layout.count() - 1, item_widget)

    def _on_change_command(self, command: AutomationCommand) -> None:
        """Handle change button clicked - show edit dialog."""
        if not self.controller:
            self._show_error("Controller not initialized.")
            return

        dialog = CommandEditDialog(command, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            action, new_phrase = dialog.get_result()

            if action == "save" and new_phrase:
                self.controller.handle_change_command_phrase(command, new_phrase)
                self.logger.info(f"Changed command phrase: {command.command_key} -> {new_phrase}")
            elif action == "delete":
                self.controller.handle_delete_command(command)
                self.logger.info(f"Deleted command: {command.command_key}")

    def _on_add_command_clicked(self) -> None:
        """Handle add command button clicked."""
        command_phrase = self.command_phrase_entry.text().strip()
        hotkey_value = self.hotkey_entry.text().strip()

        if not command_phrase:
            QMessageBox.warning(self, "Invalid Input", "Please enter a command phrase.")
            return

        if not hotkey_value:
            QMessageBox.warning(self, "Invalid Input", "Please enter a hotkey value.")
            return

        if not self.controller:
            QMessageBox.critical(self, "Error", "Controller not initialized.")
            return

        self.controller.handle_add_command(command_phrase, hotkey_value)

    def _on_reset_clicked(self) -> None:
        """Handle reset to defaults button clicked."""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all commands to defaults? This will remove all custom commands.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.controller:
                self.controller.handle_reset_to_defaults()

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
