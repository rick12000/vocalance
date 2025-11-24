"""Qt-based commands management view - FULLY INTEGRATED WITH COMMAND EDIT DIALOG.

Displays commands grouped by category with edit/delete capabilities.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import QDialog, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.ui.components.complex_components import GroupHeader, ListItem
from vocalance.app.ui.components.layouts import Box, ScrollableContainer, TransparentWidget, TwoColumnLayout
from vocalance.app.ui.components.simple_components import Button, Input, Label
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.views.qt_base_view import QtBaseView


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
        desc_frame = Box()
        desc_frame._layout

        desc_title = Label("Description", variant="subtitle")
        desc_frame.add(desc_title)

        desc_text = self._get_command_description()
        desc_label = Label(desc_text, variant="body")
        desc_label.setWordWrap(True)
        desc_frame.add(desc_label)

        layout.addWidget(desc_frame)

        # Edit tile
        edit_frame = Box()

        edit_title = Label("Edit Command Phrase", variant="subtitle")
        edit_frame.add(edit_title)

        self.entry = Input()
        self.entry.setText(self.command.command_key)
        self.entry.selectAll()
        edit_frame.add(self.entry)

        save_btn = Button(text="Save Changes", variant="primary")
        save_btn.clicked.connect(self._on_save)
        edit_frame.add(save_btn)

        layout.addWidget(edit_frame)

        # Delete tile
        delete_frame = Box()

        delete_title = Label("Delete Command", variant="subtitle")
        delete_frame.add(delete_title)

        if self.command.is_custom:
            delete_desc = Label("This is a custom command and can be safely deleted.", variant="body")
            delete_desc.setWordWrap(True)
            delete_frame.add(delete_desc)

            delete_btn = Button(text="Delete Command", variant="danger")
            delete_btn.clicked.connect(self._on_delete)
            delete_frame.add(delete_btn)
        else:
            delete_desc = Label("This is a built-in command and cannot be deleted.", variant="body")
            delete_desc.setWordWrap(True)
            delete_frame.add(delete_desc)

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


class QtCommandsView(QtBaseView):
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
        """Build UI with TwoColumnLayout."""
        # Create two-column layout with titles
        self.layout = TwoColumnLayout("Add Command", "Manage Commands", self)
        self.main_layout.addWidget(self.layout, stretch=1)

        # Setup panels
        self._setup_add_command_form()
        self._setup_commands_list_panel()

    def _setup_add_command_form(self) -> None:
        """Setup add command form in left content area using systematic spacing."""
        content = self.layout.left_content

        # Command phrase input
        command_phrase_label = Label("Command Phrase:", variant="small")
        content.add(command_phrase_label)
        self.command_phrase_entry = Input(placeholder="Enter command phrase...")
        content.add(self.command_phrase_entry)

        # Hotkey input
        hotkey_label = Label("Hotkey:", variant="small")
        content.add(hotkey_label)
        self.hotkey_entry = Input(placeholder="e.g. ctrl+alt+7")
        content.add(self.hotkey_entry)

        # Add button
        self.add_btn = Button(text="Add", variant="primary")
        self.add_btn.clicked.connect(self._on_add_command_clicked)
        content.add(self.add_btn)

        content.add_stretch()

    def _setup_commands_list_panel(self) -> None:
        """Setup commands list panel using systematic spacing."""
        content = self.layout.right_content

        # Commands list widget with guaranteed transparent background
        # Use TransparentWidget to prevent stylesheet and palette interference
        self.commands_list_widget = TransparentWidget()
        self.commands_list_layout = QVBoxLayout(self.commands_list_widget)
        self.commands_list_layout.setSpacing(theme.config.container.list_item_spacing)
        self.commands_list_layout.setContentsMargins(0, 0, theme.config.spacing.small, 0)

        # Scroll area for commands
        scroll_area = ScrollableContainer()
        scroll_area.content_layout.addWidget(self.commands_list_widget)
        content.add(scroll_area, stretch=1)

        # Reset to defaults button
        self.reset_btn = Button(text="Reset", variant="danger")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        content.add(self.reset_btn)

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
        """Display commands in a grouped list using systematic spacing."""
        # Clear existing items
        while self.commands_list_layout.count() > 0:
            item = self.commands_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not commands:
            # Show empty message
            empty_label = Label("No commands available.", variant="body", color=theme.config.text.medium, align="center")
            self.commands_list_layout.addWidget(empty_label)
        else:
            grouped_commands = self._group_commands(commands)
            sorted_groups = self._sort_groups(grouped_commands)

            for group_index, (group_name, group_commands) in enumerate(sorted_groups):
                # Group header using systematic spacing
                is_first = group_index == 0
                group_header = GroupHeader(group_name, is_first=is_first)
                self.commands_list_layout.addWidget(group_header)

                # Commands in this group
                sorted_commands = sorted(group_commands, key=lambda cmd: cmd.command_key.lower())
                for command in sorted_commands:
                    self._create_command_item_inline(command)

        # Add stretch at the end
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

    def _create_command_item_inline(self, command: AutomationCommand) -> None:
        """Create a command list item using systematic spacing."""
        # Create list item with systematic spacing
        item = ListItem()

        # Command phrase label - smaller font
        phrase_label = Label(command.command_key, variant="small", color=theme.config.text.lightest)
        item.add(phrase_label, stretch=1)

        # Change button
        change_btn = Button(text="Change", variant="primary")
        change_btn.setFixedWidth(90)
        change_btn.clicked.connect(lambda checked, c=command: self._on_change_command(c))
        item.add(change_btn)

        # Add to layout
        self.commands_list_layout.addWidget(item)

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
            "Reset",
            "Are you sure you want to reset all commands to defaults? This will remove all custom commands.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.controller:
                self.controller.handle_reset_to_defaults()

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
