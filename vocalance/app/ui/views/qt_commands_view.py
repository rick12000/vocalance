from collections import defaultdict
from typing import Dict, List, Optional

from PySide6.QtWidgets import QDialog, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.ui.components.buttons import ChangeButton, DangerButton, PrimaryButton
from vocalance.app.ui.components.dialogs import CommandEditDialog
from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BodyLabel, SmallLabel
from vocalance.app.ui.components.layouts import (
    CollapsibleSection,
    ListItem,
    ScrollableContainer,
    TransparentWidget,
    TwoColumnLayout,
)
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.views.qt_base_view import QtBaseView


class QtCommandsView(QtBaseView):
    """Qt-based commands management view.

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
        """Setup add command form in left content area."""
        content = self.layout.left_content

        # Command phrase input
        command_phrase_label = SmallLabel("Command Phrase:")
        content.add(command_phrase_label)
        self.command_phrase_entry = TextInput(placeholder="Enter command phrase...")
        content.add(self.command_phrase_entry)

        # Hotkey input
        hotkey_label = SmallLabel("Hotkey:")
        content.add(hotkey_label)
        self.hotkey_entry = TextInput(placeholder="e.g. ctrl+alt+7")
        content.add(self.hotkey_entry)

        # Add button
        self.add_btn = PrimaryButton(text="Add")
        self.add_btn.clicked.connect(self._on_add_command_clicked)
        content.add(self.add_btn)

        content.add_stretch()

    def _setup_commands_list_panel(self) -> None:
        """Setup commands list panel."""
        content = self.layout.right_content

        # Commands list widget
        self.commands_list_widget = TransparentWidget()
        self.commands_list_layout = QVBoxLayout(self.commands_list_widget)
        self.commands_list_layout.setSpacing(theme.config.container.list_item_spacing)
        self.commands_list_layout.setContentsMargins(0, 0, theme.config.spacing.small, 0)

        # Scroll area for commands
        scroll_area = ScrollableContainer()
        scroll_area.content_layout.addWidget(self.commands_list_widget)
        content.add(scroll_area, stretch=1)

        # Reset to defaults button
        self.reset_btn = DangerButton(text="Reset")
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
            self.command_phrase_entry.clear()
            self.hotkey_entry.clear()
            self.logger.info(f"Command created: {command_phrase}")
        except Exception as e:
            self.logger.error(f"Error handling command created: {e}", exc_info=True)

    def _on_command_updated(self, old_phrase: str, new_phrase: str) -> None:
        """Handle command updated event."""
        self.logger.info(f"Command updated: {old_phrase} -> {new_phrase}")

    def _on_command_deleted(self, command_phrase: str) -> None:
        """Handle command deleted event."""
        self.logger.info(f"Command deleted: {command_phrase}")

    def _on_validation_error(self, error_msg: str, command_phrase: str) -> None:
        """Handle validation error from controller."""
        self._show_error(f"Validation error: {error_msg}")

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._show_error(error_msg)

    def _display_commands(self, commands: List[AutomationCommand]) -> None:
        """Display commands in a grouped list with collapsible sections."""
        # Clear existing items
        while self.commands_list_layout.count() > 0:
            item = self.commands_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not commands:
            empty_label = BodyLabel("No commands available.", align="center", color=theme.config.text.medium)
            self.commands_list_layout.addWidget(empty_label)
        else:
            grouped_commands = self._group_commands(commands)
            sorted_groups = self._sort_groups(grouped_commands)

            for group_index, (group_name, group_commands) in enumerate(sorted_groups):
                is_first = group_index == 0
                # Create collapsible section (starts collapsed by default)
                collapsible_section = CollapsibleSection(group_name, is_first=is_first, start_expanded=False)

                # Add commands to the section
                sorted_commands = sorted(group_commands, key=lambda cmd: cmd.command_key.lower())
                for command in sorted_commands:
                    command_item = self._create_command_item_widget(command)
                    collapsible_section.add_item(command_item)

                self.commands_list_layout.addWidget(collapsible_section)

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

    def _create_command_item_widget(self, command: AutomationCommand) -> QWidget:
        """Create a command list item widget."""
        item = ListItem()

        phrase_label = SmallLabel(command.command_key, color=theme.config.text.medium)
        item.add(phrase_label, stretch=1)

        change_btn = ChangeButton(command=lambda checked, c=command: self._on_change_command(c))
        item.add(change_btn)

        return item

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
